// File         : Turtlebot3Robot.cpp
// Date         : 4 Sept 2024
// Description  : Turtlebot3 CPP Controller
// Author       :  DarkStealthX
// Modifications: - Communication using zeromq
//                - Publish state and subscribe start and step message

#include <webots/Supervisor.hpp>
#include <webots/utils/AnsiCodes.hpp>
#include <webots/Lidar.hpp>
#include <webots/Motor.hpp>

#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm> // for std::min and std::max

#include <stdarg.h>
#include <string.h>
#include <span>

#include <optional>
#include <vector>
#include <cmath>

#include <thread>
#include "library/Agent_Subscriber.hpp"
#include "library/Robo_Publisher.hpp"

// for ROS2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2/exceptions.h"
#include <builtin_interfaces/msg/time.hpp>

#define COPY_FIELD(Src, Dest, Field) (Dest.Field = Src.Field)

using namespace webots;
using namespace std;
// using namespace cv;

#define LIDAR_STATE 10

void logger(string mode,
            string head_name,
            const char *format, ...)
{

    va_list args;
    va_start(args, format);

    string head;
    if (mode.starts_with("info") == 1)
    {
        head = AnsiCodes::BLUE_BACKGROUND;
    }
    else if (mode.starts_with("warn") == 1)
    {
        head = AnsiCodes::YELLOW_BACKGROUND;
    }
    else if (mode.starts_with("err") == 1)
    {
        head = AnsiCodes::RED_BACKGROUND;
    }
    else if (mode.starts_with("custom") == 1)
    {
        head = AnsiCodes::MAGENTA_BACKGROUND;
    }
    else if (mode.starts_with("hidden") == 1)
    {
        head = AnsiCodes::GREEN_BACKGROUND;
    }

    // print formatted data using vsnprintf
    char buffer[1024], data[1024];
    long int ret = vsnprintf(buffer, sizeof(buffer), format, args);
    if (ret >= 0 && ret < (int)sizeof(buffer))
    {
        strcpy(data, buffer);
    }
    else
    {
        strcpy(data, (char *)"err");
    }

    if (head_name.empty())
    {
        cout << head << data << AnsiCodes::RESET << "\r\n";
    }
    else
    {
        int tab_times = head_name.length() > 10 ? 1 : 2;
        string tabs = string(tab_times, '\t');
        cout << head << head_name << AnsiCodes::RESET << tabs << data << endl;
    }
}

auto rounding = [](double data) -> double
{
    return std::round(data * 1000.0f) / 1000.0f;
};

auto rounding2Dec = [](double data) -> double
{
    return std::round(data * 100.0f) / 100.0f;
};

class Turtlebot3_Env : public Supervisor, public rclcpp::Node
{
private:
    // webots variable
    Lidar *lidar;
    Motor *lidar_motor[2];
    Motor *motors[2];

    // lidar variable
    float *raw_lidar_data;
    LidarPoint *point_cloud_data;
    int lidar_width;
    double lidar_minrange;
    double lidar_maxrange;
    int lidar_num_state = 10;
    std::vector<double> last_lidar_data;
    int lidar_pushed_data = 0;
    bool lidar_data_arrived = false;

    // zmq protocol init
    Robo_Publisher robo_msg;
    Agent_Subscriber agent_msg;
    bool half_degree = true;

    // thread protocol init
    std::thread subscriberThread;
    std::thread publisherThread;
    std::thread ros_thread_;
    int state_idx = 0;

    // tb3 physical variable
    const double robot_wheelbase = 0.16;                                   // 0.16 meter
    const double robot_wheelradius = 0.033;                                // 0.033 meter
    const double max_linear_velocity = 0.22;                               // 0.22 m/s
    const double max_angular_velocity = 2.0;                               // (default) 2.84 rad/s, 1.5
    const double max_speedwheel = max_linear_velocity / robot_wheelradius; // 6.67 rad/s
    const double robot_length = 0.178;
    const double robot_width = 0.138;
    const int lidar_degree = 360;
    const int lidar_data = 360;

    // ros2 variable
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserPublisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr OdomPublisher_;
    rclcpp::TimerBase::SharedPtr pub_timer_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    // rclcpp::TimerBase::SharedPtr pub_timer_tf_;


public:
    
    struct Quaternion
    {
        double w, x, y, z;
    };

    struct robot_position
    {
        double x, y, z, angle;
    };

    struct robot_velocity
    {
        double linear_vel_x, linear_vel_y, linear_vel_z;
        double angular_vel_x, angular_vel_y, angular_vel_z;
    };

    struct target_position
    {
        double x, y, z;
    };

    struct robot_step
    {
        double linear_vel, angular_vel;
    };

    // robot counter to make sure sequencing
    int robot_counter = 0;
    int step_max = 1024;

    // webots initialization
    webots::Node *robot_node, *target_node;
    webots::Field *trans_field, *rot_field, *target_trans_field;
    robot_position robot_pos;
    target_position target_pos;
    robot_velocity robot_vel;
    robot_step robot_act;
    int time_step;


    Turtlebot3_Env() : robo_msg("tcp://127.0.0.1:5555", "tcp://127.0.0.1:6666"),
                       agent_msg("tcp://127.0.0.1:7777", "tcp://127.0.0.1:8888"),
                       subscriberThread(&Agent_Subscriber::start, &agent_msg),
                       publisherThread(&Robo_Publisher::start, &robo_msg),
                       Node("tb3_robot")
    {


        // initialize ros2 publisher
        laserPublisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
        OdomPublisher_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this); //odom to base_link


        string head_name = "TB3_Init";
        robot_node = getFromDef("TurtleBot3Burger");
        target_node = getFromDef("target_ball");
        trans_field = robot_node->getField("translation");
        target_trans_field = target_node->getField("translation");
        rot_field = robot_node->getField("rotation");

        time_step = (int)getBasicTimeStep();
        logger("info", head_name, "webots robot timestep : %d ms", time_step);

        pub_timer_ = this->create_wall_timer(std::chrono::milliseconds(250),
                                             std::bind(&Turtlebot3_Env::publishRviz, this));
        // pub_timer_tf_ = this->create_wall_timer(std::chrono::milliseconds(250),
        //                                         std::bind(&Turtlebot3_Env::publishTF, this));

        initialize_device();
    }

    ~Turtlebot3_Env()
    {
        if(ros_thread_.joinable())
        {
            ros_thread_.join(); // Ensure the spin thread is properly joined
        }
        cout << "while loop out " << endl;
        agent_msg.stop();
        cout << "agent msg stop " << endl;
        robo_msg.stop();
        cout << "robo msg stop " << endl;
        publisherThread.join();
        cout << "robot msg has been stop " << endl;
        subscriberThread.join();
        cout << "agent msg has been stop " << endl;
    }

    void initialize_device()
    {
        initialize_lidar();

        initialize_motor();
    }

    void initialize_lidar()
    {

        string head_name = "TB3_Lidar_Init";
        lidar = getLidar("LDS-01");
        // lidar = getLidar("RPlidar A2");

        // parameter is same on example turtlebot3 burger
        lidar_motor[0] = getMotor("LDS-01_main_motor");
        lidar_motor[1] = getMotor("LDS-01_secondary_motor");
        lidar_motor[0]->setPosition(std::numeric_limits<double>::infinity());
        lidar_motor[1]->setPosition(std::numeric_limits<double>::infinity());
        lidar_motor[0]->setVelocity(30.0f);
        lidar_motor[1]->setVelocity(60.0f);

        // lidar->disablePointCloud();
        lidar->enable(time_step);
        // lidar->enablePointCloud();
        lidar_width = lidar->getHorizontalResolution();
        lidar_minrange = lidar->getMinRange();
        lidar_maxrange = lidar->getMaxRange();

        cout << "lidar freq : " << (float)(lidar->getFrequency()) << endl;
        logger("info",
               head_name,
               "lidar enabled with min: %.2f m, max: %.2f m and every: %.2f deg",
               lidar_minrange,
               lidar_maxrange,
               (float)(360.0f/lidar_width));
    }

    void resetting_lidar()
    {
        lidar = getLidar("LDS-01");
        // lidar = getLidar("RPlidar A2");

        // parameter is same on example turtlebot3 burger
        lidar_motor[0] = getMotor("LDS-01_main_motor");
        lidar_motor[1] = getMotor("LDS-01_secondary_motor");
        lidar_motor[0]->setPosition(std::numeric_limits<double>::infinity());
        lidar_motor[1]->setPosition(std::numeric_limits<double>::infinity());
        lidar_motor[0]->setVelocity(30.0f);
        lidar_motor[1]->setVelocity(60.0f);

        // lidar->disablePointCloud();
        lidar->enable(time_step);
        // lidar->enablePointCloud();
        lidar_width = lidar->getHorizontalResolution();
        lidar_minrange = lidar->getMinRange();
        lidar_maxrange = lidar->getMaxRange();
    }

    void initialize_motor()
    {
        motors[0] = getMotor("left wheel motor");
        motors[1] = getMotor("right wheel motor");

        motors[0]->setPosition(std::numeric_limits<double>::infinity());
        motors[1]->setPosition(std::numeric_limits<double>::infinity());

        motors[0]->setVelocity(0.0);
        motors[1]->setVelocity(0.0);
    }

    robot_step scaled_act(robot_step data)
    {
        const float inp_l = -1.00;
        const float inp_h = 1.00;
        // const float out_linear_l = 0.075;
        const float out_linear_l = 0.000;
        robot_step process;

        process.linear_vel = (max_linear_velocity - out_linear_l) *
                                 ((data.linear_vel - inp_l) / (inp_h - inp_l)) +
                             out_linear_l;

        process.angular_vel = (max_angular_velocity - (-max_angular_velocity)) *
                                  ((data.angular_vel - inp_l) / (inp_h - inp_l)) +
                              (-max_angular_velocity);
        return process;
    }

    void set_action_robot(robot_step data)
    {
        // logger("custom", "c++", "linear_vel : %f", data.linear_vel);
        robot_step data_scaled = scaled_act(data);
        //  logger("custom", "c++", "linear_vel : %f", data_scaled.linear_vel);
        // inverse kinematics: calculate wheels speeds (rad/s) from desired linear and angular velocity (m/s, rad/s)
        double v = std::min(max_linear_velocity, (double)data_scaled.linear_vel);

        // set action robot linear_v in m/s and angular_v in rad/s,
        //    if l_v + and r_v - robot turn right angular vel maximum.
        //    if l_v - and r_v + robot turn left angular vel minimum
        double r_v = (v - (robot_wheelbase / 2) * data_scaled.angular_vel) / robot_wheelradius;
        double l_v = (v + (robot_wheelbase / 2) * data_scaled.angular_vel) / robot_wheelradius;

        // constraints
        l_v = (l_v > 0) ? std::min(max_speedwheel, l_v) : std::max(-max_speedwheel, l_v); // rad/s
        r_v = (r_v > 0) ? std::min(max_speedwheel, r_v) : std::max(-max_speedwheel, r_v); // rad/s

        motors[0]->setVelocity(l_v);
        motors[1]->setVelocity(r_v);
    }

    void read_lidar_data()
    {
        // string head_name = "TB3_Lidar_read";
        // lidar->disablePointCloud();
        raw_lidar_data = const_cast<float *>(lidar->getRangeImage());
        // lidar->enablePointCloud();

        // logger("info","C++", "lidar data depan : %.2f", raw_lidar_data[199]);
    }

    robot_position get_robot_position()
    {
        robot_position robot_pos;
        string head_name = "pos_robot";
        trans_field = robot_node->getField("translation");
        rot_field = robot_node->getField("rotation");
        double *robot_trans = const_cast<double *>(trans_field->getSFVec3f());
        double *robot_rot = const_cast<double *>(rot_field->getSFRotation());
        double angle = axis_ang_to_yaw(robot_rot);

        // logger("info",
        //        head_name,
        //        "robot pos x:%.3f y:%.3f, ang:%.3f",
        //        robot_trans[0], robot_trans[1], angle);

        robot_pos.x = robot_trans[0];
        robot_pos.y = robot_trans[1];
        robot_pos.z = robot_trans[2];
        robot_pos.angle = angle;

        return robot_pos;
    }

    robot_velocity get_robot_velocity()
    {
        robot_velocity vel;
        double *data = const_cast<double*>(robot_node->getVelocity());
        vel.linear_vel_x = data[0];
        vel.linear_vel_y = data[1];
        vel.linear_vel_z = data[2];
        vel.angular_vel_x = data[3];
        vel.angular_vel_y = data[4];
        vel.angular_vel_z = data[5];

        return vel;
    }

    target_position get_target_position()
    {
        target_position target_pos;
        string head_name = "pos_target";
        target_trans_field = target_node->getField("translation");
        double *target_trans = const_cast<double *>(target_trans_field->getSFVec3f());

        // logger("info",
        //        head_name,
        //        "robot pos x:%.3f y:%.3f, z:%.3f",
        //        target_trans[0], target_trans[1], target_trans[2]);

        target_pos.x = target_trans[0];
        target_pos.y = target_trans[1];
        target_pos.z = target_trans[2];

        return target_pos;
    }

    void set_robot_position()
    {
        double trans_data[3] = {robot_pos.x, robot_pos.y, robot_pos.z};
        double rot_data[4] = {0.0, 0.0, 1.0, robot_pos.angle};
        trans_field = robot_node->getField("translation");
        rot_field = robot_node->getField("rotation");
        trans_field->setSFVec3f(trans_data);
        rot_field->setSFRotation(rot_data);
    }

    void set_target_ball_position()
    {
        double trans_data[3] = {target_pos.x, target_pos.y, target_pos.z};
        target_trans_field = target_node->getField("translation");
        target_trans_field->setSFVec3f(trans_data);
    }

    double axis_ang_to_yaw(double *axis_ang)
    {
        double yaw = 0.0;

        double s1 = sin(axis_ang[3] / 2);
        Quaternion q;
        q.w = cos(axis_ang[3] / 2);
        q.x = axis_ang[0] * s1;
        q.y = axis_ang[1] * s1;
        q.z = axis_ang[2] * s1;

        double t3 = +2.0 * (q.w * q.z + q.x * q.y);
        double t4 = +1.0 - 2.0 * (pow(q.y, 2) + pow(q.z, 2));
        yaw = atan2(t3, t4);

        return yaw;
    }

    double euclideanDistance2D(double x1, double y1, double x2, double y2)
    {
        return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
    }

    // Function to calculate theta to target
    double thetaToTarget(const double position[2], const double positionTarget[2], double angle)
    {
        // Compute differences
        double skew_x = positionTarget[0] - position[0];
        double skew_y = positionTarget[1] - position[1];

        // Compute dot product and magnitudes
        double dot = skew_x * 1.0 + skew_y * 0.0; // Simplifies to skew_x
        double mag1 = std::sqrt(std::pow(skew_x, 2) + std::pow(skew_y, 2));
        double mag2 = 1.0; // The magnitude of (1, 0) is 1

        // Compute beta (angle between vectors)
        double beta = std::acos(dot / (mag1 * mag2));

        // Adjust beta based on the direction of skew_y
        if (skew_y < 0)
        {
            beta = (skew_x < 0) ? -beta : -beta;
        }

        // Compute theta
        double theta = beta - angle;

        // Normalize theta to be within the range [-M_PI, M_PI]
        if (theta > M_PI)
        {
            theta -= 2.0 * M_PI;
        }
        else if (theta < -M_PI)
        {
            theta += 2.0 * M_PI;
        }

        return theta;
    }

    std::pair<double, double> calculate_distance_and_yaw()
    {
        // comment this because double get robot position() called at push_lidar_to_pcl()
        robot_pos = get_robot_position();
        target_pos = get_target_position();

        rot_field = robot_node->getField("rotation");
        double *robot_rot = const_cast<double *>(rot_field->getSFRotation());

        double robot_xy[2] = {robot_pos.x, robot_pos.y};
        double target_xy[2] = {target_pos.x, target_pos.y};
        double theta = thetaToTarget(robot_xy,
                                     target_xy,
                                     axis_ang_to_yaw(robot_rot));

        return std::make_pair(theta, euclideanDistance2D(robot_pos.x, robot_pos.y,
                                                         target_pos.x, target_pos.y));
    }

    void run()
    {

        bool first = true;

        unsigned int min_range = half_degree ? (int)90 * (lidar_width / 360) : 0;
        unsigned int max_range = half_degree ? (int)270 * (lidar_width / 360) : lidar_width;
        // unsigned int laser_step = (int)(max_range - min_range) / (lidar_num_state - 1);
        unsigned int laser_step = (int)(max_range - min_range) / (lidar_num_state);

        //because of setup joystick we need step once
        step(time_step);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        while (step(time_step) != -1)
        {
            // logger("info", "Turtlebot3_Env", "hello world %d", i);
            // robot_pos = get_robot_position();
            // waiting agent msg arrived
            // reset or step parse there...
            //  robo_msg.receiveSubscriptions();
            //  logger("custom", "c++", "checking subscriptions");
            //  logger("custom", "c++", "subs : %d", robo_msg.subscriptions.count("state"));
            if (robo_msg.subscriptions.count("state") > 0)
            {
                // logger("custom", "c++", "listening agent message");
                while (!agent_msg.is_agent_message_start_received &&
                       !agent_msg.is_agent_message_step_received)
                    ;
                if (agent_msg.is_agent_message_start_received)
                {
                    // logger("custom", "c++", "agent start msg arrived");
                    agent_msg.is_agent_message_start_received = false;
                    robot_pos.x = agent_msg.start_msg.x();
                    robot_pos.y = agent_msg.start_msg.y();
                    robot_pos.z = agent_msg.start_msg.z();
                    robot_pos.angle = agent_msg.start_msg.angle();
                    target_pos.x = agent_msg.start_msg.target_x();
                    target_pos.y = agent_msg.start_msg.target_y();
                    target_pos.z = agent_msg.start_msg.target_z();
                    step_max = agent_msg.start_msg.max_steps();
                    lidar_num_state = agent_msg.start_msg.lidar_for_state();
                    laser_step = (int)(max_range - min_range) / (lidar_num_state - 1);
                    //cout << "lidar num state : " << lidar_num_state << endl;
                    set_robot_position();
                    set_target_ball_position();
                    resetting_lidar();
                    robot_node->resetPhysics();
                    target_node->resetPhysics();
                    // cout << "stepping..." << endl;
                    step(time_step);
                    // coud << "done stepping..."<<endl;
                }
                if (agent_msg.is_agent_message_step_received)
                {
                    // logger("custom", "c++", "agent step msg arrived");
                    agent_msg.is_agent_message_step_received = false;
                    robot_act.linear_vel = (agent_msg.step_msg.act_0()) ? agent_msg.step_msg.act_0() : 0.000;
                    robot_act.angular_vel = (agent_msg.step_msg.act_1()) ? agent_msg.step_msg.act_1() : 0.000;

                    // because can be fixed steps (preferred) so check min lidar then act to zeroing
                    if ((int)last_lidar_data.size() == lidar_num_state)
                    {
                        auto minIt = std::min_element(last_lidar_data.begin(), last_lidar_data.end());
                        if (*minIt <= 0.12)
                        {
                            robot_act.linear_vel = -1.000;
                            robot_act.angular_vel = 0.000;
                        }
                    }

                    set_action_robot(robot_act);
                    // step(time_step);
                }
            }

            // logger("custom", "c++", "reading lidar..");
            read_lidar_data();
            robot_pos = get_robot_position();
            lidar_data_arrived = true;


            if (robo_msg.subscriptions.count("state") > 0)
            {
                // cout << "calculate ..." <<endl;
                auto [yaw, distance] = calculate_distance_and_yaw();
                // i++;
                std::string encoded_msg;
                robot_msg msg;
                msg.set_idx(state_idx);
                msg.set_distance_length(rounding(distance));
                msg.set_angular_length(rounding(yaw));

                last_lidar_data.clear();
                // for (unsigned int i = min_range; i <= max_range; i += laser_step)
                for (unsigned int i = min_range; i < max_range; i += laser_step)
                {
                    std::vector<float> temp_data;
                    for (unsigned int temp=i;temp<i+laser_step;temp++)
                    {
                        if (std::isinf(raw_lidar_data[i]))
                        {
                            if(std::isinf(raw_lidar_data[i-1])==false and std::isinf(raw_lidar_data[i+1]==false))
                            {
                                raw_lidar_data[i] = (raw_lidar_data[i-1]+raw_lidar_data[i+1])/2.0f;
                            }
                            else
                            {
                                raw_lidar_data[i] = 0.000;
                            }
                        }
                        temp_data.push_back(raw_lidar_data[temp]);
                    }
                    auto minIt = std::min_element(temp_data.begin(), temp_data.end());
                    // msg.add_lidar_data(rounding2Dec(raw_lidar_data[i]));
                    // last_lidar_data.push_back(raw_lidar_data[i]);
                    msg.add_lidar_data(rounding2Dec(*minIt));
                    last_lidar_data.push_back(*minIt);
                }
                msg.SerializeToString(&encoded_msg);

                if (robo_msg.success_publish or first)
                {
                    // set encoded msg and enable msg_publish to publish once
                    //  cout << "idx : " << state_idx << endl;
                    robo_msg.encoded_msg = encoded_msg;
                    robo_msg.msg_publish = true;
                    first = false;

                    state_idx++;
                    robo_msg.success_publish = false;
                }
                // std::cout << "wait..." << std::endl;
                while (!robo_msg.is_state_published)
                    ;
                robo_msg.is_state_published = false;
            }

        }
    }

    void ros_run()
    {
        //launch a separate thread for rclcpp::spin
        ros_thread_ = std::thread(&Turtlebot3_Env::spinning, this);
        RCLCPP_INFO(this->get_logger(), "thread running...");
    }

    void spinning()
    {
        //this will spin in a separate thread
        rclcpp::spin(shared_from_this());
    }

    auto createQuaternionMsgFromYaw(double yaw)
    {
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    return tf2::toMsg(q);
    }

    builtin_interfaces::msg::Time webotsTime()
    {
        builtin_interfaces::msg::Time timeMsg;
        double timeInSeconds = this->getTime();
        timeMsg.sec = static_cast<int32_t>(timeInSeconds);
        timeMsg.nanosec = static_cast<uint32_t>((timeInSeconds - timeMsg.sec) * 1e9);
        return timeMsg;
    }

    //the data must be same according lidar do you use...
    void publishRviz()
    {
        if(lidar_data_arrived)
        {
            /////////////////////////////////////////////////////////////////////////////
            //// tf odom to base_link
            geometry_msgs::msg::TransformStamped odom_trans;

            // Read message content and assign it to
            // corresponding tf variables
            // odom_trans.header.stamp = this->get_clock()->now();
            odom_trans.header.stamp = webotsTime();
            odom_trans.header.frame_id = "odom";
            odom_trans.child_frame_id = "base_link";

            //get from rotation robot tb3
            robot_pos = get_robot_position();
            double *robot_rot = const_cast<double *>(rot_field->getSFRotation());
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, axis_ang_to_yaw(robot_rot));

            odom_trans.transform.translation.x = -robot_pos.x;
            odom_trans.transform.translation.y = -robot_pos.y;
            odom_trans.transform.translation.z = 0.0;
            odom_trans.transform.rotation.x = q.x();
            odom_trans.transform.rotation.y = q.y();
            odom_trans.transform.rotation.z = q.z();
            odom_trans.transform.rotation.w = q.w();


            // Send the transformation
            tf_broadcaster_->sendTransform(odom_trans);
            /////////////////////////////////////////////////////////////////////////////

            nav_msgs::msg::Odometry odom_msg;
            // odom_msg.header.stamp = this->get_clock()->now();
            odom_msg.header.stamp = webotsTime();
            odom_msg.header.frame_id = "odom";

            //get from rotation robot tb3
            // robot_pos = get_robot_position();
            // double *robot_rot = const_cast<double *>(rot_field->getSFRotation());
            // tf2::Quaternion q;
            // q.setRPY(0.0, 0.0, -axis_ang_to_yaw(robot_rot));

            odom_msg.pose.pose.position.x = -robot_pos.x;
            odom_msg.pose.pose.position.y = -robot_pos.y;
            odom_msg.pose.pose.position.z = 0.0;
            odom_msg.pose.pose.orientation.x = q.x();
            odom_msg.pose.pose.orientation.y = q.y();
            odom_msg.pose.pose.orientation.z = q.z();
            odom_msg.pose.pose.orientation.w = q.w();
            
            robot_velocity vel = get_robot_velocity();
            odom_msg.child_frame_id = "base_link";
            odom_msg.twist.twist.linear.x = vel.linear_vel_x;
            odom_msg.twist.twist.angular.z = vel.angular_vel_z;

            // Publish the message
            OdomPublisher_->publish(odom_msg);

            //// publish lidar laser data
            auto lidar_msg = sensor_msgs::msg::LaserScan();
            // lidar_msg.header.stamp = this->get_clock()->now();
            lidar_msg.header.stamp = webotsTime();
            lidar_msg.header.frame_id = "base_link";
            lidar_msg.angle_min = 0;  // 0 degrees
            lidar_msg.angle_max = 2*M_PI;   // 360 degrees
            lidar_msg.angle_increment = (lidar_degree/lidar_data)*(M_PI/180);  // Angular resolution
            lidar_msg.time_increment = 0.0;    // Time between measurements
            lidar_msg.scan_time = (float)(time_step/1000.0f);         // Time between scans
            lidar_msg.range_min = 0.115;         // Minimum range
            lidar_msg.range_max = 10.0;        // Maximum range
            std::vector<float> data;

            // Convert each element from double to float and add to the vector
            // for (int i = 0; i < lidar_data; i++) {
            for (int i = lidar_data-1; i >= 0; i--) {
                data.push_back(static_cast<float>(raw_lidar_data[i]));
            }
            lidar_msg.ranges = data;

            //publish the message
            laserPublisher_->publish(lidar_msg);
            lidar_data_arrived = false;
        }

    }

};

int main(int argc, char *argv[])
{

    // Turtlebot3_Env *myEnv = new Turtlebot3_Env();
    rclcpp::init(argc, argv);
    // make shared pointer
    auto node = std::make_shared<Turtlebot3_Env>();
    
    //start the spinning in a separate thread
    node->ros_run();

    //run other task i.e. webots with zmq communication
    node->run();

    rclcpp::shutdown();
    return 0;
}
