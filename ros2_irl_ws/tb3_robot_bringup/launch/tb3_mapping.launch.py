from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()

    # get the path to the TurtleBot3Robot package share directory
    turtlebot3_robot_share_dir = get_package_share_directory('tb3_robot')

    # Path to the config directory (after it's copied during the build)
    default_slam_config_file = os.path.join(turtlebot3_robot_share_dir, 'config', 'slam_config.yaml')
    default_rviz_file = os.path.join(turtlebot3_robot_share_dir, 'config', 'default_rviz.rviz')

    return LaunchDescription([
        # Declare the launch arguments (parameters that can be passed from the command line)
        DeclareLaunchArgument(
            'slam_params_file',
            default_value=default_slam_config_file,
            description='Full path to the slam_toolbox configuration file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Add the TurtleBot3Robot node
        Node(
            package='tb3_robot',
            executable='tb3_robot',
            name='tb3robot_node',
            # parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),

        # Include the slam_toolbox launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')
            ),
            launch_arguments={
                'slam_params_file': LaunchConfiguration('slam_params_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }.items()
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', default_rviz_file]
        ),
    ])
