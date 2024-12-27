import math
from os import close
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        """
        PID controller initialization.
        
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param dt: Time step
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        """
        Compute the control output using PID formula.
        
        :param error: The current error
        :return: Control output
        """
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class RobotController:
    def __init__(self, kp_linear, ki_linear, kd_linear, kp_angular, ki_angular, kd_angular, dt, safe_distance):
        """
        Robot controller for linear and angular velocity control.

        :param kp_linear, ki_linear, kd_linear: Gains for linear velocity PID controller
        :param kp_angular, ki_angular, kd_angular: Gains for angular velocity PID controller
        :param dt: Time step
        """
        self.linear_pid = PIDController(kp_linear, ki_linear, kd_linear, dt)
        self.angular_pid = PIDController(kp_angular, ki_angular, kd_angular, dt)
        self.safe_distance = safe_distance

    def compute_velocities(self, current_x, current_y, current_theta, goal_x, goal_y, lidar_readings):
        # Goal-reaching control
        distance_error = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        goal_theta = math.atan2(goal_y - current_y, goal_x - current_x)
        angular_error = goal_theta - current_theta
        angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))  # Normalize
        print(f"goal_theta: {goal_theta:.3f}, current_theta: {current_theta:.3f}, angular_error: {angular_error:.3f}")

        linear_velocity = self.linear_pid.compute(distance_error)
        angular_velocity = self.angular_pid.compute(angular_error)

        # Obstacle avoidance
        lidar_readings = lidar_readings.tolist()
        min_distance = min(lidar_readings)
        if min_distance < self.safe_distance:
            # print("this")
            # Determine direction to steer away from the closest obstacle
            closest_index = lidar_readings.index(min_distance)
            # Positive angular velocity if the obstacle is on the left, negative if on the right
            if closest_index == 4:
                nom = 2
            elif closest_index == 3 or closest_index == 5:
                nom = 1.5
            elif closest_index == 2 or closest_index == 6:
                nom = 1.25
            elif closest_index == 1 or closest_index == 7:
                nom = 1
            elif closest_index == 0 or closest_index == 8:
                nom = 0.75
            angular_avoidance = -(nom/min_distance) if closest_index < len(lidar_readings) // 2 else (nom/min_distance)
            angular_velocity += angular_avoidance * (1 - min_distance / self.safe_distance)

            # Reduce linear velocity to prevent collisions
            linear_velocity *= min_distance / self.safe_distance

        return linear_velocity, -angular_velocity
