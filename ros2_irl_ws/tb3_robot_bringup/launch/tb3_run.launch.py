from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    return LaunchDescription([
        # Add the TurtleBot3Robot node
        Node(
            package='tb3_robot',
            executable='tb3_robot',
            name='tb3robot_node',
            # parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
    ])
