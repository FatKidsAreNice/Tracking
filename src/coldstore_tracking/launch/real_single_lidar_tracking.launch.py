from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = PathJoinSubstitution(
        [FindPackageShare('coldstore_tracking'), 'config', 'real_single_lidar_tracking.yaml']
    )

    return LaunchDescription(
        [
            Node(
                package='coldstore_tracking',
                executable='cloud_transform_merge_node',
                name='cloud_transform_merge_node',
                output='screen',
                parameters=[params_file],
            ),
            Node(
                package='coldstore_tracking',
                executable='cluster_detector_node',
                name='cluster_detector_node',
                output='screen',
                parameters=[params_file],
            ),
            Node(
                package='coldstore_tracking',
                executable='track_manager_node',
                name='track_manager_node',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
