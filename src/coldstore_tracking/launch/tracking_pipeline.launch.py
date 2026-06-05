from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    common_parameters = [
        {'use_sim_time': True},
    ]

    return LaunchDescription([
        Node(
            package='coldstore_tracking',
            executable='cloud_transform_merge_node',
            name='cloud_transform_merge_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='coldstore_tracking',
            executable='cluster_detector_node',
            name='cluster_detector_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='coldstore_tracking',
            executable='track_manager_node',
            name='track_manager_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='coldstore_tracking',
            executable='virtual_scanner_node',
            name='virtual_scanner_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='coldstore_tracking',
            executable='id_assignment_node',
            name='id_assignment_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='coldstore_tracking',
            executable='regal_mover_node',
            name='regal_mover_node',
            output='screen',
            parameters=common_parameters,
        ),
    ])