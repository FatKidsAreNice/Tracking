from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration('params_file')
    default_params_file = PathJoinSubstitution(
        [FindPackageShare('coldstore_tracking'), 'config', 'bev_dataset_export.yaml']
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params_file,
                description='Parameter file for periodic merged-cloud BEV capture.',
            ),
            Node(
                package='coldstore_tracking',
                executable='bev_dataset_export_node',
                name='bev_dataset_export_node',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
