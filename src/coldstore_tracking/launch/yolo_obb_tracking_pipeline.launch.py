from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration('params_file')
    launch_gui = LaunchConfiguration('launch_gui')

    default_params_file = PathJoinSubstitution(
        [FindPackageShare('coldstore_tracking'), 'config', 'yolo_obb_bev_detector.yaml']
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params_file,
                description='Parameter file for the YOLO OBB tracking pipeline.',
            ),
            DeclareLaunchArgument(
                'launch_gui',
                default_value='true',
                description='Start the desktop track overview GUI.',
            ),
            Node(
                package='coldstore_tracking',
                executable='yolo_obb_bev_detector_node',
                name='yolo_obb_bev_detector_node',
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
            Node(
                package='coldstore_tracking',
                executable='track_overview_gui_node',
                name='track_overview_gui_node',
                output='screen',
                parameters=[params_file],
                condition=IfCondition(launch_gui),
            ),
        ]
    )
