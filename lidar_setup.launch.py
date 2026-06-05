from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # LiDAR Daten (wie gehabt)
    lidars = [
        ['lidar_01', -7.7,  3.1, 4.0, 0.0, 3.14159, 0.0],
        ['lidar_02',  0.0,  3.1, 4.0, 0.0, 3.14159, 0.0],
        ['lidar_03',  7.7,  3.1, 4.0, 0.0, 3.14159, 0.0],
        ['lidar_04', -7.7, -3.1, 4.0, 0.0, 3.14159, 0.0],
        ['lidar_05',  0.0, -3.1, 4.0, 0.0, 3.14159, 0.0],
        ['lidar_06',  7.7, -3.1, 4.0, 0.0, 3.14159, 0.0],
    ]

    # Kamera Daten aus deiner SDF (x, y, z, yaw, pitch, roll)
    # Pose in SDF: <pose> x y z R P Y </pose> -> Mapping auf Launch Arguments
    cameras = [
        ['cam_in',    -6.35, -5.2,  4.0, 0.0, 1.57, 0.0],
        ['cam_out_1',  10.6,  2.25, 4.0, 0.0, 1.57, 0.0],
        ['cam_out_2',  10.6, -2.25, 4.0, 0.0, 1.57, 0.0],
    ]

    nodes = []

    # LiDAR Nodes erstellen
    for l in lidars:
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'tf_{l[0]}',
            arguments=[
                '--x', str(l[1]), '--y', str(l[2]), '--z', str(l[3]),
                '--yaw', str(l[4]), '--pitch', str(l[5]), '--roll', str(l[6]),
                '--frame-id', 'world', 
                '--child-frame-id', f'{l[0]}/link/s'
            ]
        ))

    # Kamera Nodes erstellen
    for c in cameras:
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'tf_{c[0]}',
            arguments=[
                '--x', str(c[1]), '--y', str(c[2]), '--z', str(c[3]),
                '--yaw', str(c[4]), '--pitch', str(c[5]), '--roll', str(c[6]),
                '--frame-id', 'world', 
                '--child-frame-id', f'{c[0]}/link/s'
            ]
        ))

    return LaunchDescription(nodes)
