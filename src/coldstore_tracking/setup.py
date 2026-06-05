from glob import glob
from setuptools import find_packages, setup

package_name = 'coldstore_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/coldstore_tracking']),
        ('share/coldstore_tracking', ['package.xml']),
        ('share/coldstore_tracking/launch', glob('launch/*.launch.py')),
        ('share/coldstore_tracking/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenAI',
    maintainer_email='noreply@example.com',
    description='Minimal cold store tracking pipeline for ROS 2 Jazzy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cloud_transform_merge_node = coldstore_tracking.cloud_transform_merge_node:main',
            'cluster_detector_node = coldstore_tracking.cluster_detector_node:main',
            'track_manager_node = coldstore_tracking.track_manager_node:main',
            'track_overview_gui_node = coldstore_tracking.track_overview_gui_node:main',
            'virtual_scanner_node = coldstore_tracking.virtual_scanner_node:main',
            'id_assignment_node = coldstore_tracking.id_assignment_node:main',
            'regal_mover_node = coldstore_tracking.regal_mover_node:main',
            'bev_dataset_export_node = coldstore_tracking.bev_dataset_export_node:main',
        ],
    },
)
