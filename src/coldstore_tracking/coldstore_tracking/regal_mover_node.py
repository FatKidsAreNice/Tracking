from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class RegalMoverNode(Node):
    def __init__(self) -> None:
        super().__init__('regal_mover_node')

        self.declare_parameter('world_name', 'kuehlhaus_world')
        self.declare_parameter('entity_name', 'regal')
        self.declare_parameter('gz_binary', 'gz')
        self.declare_parameter('service_timeout_ms', 2000)
        self.declare_parameter('delete_wait_sec', 0.35)

        self.world_name = str(self.get_parameter('world_name').value)
        self.entity_name = str(self.get_parameter('entity_name').value)
        self.gz_binary = str(self.get_parameter('gz_binary').value)
        self.service_timeout_ms = int(self.get_parameter('service_timeout_ms').value)
        self.delete_wait_sec = float(self.get_parameter('delete_wait_sec').value)

        self.spawn_dir = Path.home() / '.ros' / 'coldstore_tracking'
        self.spawn_dir.mkdir(parents=True, exist_ok=True)
        self.spawn_sdf_path = self.spawn_dir / f'{self.entity_name}_spawn.sdf'

        self.waypoints: Dict[str, Waypoint] = {
            'entry': Waypoint('entry', -6.35, -5.20, 0.95, 0.0, 0.0, 0.0),
            'center': Waypoint('center', 0.00, 0.00, 0.95, 0.0, 0.0, 0.0),
            'exit_top': Waypoint('exit_top', 10.60, 2.25, 0.95, 0.0, 0.0, 0.0),
            'exit_bottom': Waypoint('exit_bottom', 10.60, -2.25, 0.95, 0.0, 0.0, 0.0),
        }

        self.entry_srv = self.create_service(
            Trigger,
            '/tracking/move_regal_entry',
            partial(self.move_regal_callback, 'entry'),
        )
        self.center_srv = self.create_service(
            Trigger,
            '/tracking/move_regal_center',
            partial(self.move_regal_callback, 'center'),
        )
        self.exit_top_srv = self.create_service(
            Trigger,
            '/tracking/move_regal_exit_top',
            partial(self.move_regal_callback, 'exit_top'),
        )
        self.exit_bottom_srv = self.create_service(
            Trigger,
            '/tracking/move_regal_exit_bottom',
            partial(self.move_regal_callback, 'exit_bottom'),
        )

        self.get_logger().info('regal_mover_node started.')
        self.get_logger().info(f'World: {self.world_name}')
        self.get_logger().info(f'Entity: {self.entity_name}')
        self.get_logger().info(f'Spawn SDF path: {self.spawn_sdf_path}')
        self.get_logger().info(
            'Services: '
            '/tracking/move_regal_entry, '
            '/tracking/move_regal_center, '
            '/tracking/move_regal_exit_top, '
            '/tracking/move_regal_exit_bottom'
        )

    def move_regal_callback(
        self,
        waypoint_key: str,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        waypoint = self.waypoints[waypoint_key]
        success, message = self.respawn_regal_at(waypoint)

        response.success = success
        response.message = message
        return response

    def respawn_regal_at(self, waypoint: Waypoint) -> tuple[bool, str]:
        self.get_logger().info(
            f'Respawning "{self.entity_name}" at waypoint "{waypoint.name}" '
            f'({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f}).'
        )

        delete_success, delete_output = self.delete_existing_entity()
        if delete_success:
            self.get_logger().info(f'Delete request accepted for "{self.entity_name}".')
        else:
            self.get_logger().warning(
                f'Delete request returned non-success. Continuing with spawn anyway. Output: {delete_output}'
            )

        time.sleep(self.delete_wait_sec)

        spawn_success, spawn_output = self.spawn_entity(waypoint)
        if not spawn_success:
            return False, (
                f'Spawn failed for "{self.entity_name}" at "{waypoint.name}". '
                f'Output: {spawn_output}'
            )

        return True, (
            f'"{self.entity_name}" respawned at "{waypoint.name}" '
            f'({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f}).'
        )

    def delete_existing_entity(self) -> tuple[bool, str]:
        service_name = f'/world/{self.world_name}/remove'
        request = f'name: "{self.entity_name}" type: 2'

        return self.call_gz_service(
            service_name=service_name,
            request_type='gz.msgs.Entity',
            response_type='gz.msgs.Boolean',
            request=request,
        )

    def spawn_entity(self, waypoint: Waypoint) -> tuple[bool, str]:
        service_name = f'/world/{self.world_name}/create'

        sdf_text = self.build_regal_sdf(waypoint)
        self.spawn_sdf_path.write_text(sdf_text, encoding='utf-8')

        request = (
            f'sdf_filename: "{self.spawn_sdf_path}" '
            f'name: "{self.entity_name}" '
            f'allow_renaming: false'
        )

        return self.call_gz_service(
            service_name=service_name,
            request_type='gz.msgs.EntityFactory',
            response_type='gz.msgs.Boolean',
            request=request,
        )

    def call_gz_service(
        self,
        service_name: str,
        request_type: str,
        response_type: str,
        request: str,
    ) -> tuple[bool, str]:
        command = [
            self.gz_binary,
            'service',
            '-s',
            service_name,
            '--reqtype',
            request_type,
            '--reptype',
            response_type,
            '--timeout',
            str(self.service_timeout_ms),
            '--req',
            request,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(5.0, self.service_timeout_ms / 1000.0 + 2.0),
                check=False,
            )
        except FileNotFoundError:
            return False, f'Gazebo CLI not found: {self.gz_binary}'
        except subprocess.TimeoutExpired:
            return False, f'Gazebo service call timed out: {" ".join(command)}'

        output = '\n'.join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )

        if completed.returncode != 0:
            return False, output or f'Gazebo service command failed with code {completed.returncode}.'

        normalized = output.lower()
        success = ('data: true' in normalized) or (normalized.strip() == 'true')

        return success, output

    def build_regal_sdf(self, waypoint: Waypoint) -> str:
        return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{self.entity_name}">
    <static>false</static>
    <pose>{waypoint.x} {waypoint.y} {waypoint.z} {waypoint.roll} {waypoint.pitch} {waypoint.yaw}</pose>

    <link name="link">
      <gravity>false</gravity>
      <kinematic>true</kinematic>
      <self_collide>false</self_collide>

      <inertial>
        <mass>20.0</mass>
        <inertia>
          <ixx>5.6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>5.6</iyy>
          <iyz>0</iyz>
          <izz>2.1</izz>
        </inertia>
      </inertial>

      <collision name="c">
        <geometry>
          <box>
            <size>0.8 0.8 1.8</size>
          </box>
        </geometry>
      </collision>

      <visual name="v">
        <geometry>
          <box>
            <size>0.8 0.8 1.8</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegalMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()