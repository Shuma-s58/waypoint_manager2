#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import yaml
import math
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32
from copy import deepcopy

# YAML のパスを適宜変更してください
#WAYPOINT_PATH = '/home/ros2_ws/src/orne-box/orne_box_navigation_executor/config/waypoints/tsudanuma2-3.yaml'
WAYPOINT_PATH = '/home/ros/ros2_ws/src/orne-box/orne_box_navigation_executor/config/waypoints/tsukuba2025_all.yaml'

# 表示ウィンドウサイズ（先頭から何個表示するか）
WINDOW_SIZE = 5

# タイマー周期（秒）
TIMER_PERIOD = 0.2

class WaypointWindowNode(Node):
    def __init__(self):
        super().__init__('waypoint_window_node')

        # action client for NavigateToPose
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # publishers / subscribers
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoint_markers', 1)
        self.current_index_pub = self.create_publisher(Int32, 'waypoint_window/current_index', 1)

        qos_profile = qos.qos_profile_sensor_data
        self.mcl_sub = self.create_subscription(PoseWithCovarianceStamped, 'mcl_pose', self.mcl_callback, qos_profile)

        # load YAML waypoints
        self.all_waypoints = []
        self.load_waypoints(WAYPOINT_PATH)

        # indices
        self.current_index = 0  # ウィンドウの先頭（かつナビゲート中のゴール）
        self.window_size = WINDOW_SIZE

        # robot pose (mcl)
        self.mcl_pose = None

        # currently-target pose (PoseStamped)
        self.current_goal_pose = None

        # timer
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

        # 初回ゴール送信（もし waypoints があるなら）
        if len(self.all_waypoints) > 0:
            self.set_and_send_goal(self.current_index)

    def load_waypoints(self, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        try:
            waypoints = cfg['waypoint_server']['waypoints']
        except Exception as e:
            self.get_logger().error(f'YAML structure error: {e}')
            waypoints = []

        for wp in waypoints:
            pos = wp.get('position', {})
            eul = wp.get('euler_angles', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            props = wp.get('properties', {})
            item = {
                'position': {'x': float(pos.get('x', 0.0)), 'y': float(pos.get('y', 0.0))},
                'euler_angles': {'x': float(eul.get('x', 0.0)), 'y': float(eul.get('y', 0.0)), 'z': float(eul.get('z', 0.0))},
                'properties': deepcopy(props)
            }
            self.all_waypoints.append(item)

        self.get_logger().info(f'Loaded {len(self.all_waypoints)} waypoints from {path}')

    def mcl_callback(self, msg: PoseWithCovarianceStamped):
        self.mcl_pose = msg.pose.pose

    def quaternion_from_euler(self, roll, pitch, yaw):
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        q = r.as_quat()  # [x,y,z,w]
        return q

    def _stop_flag_from_props(self, props):
        """
        properties の中から Stop フラグを判定する。
        期待するキー: 'Stop_wp'（値: 'Stop_ON' / 'Stop_OFF'）
        大文字小文字は無視して比較する。
        戻り値: 'ON' | 'OFF'
        デフォルトは 'OFF'
        """
        if not isinstance(props, dict):
            return 'OFF'
        # 直接キーをチェック
        for k in ('Stop_wp', 'stop_wp', 'Stop_Wp', 'stop'):
            if k in props:
                v = str(props[k]).upper()
                if 'STOP_ON' in v or 'ON' == v:
                    return 'ON'
                if 'STOP_OFF' in v or 'OFF' == v:
                    return 'OFF'
        # もしどのキーもなければ、探し方を広げる（キーに 'stop' を含むものを探す）
        for k, v_raw in props.items():
            if 'stop' in str(k).lower():
                v = str(v_raw).upper()
                if 'STOP_ON' in v or 'ON' == v:
                    return 'ON'
                if 'STOP_OFF' in v or 'OFF' == v:
                    return 'OFF'
        # デフォルト
        return 'OFF'

    def make_marker_for_wp(self, display_idx, original_idx, wp, frame_id='map'):
        """
        シンプルな円柱マーカー（id=display_idx*2）と向き矢印（id=display_idx*2+1）を返す。
        色は properties の Stop フラグに依存（STOP_ON -> 赤, STOP_OFF -> 緑）。
        """
        markers = []

        stop_flag = self._stop_flag_from_props(wp.get('properties', {}))

        # cylinder
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'waypoints'
        m.id = display_idx * 2
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = wp['position']['x']
        m.pose.position.y = wp['position']['y']
        m.pose.position.z = 0.02
        radius = 0.2
        if 'goal_radius' in wp.get('properties', {}):
            try:
                radius = float(wp['properties']['goal_radius'])
            except:
                pass
        m.scale.x = radius
        m.scale.y = radius
        m.scale.z = 0.04

        # 色を Stop フラグから決定
        if stop_flag == 'ON':
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
        else:  # OFF / default
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.9

        markers.append(m)

        # arrow for orientation
        q = self.quaternion_from_euler(float(wp['euler_angles']['x']),
                                       float(wp['euler_angles']['y']),
                                       float(wp['euler_angles']['z']))
        ma = Marker()
        ma.header.frame_id = frame_id
        ma.header.stamp = self.get_clock().now().to_msg()
        ma.ns = 'waypoint_arrows'
        ma.id = display_idx * 2 + 1
        ma.type = Marker.ARROW
        ma.action = Marker.ADD
        ma.pose.position.x = wp['position']['x']
        ma.pose.position.y = wp['position']['y']
        ma.pose.position.z = 0.04
        ma.pose.orientation.x = q[0]
        ma.pose.orientation.y = q[1]
        ma.pose.orientation.z = q[2]
        ma.pose.orientation.w = q[3]
        ma.scale.x = 0.25
        ma.scale.y = 0.05
        ma.scale.z = 0.05
        # 矢印はわかりやすく黒系（不変）
        ma.color.r = 0.1
        ma.color.g = 0.1
        ma.color.b = 0.1
        ma.color.a = 1.0
        markers.append(ma)

        return markers

    def publish_window_markers(self):
        ma = MarkerArray()
        n = len(self.all_waypoints)
        start = self.current_index
        end = min(n - 1, start + self.window_size - 1)
        display_id = 0
        for i in range(start, end + 1):
            wp = self.all_waypoints[i]
            markers = self.make_marker_for_wp(display_id, i, wp)
            ma.markers.extend(markers)
            display_id += 1
        self.marker_pub.publish(ma)

    def set_and_send_goal(self, index):
        if index < 0 or index >= len(self.all_waypoints):
            self.get_logger().warn('set_and_send_goal: index out of range')
            return

        wp = self.all_waypoints[index]
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = wp['position']['x']
        pose.pose.position.y = wp['position']['y']
        pose.pose.position.z = 0.0
        q = self.quaternion_from_euler(float(wp['euler_angles']['x']),
                                       float(wp['euler_angles']['y']),
                                       float(wp['euler_angles']['z']))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.current_goal_pose = pose

        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('NavigateToPose action server not available (timeout).')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        def handle_response(fut):
            goal_handle = fut.result()
            if not goal_handle.accepted:
                self.get_logger().warn('Goal was rejected by action server.')
                return
            self.get_logger().info(f'Goal accepted for waypoint {index}.')
            res_fut = goal_handle.get_result_async()
            def result_cb(rf):
                result = rf.result()
                self.get_logger().info(f'Goal result received: status={result.status}')
            res_fut.add_done_callback(result_cb)
        send_goal_future.add_done_callback(handle_response)

        self.get_logger().info(f'Sent goal for waypoint idx={index} -> ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')

    def distance_to_current_goal(self):
        if self.mcl_pose is None or self.current_goal_pose is None:
            return None
        dx = self.current_goal_pose.pose.position.x - self.mcl_pose.position.x
        dy = self.current_goal_pose.pose.position.y - self.mcl_pose.position.y
        return math.hypot(dx, dy)

    def current_goal_radius(self):
        wp = None
        if 0 <= self.current_index < len(self.all_waypoints):
            wp = self.all_waypoints[self.current_index]
        if wp is None:
            return 0.5
        props = wp.get('properties', {})
        if 'goal_radius' in props:
            try:
                return float(props['goal_radius'])
            except:
                return 0.5
        return 0.5

    def timer_callback(self):
        # publish visible window markers
        self.publish_window_markers()
        # publish current index for debugging or other nodes
        self.current_index_pub.publish(Int32(data=self.current_index))

        # arrival 判定（mcl を使ってチェック）
        dist = self.distance_to_current_goal()
        if dist is None:
            return

        radius = self.current_goal_radius()
        # 到着判定: dist <= radius
        if dist <= radius:
            if self.current_index < len(self.all_waypoints) - 1:
                self.get_logger().info(f'Waypoint {self.current_index} reached (dist={dist:.2f} <= {radius:.2f}). Sliding window.')
                self.current_index += 1
                self.set_and_send_goal(self.current_index)
            else:
                self.get_logger().info('Final waypoint reached; no further waypoints.')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointWindowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

