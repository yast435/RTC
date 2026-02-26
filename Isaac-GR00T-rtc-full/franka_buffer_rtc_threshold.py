#!/usr/bin/env python3
"""
Franka双臂机器人 ROS2 + GR00T Chunk 模式客户端

核心特性：
=====================================================buffer < threshold触发推理=======================================================

话题配置：
发布:
  - /left/websocket/action (JointTrajectory)
  - /right/websocket/action (JointTrajectory)
  - /left/gripper_client/target_gripper_width_percent (Float32, 0-1)
  - /right/gripper_client/target_gripper_width_percent (Float32, 0-1)

接收:
  - /left/franka/joint_states (JointState)
  - /right/franka/joint_states (JointState)
  - /left/franka_gripper/joint_states (JointState)
  - /right/franka_gripper/joint_states (JointState)
  - /left/franka/eef_pose (PoseStamped) [可选: --use-eef-pose]
  - /right/franka/eef_pose (PoseStamped) [可选: --use-eef-pose]
  - /head/color/image_raw/compressed (CompressedImage)
  - /left/color/image_raw/compressed (CompressedImage)
  - /right/color/image_raw/compressed (CompressedImage)
"""

import argparse
import time
import json
import os
import threading
from collections import deque
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Header

import cv2

import sys
sys.path.append('/workspace/Isaac-GR00T')
from gr00t.policy.server_client import PolicyClient

# ==================== 数据记录结构 ====================

@dataclass
class InferenceRecord:
    """单次推理记录"""
    inference_id: int
    timestamp: float
    inference_time_ms: float
    
    # 当前机器人状态
    left_arm_state: List[float]
    right_arm_state: List[float]
    left_gripper_state: float
    right_gripper_state: float
    
    # 模型原始输出（整个 chunk）
    model_output_left_arm: List[List[float]]
    model_output_right_arm: List[List[float]]
    model_output_left_gripper: List[float]
    model_output_right_gripper: List[float]
    
    # 实际发送的动作（截取后的 chunk）
    sent_left_arm: List[List[float]] = field(default_factory=list)
    sent_right_arm: List[List[float]] = field(default_factory=list)
    sent_left_gripper: List[float] = field(default_factory=list)
    sent_right_gripper: List[float] = field(default_factory=list)

class FrankaGr00tChunkClient(Node):
    """
    Chunk 模式客户端：一次发送整个轨迹
    """
    
    def __init__(
        self,
        groot_host: str = "localhost",
        groot_port: int = 6000,
        action_duration: float = 0.05,
        task_description: str = "Pick up the object",
        min_queue_threshold: int = 10,
        # EEF 位姿（可选）
        use_eef_pose: bool = False,
        # RTC (Real-Time Chunking)
        rtc_enabled: bool = False,
        rtc_freeze_steps: int = -1,
        rtc_beta: float = 5.0,
        rtc_mask_decay: float = 2.0,
    ):
        super().__init__('franka_groot_chunk_client')
        
        self.groot_host = groot_host
        self.groot_port = groot_port
        self.task_description = task_description
        self.min_queue_threshold = min_queue_threshold
        self.action_duration = action_duration
        
        # EEF 位姿选项
        self.use_eef_pose = use_eef_pose
        
        # RTC (Real-Time Chunking) 状态
        self.rtc_enabled = rtc_enabled
        self.rtc_freeze_steps = rtc_freeze_steps  # -1=全冻结, 0=不冻结, >0=冻结指定步数
        self.rtc_beta = rtc_beta          # 引导权重裁剪 (Eq. 2)
        self.rtc_mask_decay = rtc_mask_decay  # Soft mask 指数衰减率 (Eq. 5)
        self._rtc_prev_normalized_pred = None  # 上一次推理的归一化预测 (B, H, D)
        
        # 数据缓冲区
        self.left_joint_state = None
        self.right_joint_state = None
        self.left_gripper_state = None
        self.right_gripper_state = None
        self.left_eef_pose = None  # 可选
        self.right_eef_pose = None  # 可选
        self.head_images = deque(maxlen=5)
        self.left_images = deque(maxlen=5)
        self.right_images = deque(maxlen=5)
        
        # 夹爪参数（简化版：直接发布 Float32 值到话题）
        self.gripper_state = {
            'left': {'last_value': 0.5},
            'right': {'last_value': 0.5},
        }
        
        # 状态标志
        self.robot_state_ready = False
        self.camera_ready = False
        self.groot_connected = False
        self.step_count = 0
        self.inference_count = 0
        
        # 数据记录
        self.save_actions = True
        self.inference_records: List[InferenceRecord] = []
        
        # 动作缓冲区（用于数据记录）
        self.action_buffer = {
            'left_arm': None, 'right_arm': None,
            'left_gripper': None, 'right_gripper': None,
        }
        
        # ---- 独立发布线程 & 连续动作队列 ----
        # 每帧元素: (left_arm_7d, right_arm_7d, left_gripper_float, right_gripper_float)
        self._action_queue = deque()
        self._queue_lock = threading.Lock()
        self._last_published_action = None  # 队列空时保持最后位置
        self._publisher_running = False
        self._total_chunk_size = 0  # 模型输出步数 H（首次推理后确定）
        
        # 关节名称（注意：控制器接收的关节名称不带 left/right 前缀）
        self.left_arm_joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        self.right_arm_joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        
        # QoS
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10
        )
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )
        
        # 订阅 (使用 BEST_EFFORT 以获取最新数据，避免消息堆积)
        self.create_subscription(JointState, '/left/franka/joint_states', 
                                  self.left_joint_callback, qos_best_effort)
        self.create_subscription(JointState, '/right/franka/joint_states', 
                                  self.right_joint_callback, qos_best_effort)
        self.create_subscription(JointState, '/left/franka_gripper/joint_states', 
                                  self.left_gripper_callback, qos_best_effort)
        self.create_subscription(JointState, '/right/franka_gripper/joint_states', 
                                  self.right_gripper_callback, qos_best_effort)
        self.create_subscription(CompressedImage, '/head/color/image_raw/compressed', 
                                  self.head_image_callback, qos_best_effort)
        self.create_subscription(CompressedImage, '/left/color/image_raw/compressed', 
                                  self.left_image_callback, qos_best_effort)
        self.create_subscription(CompressedImage, '/right/color/image_raw/compressed', 
                                  self.right_image_callback, qos_best_effort)
        
        # EEF 位姿订阅（可选）
        if self.use_eef_pose:
            self.create_subscription(PoseStamped, '/left/franka/eef_pose', 
                                      self.left_eef_callback, qos_best_effort)
            self.create_subscription(PoseStamped, '/right/franka/eef_pose', 
                                      self.right_eef_callback, qos_best_effort)
        
        # 发布：机械臂动作 (JointState 格式)
        self.left_action_pub = self.create_publisher(
            JointState, '/left/websocket/action', qos_reliable
        )
        self.right_action_pub = self.create_publisher(
            JointState, '/right/websocket/action', qos_reliable
        )
        
        # 发布：夹爪控制 (Float32: 0-1 范围, 0=关闭, 1=打开)
        self.left_gripper_pub = self.create_publisher(
            Float32, '/left/gripper_client/target_gripper_width_percent', qos_reliable
        )
        self.right_gripper_pub = self.create_publisher(
            Float32, '/right/gripper_client/target_gripper_width_percent', qos_reliable
        )
        
        self.groot_client: Optional[PolicyClient] = None
        
        self.get_logger().info(f"[Chunk模式] 准备连接 GR00T: {groot_host}:{groot_port}")
    
    # ==================== 回调函数 ====================
    
    def left_joint_callback(self, msg):
        self.left_joint_state = msg
        self._check_robot_ready()
    
    def right_joint_callback(self, msg):
        self.right_joint_state = msg
        self._check_robot_ready()
    
    def left_gripper_callback(self, msg):
        self.left_gripper_state = msg
    
    def right_gripper_callback(self, msg):
        self.right_gripper_state = msg
    
    def left_eef_callback(self, msg):
        """可选：左臂末端执行器位姿"""
        self.left_eef_pose = msg
    
    def right_eef_callback(self, msg):
        """可选：右臂末端执行器位姿"""
        self.right_eef_pose = msg
    
    def _decode_compressed_image(self, msg):
        """解码压缩图像"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            # 调整大小为 640x480
            img = cv2.resize(img, (640, 480))
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            self.get_logger().error(f"图像解码错误: {e}")
            return None
    
    def head_image_callback(self, msg):
        try:
            img = self._decode_compressed_image(msg)
            if img is not None:
                self.head_images.append(img)
                self._check_camera_ready()
        except Exception as e:
            self.get_logger().error(f"头部图像错误: {e}")
    
    def left_image_callback(self, msg):
        try:
            img = self._decode_compressed_image(msg)
            if img is not None:
                self.left_images.append(img)
                self._check_camera_ready()
        except Exception as e:
            self.get_logger().error(f"左侧图像错误: {e}")
    
    def right_image_callback(self, msg):
        try:
            img = self._decode_compressed_image(msg)
            if img is not None:
                self.right_images.append(img)
                self._check_camera_ready()
        except Exception as e:
            self.get_logger().error(f"右侧图像错误: {e}")
    
    def _check_robot_ready(self):
        if self.left_joint_state and self.right_joint_state and not self.robot_state_ready:
            self.robot_state_ready = True
            self.get_logger().info("✓ 机器人状态就绪")
    
    def _check_camera_ready(self):
        if self.head_images and self.left_images and self.right_images and not self.camera_ready:
            self.camera_ready = True
            self.get_logger().info("✓ 相机就绪")
    
    # ==================== 状态获取 ====================
    
    def get_eef_pose_6d(self, pose_msg: Optional[PoseStamped]) -> np.ndarray:
        """从 PoseStamped 消息中提取 6D 位姿 [x, y, z, roll, pitch, yaw]
        
        四元数转欧拉角（ZYX 顺序）
        """
        if pose_msg is None:
            return np.zeros(6, dtype=np.float32)
        
        try:
            import math
            pose = pose_msg.pose
            
            # 位置
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            
            # 四元数
            qx, qy, qz, qw = (pose.orientation.x, pose.orientation.y, 
                              pose.orientation.z, pose.orientation.w)
            
            # 四元数 -> 欧拉角 (ZYX 顺序)
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
        except Exception:
            return np.zeros(6, dtype=np.float32)
    
    def get_ordered_joint_positions(self, joint_state, prefix):
        """从 JointState 消息中提取有序的关节位置
        
        注意：Franka 控制器发布的关节名称是 'fr3_joint1' 到 'fr3_joint7'（没有 left/right 前缀）
        """
        positions = np.zeros(7, dtype=np.float32)
        for i, name in enumerate(joint_state.name):
            for j in range(1, 8):
                # 匹配 'fr3_joint1' 格式（不带 left/right 前缀）
                if f'fr3_joint{j}' == name or f'fr3_joint{j}' in name:
                    positions[j-1] = joint_state.position[i]
                    break
        return positions
    
    def get_robot_state(self):
        if not self.robot_state_ready:
            return None
        if not (self.left_gripper_state and self.left_gripper_state.position):
            return None  # 夹爪状态不可用，跳过本次推理
        if not (self.right_gripper_state and self.right_gripper_state.position):
            return None
        try:
            left_arm = self.get_ordered_joint_positions(self.left_joint_state, 'left_fr3')
            right_arm = self.get_ordered_joint_positions(self.right_joint_state, 'right_fr3')
            
            left_gripper = np.array([self.left_gripper_state.position[0] / 0.04], dtype=np.float32) \
                # if self.left_gripper_state and self.left_gripper_state.position else np.array([0.5], dtype=np.float32)
            right_gripper = np.array([self.right_gripper_state.position[0] / 0.04], dtype=np.float32) \
                # if self.right_gripper_state and self.right_gripper_state.position else np.array([0.5], dtype=np.float32)
            
            # EEF 位姿（如果启用且数据可用）
            if self.use_eef_pose:
                left_eef = self.get_eef_pose_6d(self.left_eef_pose)
                right_eef = self.get_eef_pose_6d(self.right_eef_pose)
            else:
                left_eef = np.zeros(6, dtype=np.float32)
                right_eef = np.zeros(6, dtype=np.float32)
            
            return {
                'left_arm': left_arm, 'left_gripper': left_gripper,
                'left_eef': left_eef,
                'right_arm': right_arm, 'right_gripper': right_gripper,
                'right_eef': right_eef,
            }
        except Exception as e:
            self.get_logger().error(f"状态获取失败: {e}")
            return None
    
    def get_current_joint_positions(self, side) -> Optional[np.ndarray]:
        try:
            if side == 'left' and self.left_joint_state:
                return self.get_ordered_joint_positions(self.left_joint_state, 'left_fr3')
            elif side == 'right' and self.right_joint_state:
                return self.get_ordered_joint_positions(self.right_joint_state, 'right_fr3')
        except:
            pass
        return None
    
    def build_observation(self):
        if not (self.robot_state_ready and self.camera_ready):
            return None
        try:
            state = self.get_robot_state()
            if state is None:
                return None
            
            state_obs = {k: v[np.newaxis, np.newaxis, ...] for k, v in state.items()}
            video_obs = {
                "head": self.head_images[-1][np.newaxis, np.newaxis, ...].astype(np.uint8),
                "left": self.left_images[-1][np.newaxis, np.newaxis, ...].astype(np.uint8),
                "right": self.right_images[-1][np.newaxis, np.newaxis, ...].astype(np.uint8),
            }
            return {
                "state": state_obs, 
                "video": video_obs,
                "language": {"annotation.human.task_description": [[self.task_description]]}
            }
        except Exception as e:
            self.get_logger().error(f"构建观测失败: {e}")
            return None
    
    # ==================== 核心：动作发布 ====================
    
    # ==================== 夹爪控制 ====================
    
    def publish_gripper(self, side, value):
        """
        发布夹爪控制命令
        
        Args:
            side: 'left' 或 'right'
            value: 夹爪值（0-1 范围，0=关闭，1=打开）
        """
        try:
            # 限制在 0-1 范围内
            value = float(max(0.0, min(1.0, value)))
            
            msg = Float32()
            msg.data = value
            
            if side == 'left':
                self.left_gripper_pub.publish(msg)
            else:
                self.right_gripper_pub.publish(msg)
            
            self.gripper_state[side]['last_value'] = value
                
        except Exception as e:
            self.get_logger().error(f"发布{side}夹爪命令失败: {e}")
    
    # ==================== 数据保存 ====================
    
    def save_inference_records(self):
        """保存推理记录到文件"""
        if not self.inference_records:
            self.get_logger().info("没有推理记录需要保存")
            return
        
        timestamp = int(time.time())
        output_dir = "inference_records"
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为可序列化格式
        records_list = []
        for r in self.inference_records:
            records_list.append({
                'inference_id': r.inference_id,
                'timestamp': r.timestamp,
                'inference_time_ms': r.inference_time_ms,
                'current_state': {
                    'left_arm': r.left_arm_state,
                    'right_arm': r.right_arm_state,
                    'left_gripper': r.left_gripper_state,
                    'right_gripper': r.right_gripper_state,
                },
                'model_output': {
                    'left_arm': r.model_output_left_arm,
                    'right_arm': r.model_output_right_arm,
                    'left_gripper': r.model_output_left_gripper,
                    'right_gripper': r.model_output_right_gripper,
                },
                'sent_actions': {
                    'left_arm': r.sent_left_arm,
                    'right_arm': r.sent_right_arm,
                    'left_gripper': r.sent_left_gripper,
                    'right_gripper': r.sent_right_gripper,
                },
            })
        
        output_data = {
            'metadata': {
                'task': self.task_description,
                'timestamp': timestamp,
                'total_inferences': len(self.inference_records),
                'total_steps': self.step_count,
                'min_queue_threshold': self.min_queue_threshold,
                'action_duration': self.action_duration,
            },
            'records': records_list,
        }
        
        output_file = os.path.join(output_dir, f"groot_inference_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.get_logger().info(f"✓ 推理记录已保存到: {output_file}")
        self.get_logger().info(f"  共 {len(self.inference_records)} 次推理")
    
    # ==================== GR00T 连接 ====================
    
    def connect_to_groot(self):
        try:
            self.groot_client = PolicyClient(
                self.groot_host, self.groot_port, 
                timeout_ms=15000, strict=False
            )
            if self.groot_client.ping():
                self.get_logger().info("✓ 连接 GR00T 成功")
                self.groot_client.reset()
                self.groot_connected = True
                return True
            return False
        except Exception as e:
            self.get_logger().error(f"连接失败: {e}")
            return False
    
    # ==================== 异步推理辅助方法 ====================
    
    def _build_rtc_options(self, current_queue_size: int):
        """构建 RTC frozen prefix 选项字典。

        Args:
            current_queue_size: 调用时动作队列中的剩余帧数。
                用于计算当前 chunk 已消耗了多少步，从而定位 frozen_prefix
                在 normalized_pred 中的正确偏移。
        """
        if not self.rtc_enabled or self._rtc_prev_normalized_pred is None:
            return None

        H = self._rtc_prev_normalized_pred.shape[1]  # 上一次 chunk 的总步数
        consumed = max(0, H - current_queue_size)  # 已消耗步数

        # 剩余部分对应 normalized_pred[:, consumed:, :]
        remaining = self._rtc_prev_normalized_pred[:, consumed:, :]  # (B, queue_size, D)
        available = remaining.shape[1]

        if self.rtc_freeze_steps < 0:
            K = available
        elif self.rtc_freeze_steps == 0:
            K = 0
        else:
            K = min(self.rtc_freeze_steps, available)

        if K > 0:
            frozen = remaining[:, :K, :]  # (B, K, D)
            if not hasattr(self, '_rtc_logged'):
                self.get_logger().info(
                    f"[RTC] frozen_prefix: K={K} 步 "
                    f"(已消耗 {consumed}, 可用 {available}, 设置 {self.rtc_freeze_steps}), "
                    f"beta={self.rtc_beta}, mask_decay={self.rtc_mask_decay}"
                )
                self._rtc_logged = True
            return {
                "rtc": {
                    "enabled": True,
                    "frozen_prefix": frozen,
                    "beta": self.rtc_beta,
                    "mask_decay": self.rtc_mask_decay,
                }
            }
        return None
    
    def _run_inference(self, obs, rtc_options):
        """执行单次推理调用（可在后台线程安全运行）"""
        inference_start = time.time()
        action, info = self.groot_client.get_action(obs, rtc_options)
        inference_time = (time.time() - inference_start) * 1000
        return action, info, inference_time

    # ==================== 独立发布线程 ====================

    def _publish_single_action(self, action_tuple):
        """发布单帧动作到机器人（由发布线程调用）"""
        left_arm, right_arm, left_gripper_val, right_gripper_val = action_tuple

        left_arm_msg = JointState()
        left_arm_msg.header = Header()
        left_arm_msg.header.stamp.sec = 0
        left_arm_msg.header.stamp.nanosec = 0
        left_arm_msg.name = self.left_arm_joint_names
        left_arm_msg.position = [float(p) for p in left_arm]

        right_arm_msg = JointState()
        right_arm_msg.header = Header()
        right_arm_msg.header.stamp.sec = 0
        right_arm_msg.header.stamp.nanosec = 0
        right_arm_msg.name = self.right_arm_joint_names
        right_arm_msg.position = [float(p) for p in right_arm]

        left_gripper_msg = Float32()
        left_gripper_msg.data = float(max(0.0, min(1.0, left_gripper_val)))

        right_gripper_msg = Float32()
        right_gripper_msg.data = float(max(0.0, min(1.0, right_gripper_val)))

        # 背靠背发布保证左右臂同步
        self.left_action_pub.publish(left_arm_msg)
        self.right_action_pub.publish(right_arm_msg)
        self.left_gripper_pub.publish(left_gripper_msg)
        self.right_gripper_pub.publish(right_gripper_msg)

    def _publisher_loop(self):
        """独立动作发布线程：以固定频率从队列消费并发布动作。

        该线程与推理线程完全解耦，机器人在推理期间也能继续执行动作，
        消除 chunk 边界卡顿。队列为空时保持最后一帧位置。
        """
        self.get_logger().info(
            f"[发布线程] 已启动，频率={1.0/self.action_duration:.0f}Hz"
        )
        next_time = time.monotonic()
        queue_empty_warned = False

        while self._publisher_running and rclpy.ok():
            action = None
            with self._queue_lock:
                if self._action_queue:
                    action = self._action_queue.popleft()
                    self._last_published_action = action
                    self.step_count += 1
                    queue_empty_warned = False

            if action is None:
                # 队列空：保持最后位置（不抖动）
                action = self._last_published_action
                if not queue_empty_warned and action is not None:
                    self.get_logger().warn("[发布线程] 队列已空，保持最后位置")
                    queue_empty_warned = True

            if action is not None:
                self._publish_single_action(action)

            # 精确定时：补偿发布耗时的累积误差
            next_time += self.action_duration
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()  # 已经落后，重置基准

    # ==================== Chunk 加载 ====================

    @staticmethod
    def _extract_gripper_value(gripper_data, idx):
        """从夹爪数据中提取标量值（直接输出，无窗口判断）"""
        if gripper_data is None:
            return 0.5
        val = gripper_data[idx]
        if hasattr(val, '__len__'):
            val = val[0]
        return float(max(0.0, min(1.0, val)))

    def _load_chunk_to_queue(self, action, info, queue_size_at_inference_start=None):
        """将新推理结果加载到动作队列。

        核心 RTC 时序逻辑：
        1. 推理开始时队列有 Q_before 帧（来自旧 chunk 的剩余）
        2. 推理期间发布线程继续消费，结束时队列剩 Q_now 帧
        3. consumed_during_inference = Q_before - Q_now
        4. 新 chunk 的前 K 步被 RTC 约束为匹配旧 chunk 的剩余
        5. 跳过已被消费的 consumed_during_inference 步，把剩余部分填入队列

        Returns:
            (chunk_size, skipped): 模型输出总步数 H 和跳过的步数
        """
        left_arm = action['left_arm'][0]    # (H, 7)
        right_arm = action['right_arm'][0]  # (H, 7)
        left_gripper = action.get('left_gripper', [None])[0]
        right_gripper = action.get('right_gripper', [None])[0]

        H = left_arm.shape[0]
        self._total_chunk_size = H

        # RTC: 缓存归一化预测
        if self.rtc_enabled:
            if isinstance(info, dict) and "normalized_action_pred" in info:
                self._rtc_prev_normalized_pred = info["normalized_action_pred"]
            else:
                self._rtc_prev_normalized_pred = None

        # 记录到 action_buffer（用于数据保存）
        self.action_buffer['left_arm'] = left_arm
        self.action_buffer['right_arm'] = right_arm
        self.action_buffer['left_gripper'] = left_gripper
        self.action_buffer['right_gripper'] = right_gripper

        # 构建动作帧列表
        new_actions = []
        for i in range(H):
            lg = self._extract_gripper_value(left_gripper, i)
            rg = self._extract_gripper_value(right_gripper, i)
            new_actions.append((left_arm[i], right_arm[i], lg, rg))

        # 原子替换队列：计算推理期间已消费步数并跳过
        with self._queue_lock:
            if queue_size_at_inference_start is not None:
                current_queue_size = len(self._action_queue)
                consumed_during_inference = queue_size_at_inference_start - current_queue_size
                skip = max(0, consumed_during_inference)
            else:
                skip = 0

            self._action_queue.clear()
            self._action_queue.extend(new_actions[skip:])

        return H, skip

    # ==================== 主循环 ====================
    
    def run_control_loop(self):
        """主控制循环 — 独立发布线程 + 连续动作队列架构。

        架构说明:
        ┌─────────────────────────────────────────────────────────┐
        │  发布线程 (_publisher_loop)                              │
        │  ───────────────────────                                │
        │  以固定频率 1/dt 从 _action_queue 取帧并 publish。       │
        │  队列为空时保持最后帧位置，永不停歇。                      │
        └─────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────┐
        │  主线程 (本方法)                                         │
        │  ───────────────────                                    │
        │  while True:                                            │
        │    ① 等待队列深度降到阈值 (H - s) → 该推理了              │
        │    ② 快照队列大小，构建观测 & RTC frozen_prefix           │
        │    ③ 阻塞式推理（期间发布线程继续消费旧 chunk）            │
        │    ④ 将新 chunk 加载到队列（跳过推理期间已消耗的步数）      │
        └─────────────────────────────────────────────────────────┘
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("Franka GR00T RTC 连续动作队列客户端")
        self.get_logger().info("=" * 60)

        # 启动后台 ROS 事件循环（确保及时接收消息）
        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()
        self.get_logger().info("✓ 后台 ROS 事件循环已启动")

        # 等待就绪
        while not self.robot_state_ready and rclpy.ok():
            time.sleep(0.1)
        while not self.camera_ready and rclpy.ok():
            time.sleep(0.1)

        if not self.connect_to_groot():
            self.get_logger().error("无法连接 GR00T")
            return

        # 打印配置
        self.get_logger().info(f"任务: {self.task_description}")
        self.get_logger().info(f"最小队列阈值: {self.min_queue_threshold} 帧（队列降至此值时触发推理）")
        self.get_logger().info(f"动作周期: {self.action_duration*1000:.0f} ms/步")
        self.get_logger().info("-" * 40)
        if self.use_eef_pose:
            self.get_logger().info("[EEF] 启用位姿输入")
        else:
            self.get_logger().info("[EEF] 未启用位姿输入（使用零向量）")
        if self.rtc_enabled:
            freeze_desc = (
                f"全冻结(剩余)" if self.rtc_freeze_steps < 0
                else f"不冻结" if self.rtc_freeze_steps == 0
                else f"{self.rtc_freeze_steps} 步"
            )
            self.get_logger().info(
                f"[RTC] 已启用 Real-Time Chunking，"
                f"freeze={freeze_desc}，"
                f"beta={self.rtc_beta}，mask_decay={self.rtc_mask_decay}"
            )
        else:
            self.get_logger().info("[RTC] 未启用（标准 chunk 推理）")
        self.get_logger().info("[架构] 独立发布线程 + 连续动作队列，消除 chunk 间隔卡顿")
        self.get_logger().info("-" * 60)

        # ═══════════ 首次推理（阻塞，填充队列） ═══════════
        obs = self.build_observation()
        while obs is None and rclpy.ok():
            time.sleep(0.01)
            obs = self.build_observation()

        action, info, first_inference_time = self._run_inference(obs, None)
        chunk_size, _ = self._load_chunk_to_queue(action, info)
        self.inference_count += 1
        self.get_logger().info(
            f"[首次推理] 推理={first_inference_time:.0f}ms, "
            f"chunk_size(H)={chunk_size}, 队列已填充 {chunk_size} 帧"
        )

        # ═══════════ 启动独立发布线程 ═══════════
        self._publisher_running = True
        publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        publisher_thread.start()
        self.get_logger().info("✓ 独立动作发布线程已启动")

        start_time = time.time()
        last_log_time = start_time

        try:
            while rclpy.ok():
                try:
                    # ═══════════ 1. 等待队列降至最小阈值 ═══════════
                    # 当队列快耗尽时触发推理，有效执行步数由推理延迟自然决定
                    while rclpy.ok():
                        with self._queue_lock:
                            qsize = len(self._action_queue)
                        if qsize <= self.min_queue_threshold:
                            break
                        time.sleep(self.action_duration * 0.5)

                    if not rclpy.ok():
                        break

                    # ═══════════ 2. 快照队列状态 ═══════════
                    with self._queue_lock:
                        queue_before = len(self._action_queue)

                    # ═══════════ 3. 构建观测 & RTC 选项 ═══════════
                    obs = self.build_observation()
                    if obs is None:
                        time.sleep(0.01)
                        continue

                    rtc_options = self._build_rtc_options(queue_before)

                    # ═══════════ 4. 执行推理（发布线程继续运行！） ═══════════
                    action, info, inference_time = self._run_inference(obs, rtc_options)

                    # ═══════════ 5. 加载新 chunk 到队列 ═══════════
                    chunk_size, skipped = self._load_chunk_to_queue(
                        action, info, queue_before
                    )
                    self.inference_count += 1

                    # ═══════════ 6. 日志 & 数据记录 ═══════════
                    with self._queue_lock:
                        queue_after = len(self._action_queue)

                    self.get_logger().info(
                        f"[推理 #{self.inference_count}] "
                        f"推理={inference_time:.0f}ms "
                        f"跳过={skipped}步 "
                        f"队列={queue_before}→{queue_after}"
                    )

                    if self.save_actions:
                        current_state = self.get_robot_state()
                        if current_state:
                            record = InferenceRecord(
                                inference_id=self.inference_count,
                                timestamp=time.time(),
                                inference_time_ms=inference_time,
                                left_arm_state=current_state['left_arm'].tolist(),
                                right_arm_state=current_state['right_arm'].tolist(),
                                left_gripper_state=float(current_state['left_gripper'][0]),
                                right_gripper_state=float(current_state['right_gripper'][0]),
                                model_output_left_arm=self.action_buffer['left_arm'].tolist(),
                                model_output_right_arm=self.action_buffer['right_arm'].tolist(),
                                model_output_left_gripper=self.action_buffer['left_gripper'].tolist() if self.action_buffer['left_gripper'] is not None else [],
                                model_output_right_gripper=self.action_buffer['right_gripper'].tolist() if self.action_buffer['right_gripper'] is not None else [],
                            )
                            self.inference_records.append(record)

                    # 定期状态日志
                    now = time.time()
                    if now - last_log_time >= 5.0:
                        elapsed = now - start_time
                        with self._queue_lock:
                            qs = len(self._action_queue)
                        self.get_logger().info(
                            f"[状态] 步数={self.step_count}, 推理={self.inference_count}, "
                            f"频率={self.step_count/elapsed:.1f}Hz, 队列深度={qs}"
                        )
                        last_log_time = now

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.get_logger().error(f"错误: {e}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    time.sleep(0.1)
        finally:
            self._publisher_running = False

        # 结束统计
        elapsed = time.time() - start_time
        self.get_logger().info(f"\n推理完成:")
        self.get_logger().info(f"  总耗时: {elapsed:.2f} 秒")
        self.get_logger().info(f"  总步数: {self.step_count}")
        self.get_logger().info(f"  总推理次数: {self.inference_count}")
        if elapsed > 0:
            self.get_logger().info(f"  平均频率: {self.step_count/elapsed:.2f} Hz")

        # 保存推理记录
        if self.save_actions:
            self.save_inference_records()

def main():
    parser = argparse.ArgumentParser(description="Franka GR00T Chunk 模式客户端")
    parser.add_argument("--groot-host", default="localhost")
    parser.add_argument("--groot-port", type=int, default=5555)
    parser.add_argument("--action-duration", type=float, default=0.017,
                        help="每个动作执行时间（秒）")
    parser.add_argument("--task", default="Catch duck")
    parser.add_argument("--min-queue-threshold", type=int, default=20,
                        help="队列帧数降至此阈值时触发新推理（默认20帧，预留推理期间的缓冲）")
    
    # EEF 位姿（可选）
    parser.add_argument("--use-eef-pose", action="store_true",
                        help="启用 EEF 位姿输入（订阅 /left/franka/eef_pose, /right/franka/eef_pose）")
    
    # RTC (Real-Time Chunking)
    # parser.add_argument("--rtc", action="store_true",
    #                     help="启用 Real-Time Chunking (RTC)，使相邻 chunk 边界平滑过渡")
    parser.add_argument("--rtc-freeze-steps", type=int, default=20,
                        help="RTC 冻结步数。-1=冻结所有剩余步(H-s)；0=不冻结(禁用推理RTC)；"
                             "正整数=冻结指定步数(不超过H-s)。推荐值: H-s 或 (H-s)//2")
    parser.add_argument("--rtc-beta", type=float, default=5.0,
                        help="RTC 引导权重裁剪值 β (Eq. 2)。设为 0 则退化为仅 hard replacement。"
                             "推荐值: 5.0")
    parser.add_argument("--rtc-mask-decay", type=float, default=2.0,
                        help="RTC soft mask 指数衰减率 (Eq. 5)。控制冻结区域内各步的引导权重分布。"
                             "推荐值: 2.0")
    
    args = parser.parse_args()
    rclpy.init()
    
    try:
        node = FrankaGr00tChunkClient(
            groot_host=args.groot_host,
            groot_port=args.groot_port,
            action_duration=args.action_duration,
            task_description=args.task,
            min_queue_threshold=args.min_queue_threshold,
            use_eef_pose=args.use_eef_pose,
            rtc_enabled=(args.rtc_freeze_steps != 0),
            rtc_freeze_steps=args.rtc_freeze_steps,
            rtc_beta=args.rtc_beta,
            rtc_mask_decay=args.rtc_mask_decay,
        )
        node.run_control_loop()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
