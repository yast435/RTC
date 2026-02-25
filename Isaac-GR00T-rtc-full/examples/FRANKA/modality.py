"""
Franka 双臂机械臂的 GR00T 模型训练配置文件，包含EEF
- 双臂 Franka 机械臂
- 28维状态: 左臂关节(7) + 左夹爪(1) + 右臂关节(7) + 右夹爪(1) + 左臂末端位姿(6) + 右臂末端位姿(6)
- 28维动作: 对应的命令值
- 3个摄像头: camera_left, camera_right, camera_head
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Franka 双臂配置（包含EEF）
franka_dual_arm_config = {
    # 视频配置: 使用三个摄像头
    "video": ModalityConfig(
        delta_indices=[0],  # 只使用当前帧
        modality_keys=[
            "left",   # 左侧摄像头 (对应 observation.images.camera_left)
            "right",  # 右侧摄像头 (对应 observation.images.camera_right)
            "head",   # 头部摄像头 (对应 observation.images.camera_head)
        ],
    ),
    # 状态配置: 包含双臂关节、夹爪和末端位姿
    "state": ModalityConfig(
        delta_indices=[0],  # 只使用当前状态
        modality_keys=[
            "left_arm",      # 左臂关节角度 (7维, 索引 0-6)
            "left_gripper",  # 左夹爪状态 (1维, 索引 7)
            "right_arm",     # 右臂关节角度 (7维, 索引 8-14)
            "right_gripper", # 右夹爪状态 (1维, 索引 15)
            "left_eef",      # 左臂末端执行器位姿 (6维, 索引 16-21)
            "right_eef",     # 右臂末端执行器位姿 (6维, 索引 22-27)
        ],
    ),
    # 动作配置: 60步动作预测
    "action": ModalityConfig(
        delta_indices=list(range(50)),  # 60步动作预测horizon
        modality_keys=[
            "left_arm",      # 左臂关节命令 (7维, 索引 0-6)
            "left_gripper",  # 左夹爪命令 (1维, 索引 7)
            "right_arm",     # 右臂关节命令 (7维, 索引 8-14)
            "right_gripper", # 右夹爪命令 (1维, 索引 15)
            "left_eef",      # 左臂末端执行器命令 (6维, 索引 16-21)
            "right_eef",     # 右臂末端执行器命令 (6维, 索引 22-27)
        ],
        action_configs=[
            # 左臂关节动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # 绝对位置控制
                type=ActionType.NON_EEF,            # 关节空间
                format=ActionFormat.DEFAULT,
            ),
            # 左夹爪动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 右臂关节动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 右夹爪动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            
            ),
                        # 左臂末端执行器动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,               # 笛卡尔空间
                format=ActionFormat.DEFAULT,       # xyz + roll/pitch/yaw
            ),
            # 右臂末端执行器动作配置
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # 语言标注配置
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# 注册配置到 GR00T 系统
register_modality_config(franka_dual_arm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

