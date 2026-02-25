set -x -e

export NUM_GPUS=4

# ============ 多卡训练 NCCL 配置 ============
export NCCL_P2P_DISABLE=0       # 启用 P2P (GPU 之间直接通信，走 PCIe/NVLink)
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand (无硬件)
export NCCL_SHM_DISABLE=0      # 启用共享内存
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 确保 GPU 顺序一致
export NCCL_DEBUG=INFO          # 查看 NCCL 调试信息
# export NCCL_ALGO=Ring         # 可选：使用 Ring 算法 

# ============ 启动多卡训练 ============
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
  /workspace/gr00t/gr00t/experiment/launch_finetune.py \
  --base-model-path /workspace/gr00t/nvidia/GR00T-N1.6-3B \
  --dataset-path /workspace1/dataset/franka_20260129_Catch_duck_action_v2.1 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/gr00t/examples/FRANKA/modality.py \
  --num-gpus $NUM_GPUS \
  --output-dir /workspace1/gr00t/franka_checkpoints_${NUM_GPUS}GPU_eef_chunk50 \
  --save-total-limit 8 \
  --save-steps 10000 \
  --max-steps 200000 \
  --warmup-ratio 0.08 \
  --global-batch-size 192 \
  --learning-rate 8e-5 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  --training-rtc-max-latency 24 \
  --use-wandb \
  > /workspace1/train_full.log 2>&1
