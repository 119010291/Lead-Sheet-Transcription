#!/bin/bash
#SBATCH --job-name=Juke_train  # 任务名称
#SBATCH --gres=gpu:1                # 请求 1 个 GPU
#SBATCH --mem=32G                   # 请求 32GB 内存
#SBATCH --cpus-per-task=8           # 请求 8 个 CPU 核心
#SBATCH --time=0-24:00:00           # 最大运行时间 (24 小时)
#SBATCH -o logs/juke_mel_refinedSub_TVsame_mask.out     # 标准输出日志
#SBATCH -e logs/juke_mel_refinedSub_TVsame_mask.err     # 错误输出日志

echo "=== Starting Training Job ==="

# 设置环境变量（按需保留/修改）
export PATH=/home/dw3180/ffmpeg_static/ffmpeg-7.0.2-amd64-static:$PATH
export PYTHONPATH=/sheetsage/sheetsage:$PYTHONPATH

# 构造训练命令
CMD="
# 1) 激活你的虚拟环境
source /home/dw3180/my_sheetsage/bin/activate

# 2) 检查当前 python & pytorch lightning 版本
echo 'Python:' \$(which python)
echo 'Lightning version:' \$(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')

# 3) 运行训练脚本
CUDA_LAUNCH_BLOCKING=1 python /sheetsage/calmdown/train_sub.py \
  --config /sheetsage/sheetsage/config_sub.json \
  --gpus 1 \
  --batch_size 64 
"

# 使用 Singularity 执行，上面 CMD 的内容都会在容器内执行
singularity exec --nv --cleanenv \
  --overlay /scratch/dw3180/overlay-15GB-500K.ext3:ro \
  --bind /home/dw3180/sheetsage_code:/sheetsage \
  --bind /home/dw3180/sheetsage_code/.sheetsage:/sheetsage/cache \
  --bind /scratch/dw3180/sheetsage_project/output:/sheetsage/output \
  /home/dw3180/sheetsage_latest.sif \
  bash -c "$CMD"

echo "=== Training Job Complete ==="