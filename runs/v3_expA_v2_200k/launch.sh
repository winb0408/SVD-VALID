#!/bin/bash
# 阶段 2：v2 仿真（16 个体训练 + 2 个独立体评估，R06 修复后）
# 与阶段 1 唯一的差别是数据（--train-volumes 16 卷），网络/预算/臂完全相同。
set -e
cd /ssd1/wjb/SVD-VALID
mkdir -p runs/v3_expA_v2_200k
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
VOLS=$(ls data/processed/v3_sim_v2/train/*_log1p.npy)
echo "train volumes: $(echo "$VOLS" | wc -l)"
run () {  # $1=gpu $2=arm $3=seed $4..=extra args
  local gpu=$1 arm=$2 seed=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -u train.py --train-volumes $VOLS \
      --stride 1 --patch 32 64 64 --epochs 200 --steps-per-epoch 1000 \
      --features 16 --seed $seed --out runs/v3_expA_v2_200k/${arm}_s${seed} \
      --lr-schedule poly --sampler-config configs/sampler_config.json \
      "$@" >> runs/v3_expA_v2_200k/${arm}_s${seed}.log 2>&1
}
run 0 A_rand   3407 --pairing within --mode random &
run 1 A_fixed4 3407 --pairing within --mode fixed --pattern 4 &
run 2 A_rand   3408 --pairing within --mode random &
run 3 A_fixed4 3408 --pairing within --mode fixed --pattern 4 &
wait
echo ALL_DONE
