#!/usr/bin/env bash
# Round4-B1 三域 pilot（登记于 reviews/V3_PR1_round4_prereg_b1_amendment.md，跑之前写死）。
# 三臂只改"表示域"：linear = PD/s、log1p = log1p(PD/s)、logeps = log(u + 1e-3)。
# 其余（网络/头/损失/patch/seed/数据划分/steps/epochs）全部相同。
# 预算：3 作业 × 15ep × 500step ≈ 0.63 GPU·h（Round4 累计 ≈0.78/8，作业 3/8）。
set -u
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
OUT=runs/v3_round4
mkdir -p "$OUT"
COMMON="--patch 32 64 64 --epochs 15 --steps-per-epoch 500 --features 16 --groups 4 --seed 3407 \
  --lr 1e-4 --lr-schedule poly --pairing within --mode dense \
  --arch-head residual --zero-init-final"

run() {  # $1=gpu $2=arm
  local arm=$2 D=data/processed/v3_round4_b1/$2
  local VOLS CLEANS
  VOLS=$(ls $D/train/*_x.npy | sort | tr '\n' ' ')
  CLEANS=$(ls $D/train/*_y.npy | sort | tr '\n' ' ')
  echo "[$(date '+%F %T')] start B1_${arm}_s3407 on GPU$1"
  CUDA_VISIBLE_DEVICES=$1 $PY -u train.py $COMMON \
    --train-volumes $VOLS --train-clean-volumes $CLEANS \
    --out "$OUT/b1_${arm}_s3407" --arm-label "B1_${arm}_s3407" \
    > "$OUT/b1_${arm}_s3407.log" 2>&1
  echo "[$(date '+%F %T')] done B1_${arm}_s3407 rc=$?"
}

run 1 linear &
run 2 log1p &
wait
run 0 logeps &
wait
echo "[$(date '+%F %T')] B1 pilot 三臂全部结束"
