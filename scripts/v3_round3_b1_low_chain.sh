#!/usr/bin/env bash
# Round3 B1 的 low 档两臂（见 reviews/V3_PR1_round3_amended_plan.md §4.4，登记于结果之前）。
# 与 §4.2 完全同预算/同种子/同损失，只把 vlow 换成 low。
# 第一臂立刻在 GPU0 上跑；第二臂等 vlow 两臂结束后在 GPU1 上跑（避免三臂抢同一张卡）。
set -u
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
D=data/processed/v3_round2_s0/low
OUT=runs/v3_round3/b1_low
mkdir -p "$OUT"
VOLS=$(ls $D/train/*_log1p.npy | tr '\n' ' ')
CLEANS=$(ls $D/train/*_clean.npy | tr '\n' ' ')
COMMON="--patch 32 64 64 --epochs 40 --steps-per-epoch 500 --features 16 --seed 3407 \
  --lr-schedule poly --pairing within --mode dense --train-volumes $VOLS --train-clean-volumes $CLEANS"

run() {  # $1=gpu $2=name $3=extra
  echo "[$(date '+%F %T')] start $2 on GPU$1"
  CUDA_VISIBLE_DEVICES=$1 $PY -u train.py $COMMON --out $OUT/$2 --arch-head residual \
    --arm-label "$2" $3 > "$OUT/$2.log" 2>&1
  echo "[$(date '+%F %T')] done $2 rc=$?"
}

run 0 B1_low_residual_s3407 ""

# 等 vlow 两臂都写出 history.json（训练结束的标志）
while [ ! -f runs/v3_round3/b1_vlow/B1_residual_s3407/history.json ] || \
      [ ! -f runs/v3_round3/b1_vlow/B1_zeroinit_s3407/history.json ]; do sleep 60; done
echo "[$(date '+%F %T')] vlow 两臂已结束，启动 low zeroinit"

run 1 B1_low_zeroinit_s3407 "--zero-init-final"
echo "[$(date '+%F %T')] chain 全部结束"
