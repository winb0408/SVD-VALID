#!/usr/bin/env bash
# Round5：采样创新 vs 标准 SN2N 的直接对照（预注册见
# reviews/V3_PR1_round5_prereg_sampling_direct.md，**跑之前写死**）。
#
# 三条臂**全部自监督**（--pairing cross，不给 --train-clean-volumes）：
#   S_log   标准 VALID：m=1 随机 tetris + cross
#   O_log   我们的：m=2 + 体对角角色对（Phase 2 预注册 --slots 1 4 6 3）
#   O_lin   同 O_log，但域为 linear
# 网络/损失/预算/数据/seed 全部相同；域数据直接复用 B1 已准备好的 *_x.npy。
#
# 预算（预注册 §6）：3 作业 × 15ep × 500step ≈ 0.67 GPU·h/作业。
set -u
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
OUT=runs/v3_round5
mkdir -p "$OUT"

COMMON="--patch 32 64 64 --epochs 15 --steps-per-epoch 500 --features 16 --groups 4 \
  --seed 3407 --lr 1e-4 --lr-schedule poly --arch-head residual --zero-init-final"

run() {  # $1=gpu $2=arm-name $3=domain $4...=extra args
  local gpu=$1 name=$2 dom=$3; shift 3
  local D=data/processed/v3_round4_b1/$dom/train
  local VOLS; VOLS=$(ls "$D"/*_x.npy | sort | tr '\n' ' ')
  echo "[$(date '+%F %T')] start ${name} on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train.py $COMMON \
    --train-volumes $VOLS "$@" \
    --out "$OUT/${name}_s3407" --arm-label "${name}_s3407" \
    > "$OUT/${name}_s3407.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] done ${name} rc=$rc"
  return $rc
}

# 三条臂并行：S_log 与 O_lin 走 GPU0/1，O_log 走 GPU2（GPU3 属他人，不用）
run 0 S_log   logeps --stride 1 --mode random --pairing cross &
P0=$!
run 1 O_lin   linear --stride 2 --mode fixed --slots 1 4 6 3 --pairing cross &
P1=$!
run 2 O_log   logeps --stride 2 --mode fixed --slots 1 4 6 3 --pairing cross &
P2=$!

rc=0
wait $P0 || rc=1
wait $P1 || rc=$((rc|2))
wait $P2 || rc=$((rc|4))
echo "[$(date '+%F %T')] Round5 三臂结束 rc=$rc"
exit $rc
