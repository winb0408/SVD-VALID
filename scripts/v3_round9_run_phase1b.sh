#!/usr/bin/env bash
# Round9 / Phase 1B：A、B 两臂各一条训练作业。
#
# 全部设置来自 `reviews/V3_PR1_round9_prereg.md` §4（写于任何训练之前）：
#   A = VALID-faithful random（每块独立抽 tet，一个 patch 一个全局角色置换）
#   B = 最强可行固定 tet（校准选出 pattern 5, role_order 0 1 3 2）
# 两臂除采样几何外**逐位相同**：patch、步数、lr、schedule、clip、损失权重、
# 网络、seed、数据都一致。禁止改这里的任何数字去"追结果"。
#
# 用法：bash scripts/v3_round9_run_phase1b.sh A 2      # 臂名 GPU 号
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?usage: run_phase1b.sh <A|B> <gpu>}"
GPU="${2:?usage: run_phase1b.sh <A|B> <gpu>}"
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

TRAIN=(data/processed/v3_round2_s0/low/train/s0_low_s200_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s201_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s202_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s203_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s204_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s205_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s206_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s207_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s208_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s209_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s210_log1p.npy
       data/processed/v3_round2_s0/low/train/s0_low_s211_log1p.npy)

COMMON=(--pairing within --train-volumes "${TRAIN[@]}"
        --patch 32 64 64 --epochs 50 --steps-per-epoch 200
        --lr 1e-4 --lr-schedule poly --clip-grad 20.0
        --weight-reg 1e-4 --weight-identity 1.0
        --features 16 --groups 4 --seed 3407 --arch-head direct
        --device cuda)

case "$ARM" in
  A) MODE=(--mode random --role-perm global) ;;
  B) MODE=(--mode fixed --pattern 5 --role-order 0 1 3 2) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

OUT="runs/v3_round9/$ARM"
mkdir -p "$OUT"
echo "[$(date -Is)] arm=$ARM gpu=$GPU -> $OUT" | tee -a "$OUT/run.log"
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /usr/bin/time -v "$PY" -u train.py "${MODE[@]}" "${COMMON[@]}" \
  --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo "[$(date -Is)] arm=$ARM done rc=$?" | tee -a "$OUT/run.log"
