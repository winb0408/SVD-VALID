#!/bin/bash
set -e
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
run () {  # $1=gpu $2=arm $3=seed $4..=extra args
  local gpu=$1 arm=$2 seed=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -u train.py --train-volume data/processed/v3_sim/train_log1p.npy \
      --stride 1 --patch 32 64 64 --epochs 200 --steps-per-epoch 1000 \
      --features 16 --seed $seed --out runs/v3_expA_200k/${arm}_s${seed} \
      --lr-schedule poly "$@" >> runs/v3_expA_200k/${arm}_s${seed}.log 2>&1
}
run 0 A_rand   3407 --pairing within --mode random &
run 1 A_fixed4 3407 --pairing within --mode fixed --pattern 4 &
run 2 A_rand   3408 --pairing within --mode random &
run 3 A_fixed4 3408 --pairing within --mode fixed --pattern 4 &
wait
echo ALL_DONE
