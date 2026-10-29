#!/usr/bin/env bash
# 实验 B / R04：三种表示对照（linear / log1p / logeps），其余全部与实验 A 阶段 2 相同。
#
# 前置条件：reports/v3_expA_v2/prereg.md §3.3 的 T2 门通过。
# 门槛：reports/v3_expB/prereg.md §4（写在结果之前）。
# 数据：data/processed/v3_sim_v2/repB/{rep}_train_s{100..115}.npy（共享尺度 s_glob）。
#
# 6 个 run = 3 表示 × 2 seed；4 张卡分两波（第一波 4 个，第二波 2 个），
# 每个 run 与实验 A 同样的 200k step / patch 32x64x64 / poly LR / clip 100。
set -u
cd /ssd1/wjb/SVD-VALID || exit 1
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
D=runs/v3_expB_200k
mkdir -p "$D"

vols () { ls data/processed/v3_sim_v2/repB/$1_train_s1*.npy | sort; }

run () {  # gpu rep seed
  local gpu=$1 rep=$2 seed=$3
  local tag="B_${rep}_s${seed}"
  local V; V=$(vols "$rep")
  echo "[launch] $tag on GPU$gpu  ($(echo "$V" | wc -l) volumes)"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -u train.py --train-volumes $V \
      --stride 1 --patch 32 64 64 --epochs 200 --steps-per-epoch 1000 \
      --features 16 --seed "$seed" --out "$D/$tag" \
      --pairing within --mode random --lr-schedule poly \
      --sampler-config configs/sampler_config.json \
      >> "$D/${tag}.log" 2>&1
}

# 第一波：3 个表示 × seed 3407，加 1 个 seed 3408
run 0 linear 3407 &
run 1 log1p  3407 &
run 2 logeps 3407 &
run 3 linear 3408 &
wait
echo "WAVE1_DONE $(date -Is)"

# 第二波：剩下两个 seed
run 0 log1p  3408 &
run 1 logeps 3408 &
wait
echo "ALL_DONE $(date -Is)"

# 训练结束立刻评价（端点与门槛见 reports/v3_expB/prereg.md）。失败不在这里重试，日志留档。
$PY -u scripts/v3_expB_eval.py --runs "$D" --gpu 0 --out reports/v3_expB/metrics.json \
    > "$D/eval.log" 2>&1
echo "EVAL_DONE rc=$? $(date -Is)"

# 判据（T-B1…T-B4）机械化，门槛照抄 prereg §4，不得事后调整。
$PY -u scripts/v3_expB_gate.py --metrics reports/v3_expB/metrics.json > "$D/gate.log" 2>&1
echo "GATE_DONE rc=$? $(date -Is)"
