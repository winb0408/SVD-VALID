#!/usr/bin/env bash
# 实验 C / R02+R03：采样角色与物理先验的四臂对照。
#
# 前置条件：reports/v3_expA_v2/prereg.md §3.3 的 T2 门通过。
# 门槛：reports/v3_expC/prereg.md §4（写在结果之前）。
# 事前结论：轴错位这条负控制已被证伪为空操作（prior_arms.json），故只跑四臂。
#
# 四臂（全部 cross 配对，覆盖率与坐标可核对）：
#   C_correct    --mode fixed --slots 0 1 7 6   实测先验推出的体对角
#   C_wrong_iso  --mode fixed --slots 2 7 5 0   各向同性先验推出（仅 SA64 有区分力）
#   C_role_swap  --mode fixed --slots 0 1 6 7   同 4 个体素，只把两个目标对调
#   C_random     --mode random                  每步随机抽 tet（上游做法）
#
# 4 臂 x 2 seed = 8 run，两波；预算与实验 A 阶段 2 完全相同（200k step）。
set -u
cd /ssd1/wjb/SVD-VALID || exit 1
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
D=runs/v3_expC_200k
mkdir -p "$D"
VOLS=$(ls data/processed/v3_sim_v2/train/*_log1p.npy | sort)

run () {  # gpu tag seed extra-args...
  local gpu=$1 tag=$2 seed=$3; shift 3
  echo "[launch] $tag on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -u train.py --train-volumes $VOLS \
      --stride 1 --patch 32 64 64 --epochs 200 --steps-per-epoch 1000 \
      --features 16 --seed "$seed" --out "$D/${tag}_s${seed}" \
      --pairing cross --lr-schedule poly \
      --sampler-config configs/sampler_config.json "$@" \
      >> "$D/${tag}_s${seed}.log" 2>&1
}

run 0 C_correct   3407 --mode fixed --slots 0 1 7 6 &
run 1 C_wrong_iso 3407 --mode fixed --slots 2 7 5 0 &
run 2 C_role_swap 3407 --mode fixed --slots 0 1 6 7 &
run 3 C_random    3407 --mode random &
wait
echo "WAVE1_DONE $(date -Is)"

run 0 C_correct   3408 --mode fixed --slots 0 1 7 6 &
run 1 C_wrong_iso 3408 --mode fixed --slots 2 7 5 0 &
run 2 C_role_swap 3408 --mode fixed --slots 0 1 6 7 &
run 3 C_random    3408 --mode random &
wait
echo "ALL_DONE $(date -Is)"

# 训练结束立刻评价 + 判据（端点与门槛见 reports/v3_expC/prereg.md §3/§4）。
# 端点与实验 A 阶段 2 完全一致，所以直接复用 v3_expA_v2_eval.py，只换臂名与输出路径。
$PY -u scripts/v3_expA_v2_eval.py --runs "$D" --gpu 0 \
    --arms C_correct C_wrong_iso C_role_swap C_random \
    --out reports/v3_expC/metrics.json > "$D/eval.log" 2>&1
echo "EVAL_DONE rc=$? $(date -Is)"
$PY -u scripts/v3_expC_gate.py --metrics reports/v3_expC/metrics.json \
    > "$D/gate.log" 2>&1
echo "GATE_DONE rc=$? $(date -Is)"
