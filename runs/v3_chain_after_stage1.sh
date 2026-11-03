#!/bin/bash
# 阶段 1 训练完成后的自动链：评价 → 立刻起阶段 2 训练 → 阶段 2 评价草稿
# 失败不阻断后续（每一步的日志单独落盘，便于事后逐段核对）。
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
D=runs/v3_expA_200k

echo "[chain] $(date '+%F %T') 等阶段 1 收尾（launch.log 里出现 ALL_DONE）"
until grep -q ALL_DONE $D/launch.log 2>/dev/null; do sleep 30; done
echo "[chain] $(date '+%F %T') 阶段 1 训练结束，开始评价"
$PY -u scripts/v3_expA_faithful.py --eval-only --runs $D \
    --seeds 3407 3408 --arms A_rand A_fixed4 \
    --gpus 0 --infer-gpu 0 --out reports/v3_expA_v2/stage1_eval \
    > $D/eval.log 2>&1
echo "[chain] $(date '+%F %T') 阶段 1 评价返回码 $?"

echo "[chain] $(date '+%F %T') 起阶段 2 训练（v2 仿真 16 卷）"
mkdir -p runs/v3_expA_v2_200k
bash runs/v3_expA_v2_200k/launch.sh > runs/v3_expA_v2_200k/launch.log 2>&1
echo "[chain] $(date '+%F %T') 阶段 2 训练返回码 $?"

$PY -u scripts/v3_expA_v2_eval.py --runs runs/v3_expA_v2_200k --gpu 0 \
    > runs/v3_expA_v2_200k/eval.log 2>&1
echo "[chain] $(date '+%F %T') 阶段 2 评价返回码 $?  —— CHAIN_DONE"
