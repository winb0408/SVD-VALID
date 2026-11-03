#!/usr/bin/env bash
# Round10：真实采集上完整自监督训练（A/B 两臂），预注册见
# `reviews/V3_PR1_round10_prereg_real_train.md`（**跑之前写死**）。
#
#   bash scripts/v3_round10_run.sh grid      # 方向 1 网格：2 臂 × 3 域 × {2,6} 长度 × 2 序列 = 24
#   bash scripts/v3_round10_run.sh extend    # 18 epoch 只给 grid 胜者（顺序减半，预注册 §10 偏离 1）
#   bash scripts/v3_round10_run.sh confirm   # 方向 2 确认：selections.json 的配置原样搬 = 4
#   bash scripts/v3_round10_run.sh sn2n      # SN2N 忠实移植，同一半卷 200 epoch = 4
#   bash scripts/v3_round10_run.sh deploy    # 全 400 帧部署（只展示，不参与判定）= 4
#
# 预算（预注册 §9）：48 作业 / 6 GPU·h。GPU 0/1/2 属本项目，GPU3 属他人，不碰。
set -u
cd "$(dirname "$0")/.."
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
OUT=runs/v3_round10
B1=data/processed/v3_round6_b1
B1F=data/processed/v3_round10_b1
mkdir -p "$OUT"

PHASE=${1:?usage: $0 grid|confirm|sn2n|deploy}
EPOCHS_GRID=${EPOCHS_GRID:-"2 6 18"}
# 2026-11-03：GPU0/1/2 名义上属本项目，但当天三张卡被他人重作业占满（GPU2 只剩 <1 GiB），
# 因此默认只用 0/1、并发降到 4。这个模型很小（features 16、patch 32×64×64），单作业约 1–2 GiB。
GPUS=${GPUS:-"0 1"}
MAXJOBS=${MAXJOBS:-4}
DOMAINS=${DOMAINS:-"linear log1p logeps"}

# 与 Round9 `scripts/v3_round9_run_phase1b.sh` 的 COMMON **逐字相同**，只有训练数据
# （真实半卷）与 epochs/steps 不同：steps 取 300 与 Round6 的真实数据作业对齐。
# 禁止改这里的任何数字去"追结果"（预注册 §2）。
COMMON=(--patch 32 64 64 --steps-per-epoch 300
        --lr 1e-4 --lr-schedule poly --clip-grad 20.0
        --weight-reg 1e-4 --weight-identity 1.0
        --features 16 --groups 4 --seed 3407 --arch-head direct --pairing within
        --device cuda)

arm_args() {  # $1 = A|B
  case "$1" in
    A) echo "--mode random --role-perm global" ;;
    B) echo "--mode fixed --pattern 5 --role-order 0 1 3 2" ;;
    *) echo "未知臂 $1" >&2; return 1 ;;
  esac
}

run_jobs() {  # 每行：完整 train.py 参数串 + 目录名，最多 6 并发
  local gpu_i=0 name
  while IFS=$'\t' read -r name argv; do
    [ -z "$name" ] && continue
    if [ -f "$OUT/$name/DONE" ]; then echo "  skip（已完成）$name"; continue; fi
    local n; n=$(echo "$GPUS" | wc -w)
    local gpu; gpu=$(echo "$GPUS" | cut -d' ' -f$(( gpu_i % n + 1 ))); gpu_i=$((gpu_i + 1))
    (
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$gpu $PY -u train.py $argv --out "$OUT/$name" \
        --arm-label "$name" > "$OUT/$name.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && touch "$OUT/$name/DONE"
      echo "  [GPU$gpu] rc=$rc $name"
    ) &
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  done
  wait
}

JOBS=$(mktemp)
case "$PHASE" in
  grid)
    for SEQ in SA64 HSA64; do
      for ARM in A B; do
        for DOM in $DOMAINS; do
          for EP in $EPOCHS_GRID; do
            name="${SEQ}_${ARM}_${DOM}_e${EP}_odd2even"
            printf '%s\t%s\n' "$name" "$(arm_args "$ARM") ${COMMON[*]} \
--epochs $EP --train-volumes $B1/$DOM/${SEQ}_odd_x.npy" >> "$JOBS"
          done
        done
      done
    done
    ;;
  extend)
    # 预注册 §10 偏离 1：18 epoch 只给 {2,6} 上胜出的配置跑（顺序减半，两臂同一规则）。
    SEL=reports/v3_round10/selections.json
    [ -f "$SEL" ] || { echo "缺少 $SEL（先跑 --what grid + --what select）"; exit 3; }
    $PY - "$SEL" "${EPOCH_EXTEND:-18}" > "$JOBS" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
ep = int(sys.argv[2])
common = ("--patch 32 64 64 --steps-per-epoch 300 --lr 1e-4 --lr-schedule poly "
          "--clip-grad 20.0 --weight-reg 1e-4 --weight-identity 1.0 --features 16 "
          "--groups 4 --seed 3407 --arch-head direct --pairing within --device cuda")
for seq, v in s.items():
    for arm, b in v["arms"].items():
        mode = ("--mode random --role-perm global" if arm == "A"
                else "--mode fixed --pattern 5 --role-order 0 1 3 2")
        print("%s_%s_%s_e%d_odd2even\t%s %s --epochs %d --train-volumes "
              "data/processed/v3_round6_b1/%s/%s_odd_x.npy"
              % (seq, arm, b["domain"], ep, mode, common, ep, b["domain"], seq))
PYEOF
    ;;
  confirm)
    SEL=reports/v3_round10/selections.json
    [ -f "$SEL" ] || { echo "缺少 $SEL（先跑 --what grid + --what select）"; exit 3; }
    $PY - "$SEL" > "$JOBS" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
common = ("--patch 32 64 64 --steps-per-epoch 300 --lr 1e-4 --lr-schedule poly "
          "--clip-grad 20.0 --weight-reg 1e-4 --weight-identity 1.0 --features 16 "
          "--groups 4 --seed 3407 --arch-head direct --pairing within --device cuda")
for seq, v in s.items():
    for arm, b in v["arms"].items():
        mode = ("--mode random --role-perm global" if arm == "A"
                else "--mode fixed --pattern 5 --role-order 0 1 3 2")
        print("%s_%s_%s_e%d_even2odd\t%s %s --epochs %d --train-volumes "
              "data/processed/v3_round6_b1/%s/%s_even_x.npy"
              % (seq, arm, b["domain"], b["epochs"], mode, common, b["epochs"],
                 b["domain"], seq))
PYEOF
    ;;
  sn2n)
    for SEQ in SA64 HSA64; do
      for TH in odd even; do
        name="sn2n_${SEQ}_${TH}"
        [ -f "$OUT/$name/DONE" ] && { echo "  skip（已完成）$name"; continue; }
        (
          CUDA_VISIBLE_DEVICES=$(echo "$GPUS" | cut -d' ' -f1) $PY -u scripts/v3_round7_sn2n.py --seq "$SEQ" --epochs 200 \
            --pd-npy "data/processed/v3_round6/$SEQ/$TH/pd_linear.npy" \
            --pred-npy "$OUT/$name/pd_sn2n_linear.npy" \
            --out "$OUT/$name" > "$OUT/$name.log" 2>&1
          rc=$?
          [ $rc -eq 0 ] && touch "$OUT/$name/DONE"
          echo "  [GPU $(echo "$GPUS" | cut -d' ' -f1)] rc=$rc $name"
        ) &
        while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
      done
    done
    wait
    ;;
  deploy)
    SEL=reports/v3_round10/selections.json
    [ -f "$SEL" ] || { echo "缺少 $SEL"; exit 3; }
    $PY - "$SEL" > "$JOBS" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
common = ("--patch 32 64 64 --steps-per-epoch 300 --lr 1e-4 --lr-schedule poly "
          "--clip-grad 20.0 --weight-reg 1e-4 --weight-identity 1.0 --features 16 "
          "--groups 4 --seed 3407 --arch-head direct --pairing within --device cuda")
for seq, v in s.items():
    for arm, b in v["arms"].items():
        mode = ("--mode random --role-perm global" if arm == "A"
                else "--mode fixed --pattern 5 --role-order 0 1 3 2")
        print("deploy_%s_%s_full\t%s %s --epochs %d --train-volumes "
              "data/processed/v3_round10_b1/%s/%s_full_x.npy"
              % (seq, arm, mode, common, b["epochs"], b["domain"], seq))
PYEOF
    ;;
  *) echo "用法: $0 grid|extend|confirm|sn2n|deploy"; exit 2 ;;
esac

n=$(grep -c . "$JOBS" || true)
echo "[$(date '+%F %T')] Round10 阶段 $PHASE：$n 个作业"
run_jobs < "$JOBS"
rm -f "$JOBS"
echo "[$(date '+%F %T')] Round10 阶段 $PHASE 结束"
