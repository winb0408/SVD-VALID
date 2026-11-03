#!/usr/bin/env bash
# Round6：真实采集上重新自监督训练，采样几何直接对照。
# 预注册见 reviews/V3_PR1_round6_prereg_real_selfsup.md（**跑之前写死**）。
#
# 阶段 1（1）：方向 1 = 在 odd 半训练 → 在 even 半算 X_RMSE。
#   2 几何 × 3 域 × 2 长度 × 2 数据集 = 24 作业。
# 阶段 2（2）：把方向 1 选出的配置搬到 even→odd 重训（无偏确认）。
#   2 几何 × 2 数据集 = 4 作业。
#
# 预算（预注册 §6）：≤28 作业、≤3.5 GPU·h。GPU 0/1/2 为本项目，GPU3 属他人。
set -u
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
OUT=runs/v3_round6
B1=data/processed/v3_round6_b1
mkdir -p "$OUT"

PHASE=${1:-1}
SEL=${2:-reports/v3_round6/selections.json}
case "$PHASE" in 1|2) ;; *) echo "用法: $0 1|2 [selections.json]"; exit 2;; esac

# 训练长度轴。原预注册为 {6,18}；2026-11-02 20:35 收缩为单一 6，**原因见预注册 §8**：
# GPU0/1/2 上有他人重作业，单作业耗时约为独占时的 2.9 倍，{6,18} 实耗≈7.2 GPU·h，
# 超过本轮自定的 3.5 GPU·h 上限。按"超预算立即停、不追加"的规则，砍掉 18-epoch 那一半。
EPOCH_SETS=${EPOCH_SETS:-"6"}

COMMON="--patch 32 64 64 --steps-per-epoch 300 --features 16 --groups 4 \
  --seed 3407 --lr 1e-4 --lr-schedule poly --arch-head residual --zero-init-final"

geom_args() {  # $1 = D1 | BD2
  case "$1" in
    D1)  echo "--stride 1 --mode random --pairing cross" ;;
    BD2) echo "--stride 2 --mode fixed --slots 1 4 6 3 --pairing cross" ;;
    N2N) echo "--stride 1 --mode dense --pairing cross" ;;
    *) echo "未知几何 $1" >&2; return 1 ;;
  esac
}

# 任务表：每行 "SEQ train_half eval_half geom domain epochs"
JOBS=$(mktemp)
if [ "$PHASE" = "1" ]; then
  for SEQ in SA64 HSA64; do
    for geom in D1 BD2; do
      for dom in linear log1p logeps; do
        for ep in $EPOCH_SETS; do
          echo "$SEQ odd even $geom $dom $ep"
        done
      done
    done
  done > "$JOBS"
else
  [ -f "$SEL" ] || { echo "缺少选择结果 $SEL（先跑阶段 1 与 v3_round6_select.py）"; exit 3; }
  $PY - "$SEL" <<'PY' > "$JOBS"
import json, sys
s = json.load(open(sys.argv[1]))
for seq, v in s.items():
    for geom in ("D1", "BD2"):
        b = v[geom]["best"]
        print(f"{seq} even odd {geom} {b['domain']} {b['epochs']}")
PY
  # N2N 正对照也补方向 2：三个方法必须**同处一个方向**，否则"有偏/无偏"混在一起没法比。
  # 属预注册 §8 偏离 4，理由见报告。
  if [ "${INCLUDE_N2N:-1}" = "1" ]; then
    for SEQ in SA64 HSA64; do echo "$SEQ even odd N2N log1p 6"; done >> "$JOBS"
  fi
fi

echo "[$(date '+%F %T')] Round6 阶段 $PHASE：$(wc -l < "$JOBS") 个作业"

gpu_i=0
while read -r SEQ th eh geom dom ep; do
  name="${SEQ}_${geom}_${dom}_e${ep}_${th}2${eh}"
  if [ -f "$OUT/$name/DONE" ]; then
    echo "  skip（已完成）$name"
    continue
  fi
  gpu=$((gpu_i % 3))
  gpu_i=$((gpu_i + 1))
  # N2N 正对照（dense 配对）必须显式给出目标卷 = 另一半，否则输入=目标=同一张图
  extra=""
  [ "$geom" = "N2N" ] && extra="--train-clean-volumes $B1/$dom/${SEQ}_${eh}_x.npy"
  # shellcheck disable=SC2046
  (
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train.py $COMMON $(geom_args "$geom") $extra \
      --epochs "$ep" --train-volumes "$B1/$dom/${SEQ}_${th}_x.npy" \
      --out "$OUT/$name" --arm-label "$name" > "$OUT/$name.log" 2>&1
    rc=$?
    # 只有正常退出才打完成标记：v3_round6_eval.py 只评有 DONE 的目录，
    # 杜绝"拿半截 checkpoint 当结果"。
    [ $rc -eq 0 ] && touch "$OUT/$name/DONE"
    echo "  [GPU$gpu] rc=$rc $name"
  ) &
  # 每 GPU 同时最多 2 个（3 GPU × 2 = 6 并发）
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
done < "$JOBS"
wait
rm -f "$JOBS"
echo "[$(date '+%F %T')] Round6 阶段 $PHASE 结束；run 目录 $(find "$OUT" -maxdepth 1 -type d -name '*_e*' | wc -l) 个"
