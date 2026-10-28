#!/usr/bin/env bash
# 等 B1 四臂训练跑完 → 逐臂跑探针（A2/A4 + r_roi 对照）→ 汇总表。
# 判读规则已写在 `reports/v3_round3/b_minimal_repair.md` §2 与 plan §4.2/§4.4，
# 本脚本只做数值收集，不含任何阈值。
set -u
cd /ssd1/wjb/SVD-VALID
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs/v3_round3/b1_collect.log
exec >>"$LOG" 2>&1

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }
wait_for() {   # $1 = history.json 路径
  while [ ! -f "$1" ]; do sleep 60; done
  ts; echo "ready: $1"
}

V=runs/v3_round3/b1_vlow
L=runs/v3_round3/b1_low
for f in $V/B1_residual_s3407/history.json $V/B1_zeroinit_s3407/history.json \
         $L/B1_low_residual_s3407/history.json $L/B1_low_zeroinit_s3407/history.json; do
  wait_for "$f"
done

ts; echo "=== all four arms done; running probes ==="
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/v3_round3_a_probes.py --level vlow --arm B1_residual \
  --seed 3407 --runs $V --gpu 0 --skip-a1 \
  --manifest data/processed/v3_round2_s0/vlow6/manifest.json \
  --out reports/v3_round3/b1_residual_probe.json
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/v3_round3_a_probes.py --level vlow --arm B1_zeroinit \
  --seed 3407 --runs $V --gpu 0 --skip-a1 \
  --manifest data/processed/v3_round2_s0/vlow6/manifest.json \
  --out reports/v3_round3/b1_zeroinit_probe.json
# 旧基线 direct 臂用同一套代码重测，保证 r_roi 与输入的读数口径一致
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/v3_round3_a_probes.py --level vlow --arm S_dense \
  --seed 3407 --runs runs/v3_round2_s0/vlow --gpu 1 --skip-a1 \
  --manifest data/processed/v3_round2_s0/vlow6/manifest.json \
  --out reports/v3_round3/b1_direct_probe.json

CUDA_VISIBLE_DEVICES=1 $PY -u scripts/v3_round3_a_probes.py --level low --arm B1_low_residual \
  --seed 3407 --runs $L --gpu 1 --skip-a1 \
  --manifest data/processed/v3_round2_s0/low6/manifest.json \
  --out reports/v3_round3/b1_residual_low_probe.json
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/v3_round3_a_probes.py --level low --arm B1_low_zeroinit \
  --seed 3407 --runs $L --gpu 1 --skip-a1 \
  --manifest data/processed/v3_round2_s0/low6/manifest.json \
  --out reports/v3_round3/b1_zeroinit_low_probe.json
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/v3_round3_a_probes.py --level low --arm S_dense \
  --seed 3407 --runs runs/v3_round2_s0/low --gpu 1 --skip-a1 \
  --manifest data/processed/v3_round2_s0/low6/manifest.json \
  --out reports/v3_round3/b1_direct_low_probe.json

ts; echo "=== summary: vlow ==="
$PY -u scripts/v3_round3_b1_summary.py
ts; echo "=== summary: low ==="
$PY -u scripts/v3_round3_b1_summary.py --tag _low
ts; echo "=== collect done ==="
