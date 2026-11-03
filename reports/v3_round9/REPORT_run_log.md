# Round9 实际运行记录（命令、时间、哈希）

本文件只记录**实际跑过什么**，不含解释与结论。结论见 `DECISION.md`、`baseline_shared_crn.md`。
时间均为本机时间（UTC+8）。

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | `/data2/wjb/anaconda3/envs/threePM/bin/python` |
| torch | 2.6.0+cu124 |
| numpy | 2.2.6 |
| GPU | 训练：A→GPU2，B→GPU0（GPU3 属于其他用户，未使用） |

---

## 2. 两条训练臂

启动脚本：`scripts/v3_round9_run_phase1b.sh`（数字全部来自 `reviews/V3_PR1_round9_prereg.md` §4）

```bash
bash scripts/v3_round9_run_phase1b.sh A 2 > runs/_r9A.log 2>&1 &
bash scripts/v3_round9_run_phase1b.sh B 0 > runs/_r9B.log 2>&1 &
```

展开后的等价命令行：

```bash
# 臂 A：VALID-faithful random（每块独立抽 tet，一个 patch 一个全局角色置换）
CUDA_VISIBLE_DEVICES=2 python -u train.py \
  --mode random --role-perm global --pairing within \
  --train-volumes data/processed/v3_round2_s0/low/train/s0_low_s{200..211}_log1p.npy \
  --patch 32 64 64 --epochs 50 --steps-per-epoch 200 \
  --lr 1e-4 --lr-schedule poly --clip-grad 20.0 \
  --weight-reg 1e-4 --weight-identity 1.0 \
  --features 16 --groups 4 --seed 3407 --arch-head direct \
  --device cuda --out runs/v3_round9/A

# 臂 B：最强可行固定 tet（校准选出 pattern 5, role_order 0 1 3 2）
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --mode fixed --pattern 5 --role-order 0 1 3 2 --pairing within \
  --train-volumes ...（同上 12 个）--patch 32 64 64 --epochs 50 --steps-per-epoch 200 \
  --lr 1e-4 --lr-schedule poly --clip-grad 20.0 \
  --weight-reg 1e-4 --weight-identity 1.0 \
  --features 16 --groups 4 --seed 3407 --arch-head direct \
  --device cuda --out runs/v3_round9/B
```

（`s{200..211}` 只是本文写法，实际脚本里是 12 个完整路径，见 `scripts/v3_round9_run_phase1b.sh`。）

| 臂 | 启动 | 墙钟 | 资源 | checkpoint | SHA-256 |
|---|---|---|---|---|---|
| A | 18:48:22 | **17:54.52** | 峰值 RSS 1.24 GB | `runs/v3_round9/A/latest.pt`（epoch 50，6.0 MB） | `5ad52986e13cf95adf3fafef218b91a7140b8ad7aea8c542429984597c32fce6` |
| B | 18:48:24 | **15:40.05** | 峰值 RSS 1.24 GB | `runs/v3_round9/B/latest.pt`（epoch 50，6.0 MB） | `51df1d0f1a59488f8bfc15ee3bedc19d3c330ea6718c10beb22718ba0cdc91f8` |

- 步数：各 **10,000** 步（50 epoch × 200 步），与预注册一致。
- **GPU·h：0.298 + 0.261 = 0.56**（训练）；推理 94 s ≈ 0.026 → **本轮合计 ≈ 0.59 GPU·h**
  （Phase 1 上限 3.0，Round9 上限 6.0）。
- checkpoint 内 `args` 已核对：A = `mode=random, role_perm=global`；
  B = `mode=fixed, pattern=5, role_order=[0,1,3,2]`。其余 16 个超参与两臂逐位相同。
- **中途没有按评价指标做任何选择**：2,000 步（epoch 10）时只看过 loss 是否数值正常
  （A 0.0117 / B 0.0132，无 NaN），未跑任何评价、未换 seed、未改配置。

### 损失轨迹（每 epoch 末的 training loss）

```
A: 0.378 0.0199 0.0144 0.0146 0.0130 0.0118 0.0124 0.0116 … 0.0087(min) … 0.009126(末)
B: 0.370 0.0193 0.0171 0.0157 0.0144 0.0134 0.0133 0.0119 … 0.0093(min) … 0.009414(末)
```
末值与最小值差 < 5%，poly 学习率已衰减到 0，视为按计划终止；**没有**以"还没收敛"为由续训。

---

## 3. 推理与评价

`infer.py` 满分辨率滑窗（`--patch 32 64 64 --overlap 0.1`），无相位抽稀、无回填：

```bash
python -u scripts/v3_round9_eval.py --runs runs/v3_round9/A runs/v3_round9/B \
    --split primary --gpu 2      # e902–e905，36.3 s
python -u scripts/v3_round9_eval.py --runs runs/v3_round9/A runs/v3_round9/B \
    --split strict  --gpu 2      # e906–e911，57.7 s（主考未过，仅作确认）
python -u scripts/v3_round9_report.py --split primary      # 出图
```

产出：`metrics_primary.json`、`metrics_strict.json`、`matched_strength.json`、
`figs/s0_e90{2..5}_compare.png`、`figures.md`。

---

## 4. 为判定服务的诊断（**不涉及新训练**）

1. **训练目标 vs 恒等解**（`f=identity` 在同一采样几何、同一 12 卷上的目标值）：
   `neighbor 0.00869 / identity 0.00832 / reg 1.211`，`total 0.01714`；
   A 臂训练结束时的模型：`neighbor 0.00784 / identity 0.00236 / reg 1.020`。
   → **训练是有效的**（目标从 0.0171 降到 0.0091），不是"没训起来"。
2. **匹配强度对比**（`matched_strength.json`）：把固定高斯调到与网络输出**相同的 roi_mean
   缩减程度**再比：A 34.55 dB vs 高斯 35.08 dB；分支误差 A 1.110 vs 高斯 1.029。
   → 不是"只是更平滑所以吃亏"，**同强度下仍然更差**。

---

## 5. 复算入口

```bash
python tests/test_round9_phase1a.py                 # 8 条实现正确性单测
python scripts/v3_round9_upstream_audit.py          # 上游逐项核对 → upstream_audit.json
python scripts/v3_round9_calibrate_baseline_arm_b.py  # 仅用 e900/e901 → calibration.json
```
