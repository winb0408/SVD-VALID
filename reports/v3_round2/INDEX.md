# Round2 产物索引：每条结论 → 数字 → 文件 → 复算命令

**用途**：审查者拿到这一轮的材料时，不必读完整报告就能核对任一结论。每一行都能独立复算。
**主报告**：`reports/v3_round2_positive_control.md`（S0）与 `reports/v3_round2_real_data_audit.md`（R0）
**审查回应**：`reviews/V3_PR1_round2_response.md`
**下一轮方案**：`reviews/V3_PR1_round2_next_plan.md`

约定：`PY=/data2/wjb/anaconda3/envs/threePM/bin/python`；所有命令在 `/ssd1/wjb/SVD-VALID` 下执行。
`runs/`、`data/` 不入 git（体积与数据许可原因），但下面每个 JSON 都已入库，
**JSON 里的数字就是报告里的数字**，不需要重跑训练即可核对。

---

## 1. R0：真实数据审计（Dryad HSA64 / SA64）

| 结论 | 数字 | 文件 | 复算 |
|---|---|---|---|
| 真实数据可用、帧数/几何一致 | HSA64 / SA64 视场与 PSF 见报告 §2 | `reports/v3_round2/HSA64_r0.json`、`SA64_r0.json` | `$PY -u scripts/v3_round2_r0_audit.py --dataset HSA64`（SA64 同理） |
| 真实数据只够支撑一次全长观测，第二次观测来自同一体的独立帧段 | 见报告 §3 | 同上 | 同上 |
| R0 门未通过（不足以启动真实数据训练） | `gate_r0_*.json` 里的判据值 | `reports/v3_round2/gate_r0_HSA64.json`、`gate_r0_SA64.json` | `$PY -u scripts/v3_round2_r0_gate.py --dataset HSA64` |
| R0 两图 | — | `reports/v3_round2/fig_r0_HSA64.png`、`fig_r0_SA64.png` | `$PY -u scripts/v3_round2_r0_audit.py --dataset HSA64`（默认出图，`--no-figure` 可关） |

## 2. S0：判据本身的标定（S0f，跑在任何训练之前）

| 结论 | 数字 | 文件 | 复算 |
|---|---|---|---|
| 判据没坏：已知答案上 10/11 判对 | 见报告 §2.1 表 | `reports/v3_round2/s0_calibration_low.json`、`s0_calibration_mid.json` | `$PY -u scripts/v3_round2_s0_calibration.py --level low`（mid 同理） |
| `zero` 对照被严厉惩罚（不是"什么都没做"能过） | 848.6 / 902.5 | `s0_calibration_low.json` | 同上 |
| 上界与参照全部无训练可复算 | 恒等 1.000 / `oracle_truth` 0.501/0.503 / `oracle_freq_wiener` **0.512/0.515** / `wiener_w7` 3.350/3.852 | `s0_low_metrics.json` 的 `controls` | `$PY -u scripts/v3_round2_s0_eval.py --level low --arms A_rand --out /tmp/x.json` |
| 噪声档与真实数据在同一表示里对齐 | 真实 SA64 0.0272 / HSA64 0.0399；S0 low **0.0198** / mid **0.0397** / high 0.0795（log1p 域 σ_eff） | `reports/v3_round2/s0_level_match.json` | `$PY -u scripts/v3_round2_s0_level_match.py --levels low mid high` |

## 3. S0 主体结果（low 档：自监督两臂未过门）

| 结论 | 数字（e900 / e901） | 文件 | 复算 |
|---|---|---|---|
| `A_rand` 未过门 | `n2n` 2.217 / 2.717 | `s0_low_metrics.json` | `$PY -u scripts/v3_round2_s0_eval.py --level low --runs runs/v3_round2_s0/low --arms A_rand A_fixed4 --out ...` |
| `A_fixed4` 未过门 | `n2n` 4.801 / 6.331 | 同上 | 同上 |
| 门判定 | 见 `gate_s0_low.json` | `reports/v3_round2/gate_s0_low.json` | `$PY -u scripts/v3_round2_s0_gate.py --level low` |
| **mid 档（= 真实数据难度）同样未过门** | `A_rand` 0.963/1.132、1.167/1.323（两种子）；`A_fixed4` 1.700/2.153、1.606/2.042 | `s0_mid_metrics.json`、`gate_s0_mid.json` | `bash runs/v3_round2_s0/launch_level.sh mid 40 500` 后 `$PY -u scripts/v3_round2_s0_eval.py --level mid ...` |
| 全部臂的总览 | — | `reports/v3_round2/figures/fig1_n2n_identity_gap.png` | `$PY -u scripts/v3_round2_s0_figures.py --no-map` |

## 4. 归因（报告 §5，六个候选逐条排除）

| 候选 | 判定 | 数字 | 文件 | 复算 |
|---|---|---|---|---|
| ① 自监督配对（I1）本身是坏目标 | **否掉** | 监督臂与自监督臂逐位近似相同（±0.8%，符号随种子翻转） | `s0_low_metrics_supervised.json` | `bash runs/v3_round2_s0/launch_supervised.sh low 40 500` + `--arms S_rand S_fixed4` |
| ② 欠训练 / 过拟合 / 没学到 | **否掉** | 训练几何里比值 0.532–0.536、留出体 0.545–0.548（差 0.013），训练损失降到 0.16 σ² | `s0_train_geometry.json`；`runs/.../history.json` | `$PY -u scripts/v3_round2_s0_train_geometry.py --level low --runs runs/v3_round2_s0/low --arms A_rand S_rand --seeds 3407 3408` |
| ③ 推理端重装配相位能救回来 | **否掉** | 三种推理口径 × 全部格子：16/16 更差 | `s0_phase_infer.json` | `$PY -u scripts/v3_round2_s0_phase_infer.py --level low --runs runs/v3_round2_s0/low` |
| ④ 训练几何 ≠ 部署几何 | **证实（最大一块）** | `n2n` 2.217/2.717 → **1.171/1.559**（−43%/−46%），第二种子 1.233/1.533 | `s0_low_metrics_dense.json` | `bash runs/v3_round2_s0/launch_dense.sh low 0 40 500` + `--arms S_dense D_n2n` |
| ④' 满分辨率下自监督目标 vs 真值目标 | **等价** | `D_n2n` 1.165/1.541 vs `S_dense` 1.171/1.559（0.5% / 1.2%，小于种子间差异） | `s0_low_metrics_dense.json` | 同上 |
| ⑤ 这条链能不能去噪任何东西（已知答案） | **证实：不能** | `vlow`（σ 降到 1/3）`n2n` **4.970/7.570**，`rmse/σ` 3.00/3.84 | `s0_vlow_metrics.json`、`s0_vlow_metrics_dense.json` | `bash runs/v3_round2_s0/launch_vlow.sh 2 40 500` + `$PY -u scripts/v3_round2_s0_eval.py --level vlow --arms S_dense --seeds 3407 --out reports/v3_round2/s0_vlow_metrics.json` |
| ⑤' 误差与噪声的关系 | 常数项 | 4 点回归 `y = 0.283x + 0.0206`，残差 ±0.003，R² = 0.29 | 上述两个 JSON 的 `controls.input` 与 `S_dense_s3407` | `$PY -u scripts/v3_round2_s0_figures.py --no-map`（图 2） |
| ⑥ 数据量不够 | **否掉** | 训练体 12 → 60：`n2n` 1.171/1.559 → **1.137/1.507**（−2.9%/−3.3%），小于种子间差异 1.7–5.3% | `s0_low60_metrics.json` | 见报告 §6 最后一段 |

## 5. 图（§5.7 全部四张）

| 图 | 文件 | 一句话 | 数据来源 |
|---|---|---|---|
| 图 1 | `figures/fig1_n2n_identity_gap.png` | 除 mid 一个格子（0.96）外，所有臂都在恒等线右侧 | 5 个 metrics JSON |
| 图 2 | `figures/fig2_error_floor.png` | 噪声降 3 倍，预测误差几乎不动（斜率 0.283） | `s0_low_metrics_dense.json` + `s0_vlow_metrics.json` |
| 图 3 | `figures/fig3_train_loss.png` | 训练损失 ÷ σ²：low 到 0.16、vlow 停在 1.15（恒等解 = 1.0） | 各 run `history.json` |
| 图 4 | `figures/fig4_error_map.png` | 输出 ≈ 输入（差值 ×20 才可见） | `runs/v3_round2_s0/low/S_dense_s3407/latest.pt` |

生成：`$PY -u scripts/v3_round2_s0_figures.py`（`--no-map` 跳过图 4；`--gpu N` 指定卡）。

## 6. 二次核对：可比性固定

`low60` 与 `low` 的可比性是逐位验证过的，不是"假定相同"：

| 项 | 证据 |
|---|---|
| 评测体 npz 逐字节相同 | SHA256 前缀 `13e44524…`（e900）、`a873ce2e…`（e901） |
| 训练体前 12 卷逐位相同，另加 48 卷 | 种子 200–211 相同，新增 212–259（严格超集） |
| 唯一变量 = 训练体数 | 其余（σ、噪声域、PSF、patch、预算、种子、损失）全部固定 |

## 7. 测试

```bash
$PY -m pytest tests/ -q          # 89 passed in 8.96s（2026-10-28 复核）
```

## 8. 明确未执行（不要在这些点上引用本轮）

| 项 | 状态 |
|---|---|
| S0 high 档训练 | **未运行**（`high` 数据集与 σ_eff 已算出，训练未跑） |
| S0c（审查 §3 里的一项） | **未运行** |
| `vlow` / `low60` 的多余种子 | **未运行**（各只 s3407；效应量远大于种子间差异，但正式结论需补） |
| 最佳候选修复（残差参数化 / 损失形式 / 容量） | **未运行**——见 `reviews/V3_PR1_round2_next_plan.md` |
| 采样先验（I1）的任何实验 | **未运行，且被本轮结论明确挡住**（先修链再谈采样） |
| B / C / D 链 | **冻结**（`runs/v3_chain_after_T2.sh` 不得执行） |
