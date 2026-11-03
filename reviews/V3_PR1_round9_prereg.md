# Round9 预注册（**在任何训练开始之前写定，不得事后修改**）

日期：2026-11-03
上位计划：`reviews/V3_PR1_round9_shared_valid_restoration_plan_2026-11-03.md`
保真核对：`reports/v3_round9/implementation_fidelity.md`
本文件作用：把 split / seed / steps / 真值 / 基线参数 / GPU 上限 / 判定门槛**全部写死**，
使 Phase 1B 的结果无法用"再换一个设置试试"来救。任何与本文不符的运行一律作废。

---

## 0. 先分清两件事（计划明确要求）

| 状态 | 内容 |
|---|---|
| **已修代码**（本轮已提交，可复算） | `train.py` 新增 `--role-perm {per_block,global}` 并透传到 `valid_random_tetris` / `_strided`，非法组合直接报错；`datasets/sampling.py` 纠正 `random_tetris_subvolumes` 与 `cross` 的措辞；`--lr-schedule` 说明改为"上游实际生效的是 poly（741 行每 batch 覆盖 686 行的 LambdaLR）"；新增 `tests/test_round9_phase1a.py`（8 条全过）、`scripts/v3_round9_upstream_audit.py`（产出 `upstream_audit.json`）、`scripts/v3_round9_calibrate_baseline_arm_b.py`（产出 `calibration.json`） |
| **已运行实验** | **无。** 截至本文件提交，本仓库**没有跑过任何 Round9 训练**。不存在任何 Round9 的 checkpoint、指标或图像。 |

---

## 1. 数据与划分（写死）

| 用途 | 内容 |
|---|---|
| 训练 | `data/processed/v3_round2_s0/low/train/s0_low_s{200..211}_log1p.npy`（**12 个场景**，每场景一条 `pd_linear`，与上游"同一张噪声图取两视图"一致） |
| 标定（选基线参数） | `low/eval` 的 **e900、e901**（只用于：arm B 几何选择 + 高斯 sigma，见 §3） |
| **Primary held-out** | `low6/eval` 的 **e902、e903、e904、e905**（4 个独立场景；历史上被 Round2–6 用过，因此**不算严格盲测**） |
| **Strict 确认集** | `low_hold/eval` 的 **e906…e911**（6 个场景，**从未用于任何选参或选方案**） |
| 绝不使用 | `low/train` 的任何 seed 作评价；`vlow*` / `mid` / `high` 作主判定（只在专门标注的敏感性分析里出现） |

场景独立性：同一场景的不同 patch / 不同帧**不算**独立样本，gate 只按**场景数**计。

---

## 2. 真值与评价

- **主终点真值**：`gt_imaging_pd`（血流 + 静态本底的**无噪完整成像期望**）。
  **不用** `gt_flow_pd`（血流分量是辅助参考，计划明写它不是 N2N 的正确目标）。
- 主指标在**线性 PD 域**算：`pred_pd = s·(exp(pred_log1p) − 1)`，`truth_pd = gt_imaging_pd`；
  每场景 `PSNR = 10·log10(peak²/MSE)`，`peak = max(gt_imaging_pd[valid])`（真值定，与任何方法无关，预防事后调标尺）。
- 同时报告（不替代主指标）：log1p 域 `MSE / r_roi / rmse`、线性域 `bg_mean`、`roi_mean`。
- **背景**：`valid & (rho <= 0)` 的 `mean`、`std`、`std/ref_bg_std`（`ref_bg_std` 由输入的背景 std 定）。
- **弱血管 / 分支**（写死定义，先于结果）：
  - `skel = skimage.morphology.skeletonize_3d(vessels)`
  - `weak = skel & (rho <= quantile(rho[skel], 0.33))`
  - 指标：`branch_rmse = RMSE(pred_pd, truth_pd)` on `skel`；`weak_rmse` on `weak`；
    以及 `weak_bias = mean(pred_pd − truth_pd)` on `weak`（过平滑会给出偏负）
- **显示**：所有对比图共用同一 dB 参考与同一 ROI（不做逐图拉伸）。

---

## 3. 基线（参数在**校准场景**上选出，本轮不再变化）

来自 `scripts/v3_round9_calibrate_baseline_arm_b.py` → `reports/v3_round9/calibration.json`：

| 基线 | 参数 | 校准依据 |
|---|---|---|
| 恒等 | 无 | — |
| 固定高斯 | `sigma = 0.60`（log1p 域，`mode="nearest"`） | e900/e901 上 log1p-MSE 最小（8.667e-05；σ0.5→1.220e-04，σ1.0→2.556e-04） |
| 维纳 | `scipy.signal.wiener(mysize=3)`（3D 逐帧 2D？）—— 本轮统一用**3D 立方窗** `mysize=3` | 无可调参数 |
| 参照上界（oracle，不可实现） | `_freq_wiener()`（用真值与第二次观测） | 只用来确认"任务有收益空间" |

> 说明：高斯 σ=0.6 在 log1p 域的 MSE 已经比"无噪信号跨角点差异"（arm B 的 J≈0.0042）小约 48×，
> 这意味着**这个正对照的噪声非常适合线性平滑**。这一点在写结果时必须如实写出来。

---

## 4. 两条臂（**只有这两条**；一条 seed；不许多跑）

共同设置（两臂**逐位相同**，只差采样几何）：

```
--patch 32 64 64
--epochs 50 --steps-per-epoch 200          # = 10,000 optimizer steps
--lr 1e-4 --lr-schedule poly               # 上游实际生效的 schedule
--clip-grad 20.0                           # 上游默认（我们历史默认 100，本轮显式改）
--weight-reg 1e-4 --weight-identity 1.0
--features 16 --groups 4 --arch-head direct
--seed 3407
--pairing within                           # 禁止 cross（见 §6）
--train-volumes <上面 12 个 log1p.npy>
```

| 臂 | 采样 | 命令关键差异 | 依据 |
|---|---|---|---|
| **A：VALID-faithful random** | 每块独立抽 tet，**一个 patch 一个全局角色置换** | `--mode random --role-perm global` | 上游实测行为（`upstream_audit.json`: 5 次抽样 rank 元组均唯一） |
| **B：最强可行固定 tet** | 8 个 pattern × 24 个角色置换全枚举，取校准场景上 `J = 0.5·MSE(truth_vk,truth_vi)+0.5·MSE(truth_vl,truth_vj)` 最小 | `--mode fixed --pattern 5 --role-order 0 1 3 2` | 校准结果 `J=0.004155`（最好）vs `0.004535`（最差），**全枚举跨度仅 1.09×** |

**B 臂选择的诚实说明**：J 的最好/最差只差 1.09×，说明"选哪一套固定几何"在这个正对照上几乎是
退化的。所以 B 臂**不是**一个强对手，它的作用是给出"固定几何能达到什么"的量级，
而不是证明"物理引导有空间"。这一点不得在结果文档里被淡化。

**中间检查点规则**：只允许在两个臂各 2,000 步时看一眼，且**只**用于判断机械故障
（NaN、loss 不降、全零输出、显存/时间异常）。**不允许**根据 2,000 步的**评价指标**决定
"要不要继续 / 换 seed / 换配置"。10,000 步是写死的终点。

---

## 5. 预算与停机

| 项 | 上限 |
|---|---|
| 本轮新增训练作业 | **2**（A、B） |
| Phase 1 GPU·h | **≤ 3.0**（两臂合计，含推理） |
| Round9 累计 GPU·h | **≤ 6.0** |
| 推理 | `infer.py`，`--patch 32 64 64 --overlap 0.1`，满分辨率滑窗平均（**不许** 8 相位回填、不许子视图拼接/插值） |
| GPU | 只用 0/1/2；**GPU3 属于其他用户** |

达到任一上限 → **立即停下并提交决定**，不得自行续跑。

---

## 6. 明确禁止（与计划一致）

- ❌ Phase 0 的水波纹 / DC / 频谱诊断（用户已取消）。
- ❌ `--pairing cross`（本仓库自创，非上游）。
- ❌ 8 相位回填、子视图推理拼接/插值后处理。
- ❌ 换网络、加大网络、加可训练参数、加 seed、重切测试集来"救"结果。
- ❌ 用 `gt_flow_pd` 当主终点。
- ❌ 任何形式的逐图自动拉伸对比图。

---

## 7. Phase 1 Gate（照抄计划，不改门槛）

在 **Primary held-out（e902–e905）** 上判定：

- **PASS（仅 exploratory）** 需同时满足：
  1. 臂 A 在 **≥4 个独立场景中的多数（≥3）** 上，**干净成像主误差**（线性 PD 域 RMSE/PSNR）优于恒等；
  2. **没有明显的弱血管 / 分支损伤**（`weak_rmse` / `branch_rmse` 不得同时劣于恒等在 ≥3 个场景上）；
  3. 在某个**有意义的噪声–结构权衡区间**内，**没有被固定高斯 / 维纳完全支配**（即存在某处
     A 的结构保真优于它们，或匹配强度后噪声指标不劣于它们）；
  4. 臂 B 的变化能用同一协议解释。
  → 结论只能写 `PROMISING_REQUEST_APPROVAL` 级别的"exploratory pass"，**不是**"方法有效"。
- **STOP_BASELINE**：无结构安全增益，或两臂都被平凡滤波支配。→ 立即中止 VALID/CRN 迁移路线，
  **不**启动物理引导采样，不加网络/损失/seed；报告不得用"差一点就过了"要求重跑。
- **BLOCKED**：采样正确性、评价真值、数据分组任一不可审计。→ 只报阻塞，不烧预算。
- 若 Primary 通过，再用 **Strict 确认集 e906–e911** 复核；Strict 上不成立则降级为
  `STOP_BASELINE`（Primary 与 Strict 冲突时以 Strict 为准并如实写明）。

判定脚本：`scripts/v3_round9_eval.py`（**先有本文件，后有该脚本**；脚本不得改门槛）。

---

## 8. 产出清单（缺一即写 NOT_RUN）

1. `reports/v3_round9/baseline_shared_crn.md` + `metrics.json` + 同标尺图 + checkpoint 哈希 + 实际耗时。
2. `reports/v3_round9/REPORT_run_log.md`：两臂的**实际**命令行、起止时间、GPU·h。
3. `reports/v3_round9/DECISION.md`：`STOP_BASELINE` / `STOP_I1_NO_DISTINCT_INTERVENTION` /
   `STOP_I1` / `PROMISING_REQUEST_APPROVAL` / `BLOCKED` 之一，必须引本文件的 §7。
4. `reports/v3_round9/physical_sampler_intervention.md`：若 Phase 1 未 PASS，正文写 `NOT_RUN` + 原因。
5. `Project log/Project_progress_status.md` 增加 Round9 段落，**保留 Round8 全部记录**。

---

## 9. 通俗总结

这份文件的作用是"先把话说死，再去做实验"。我们规定了：用哪 12 个体积训练、用哪 2 个场景挑
滤波器的参数、用哪 4 个场景当主考场、另外 6 个场景留着做没被污染过的复核；训练多少步、
学习率怎么走、梯度裁剪多少、两个对照方法各自用什么参数、最多花多少机时——全部写在跑之前。
两条训练臂在所有地方都一样，**只差"每个小块怎么取那四个角、哪两个当输入哪两个当目标"**：
A 臂照上游 VALID 的真实随机规则，B 臂用标定场景挑出来的一套固定取法。

有一条必须提前说清、不许事后淡化：B 臂这套"固定取法"的八种可能之间差别只有 9%，
换句话说在这个正对照里"选哪一套固定几何"几乎不重要。所以 B 臂不是一个强对手，
它只能回答"固定几何大概能到什么水平"。

判定也先说死：A 臂必须在 4 个主考场景里至少 3 个上把干净真值的误差降下来，而且不能把细小血管
磨掉，也不能被"高斯模糊"和"维纳滤波"这两个不用训练的方法全面压住。三条全满足才叫"探索性通过"
（注意：只是通过、不是成功）；不满足就写 STOP，并且**不许**用"再换个设置就过了"来要求重跑。
