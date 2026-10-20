# Gate A — 代码路径核对：Phase 2 的数字到底是从哪条代码路径出来的

> 对 `Chatgpt/SVD_VALID_Phase2_Audit_and_DeepSeek_Execution_Guide.md` §7 Gate A 的交付。
> 目的不是重述方法，而是让读者能**逐项对照代码与报告**，并独立判断
> "匹配空间算子"这个疑点是否成立、frame 0 到底扮演什么角色、
> 每一个分母/掩膜/尺度/ROI 是什么。
>
> 日期：2026-10-19 ｜ 本文件只核对，不引入新结论；结论见 `phase2_audit_verdict.md`。

---

## 0. 环境与版本

| 项 | 值 |
|---|---|
| HEAD SHA | `3309af42be337ac4a40ad8ce0e48e6582e02cb88` |
| Python | 3.10.19（conda env `threePM`，`/data2/wjb/anaconda3/envs/threePM/bin/python`） |
| numpy / torch / scipy | 2.2.6 / 2.6.0+cu124 / 1.15.3 |
| skimage / matplotlib / pytest | 0.25.2 / 3.10.8 / 9.1.1 |
| 硬件 | 64 cores，1006 GB RAM；无 MATLAB/Octave |

源文件 SHA-256（前 16 位），本轮审计结束时的状态：

| 文件 | SHA-256 前 16 |
|---|---|
| `scripts/build_crossfit_pairs.py` | `b4c4083dfe880630` |
| `scripts/make_crossfit_splits.py` | `d7106bd26aa8002f` |
| `scripts/evaluate_pair_statistics.py` | `c203df4915c1fc44` ← 本轮唯一改动，见 §7 |
| `datasets/sampling.py` | `2edc5f282da16761` |
| `datasets/iq_io.py` | `b1b558a047e25814` |
| `scripts/phase3_stats.py` | `1e60b58b083bdd86` |
| `scripts/audit_pair_statistics.py` | `b83db8e4cbbc953c` ← 本轮新增 |
| `scripts/audit_frame0_sensitivity.py` | 见 `phase2_audit_rerun_2026-10-19/` |
| `scripts/audit_frame0_exclusion.py` | 同上 |
| `tests/test_phase2_statistics.py` | `4ab99d3046ef6c69` ← 本轮新增 |
| `reports/phase2_pair_statistics.md` | `e99454acce004544` |
| `data/processed/crossfit/pair_statistics.json` | `98a6bce2472876e4`（gitignored） |

**数据规模。** 两个序列的原始 IQ 都是 `(37, 64, 64, 400)` complex。
有效体素 124 328 / 151 552，即 **17.96% 的死体素**（原始 IQ 恒为 0），
死体素掩膜来自 `data/processed/official/<seq>/valid_mask.npy`。
残差统计的网格是 z 裁剪后的 `(36, 64, 64)`（见 §3.4）。

---

## 1. 帧分组：C / A / B 是怎么切的

`scripts/make_crossfit_splits.py` 产出 `data/processed/crossfit/splits/*.json`。
硬约束写在 `assert_fold()` 里，并在 `build_crossfit_pairs.py` 的**使用点**重新断言一次
（不是只在校验时断言）：

```python
assert not set(fold["calibration"]) & set(fold["A"])
assert not set(fold["calibration"]) & set(fold["B"])
assert not set(fold["A"]) & set(fold["B"])
assert len(a_idx) == len(b_idx) > 0
```

最后一条不是洁癖：臂是**帧的均值**，`T_A != T_B` 会给 D_A 与 D_B 之间注入
`T_B/T_A` 的尺度差，那是定义产生的偏差，不是数据产生的。

四种分组配置，共 19 折：

| split | 策略 | 折数 | T_C | T_A = T_B | 被丢弃的帧 |
|---|---|---|---|---|---|
| `sa64_int3` | 交错 3 组 | 3 | 134 | 133 | 每折 1（大组尾部） |
| `sa64_int5` | 交错 5 组 | 10 | 240 | 80 | 每折 1 |
| `sa64_cont3` | 连续 3 组 | 3 | 133 | 133 | 每折 1 |
| `hsa64_int3` | 交错 3 组 | 3 | 134 | 133 | 每折 1 |

（`sa64_int5` 的 C 是 3 个组的并集，所以 T_C = 240；A、B 各 1 组。）

---

## 2. 四种配对的完整数据流

所有配对共用同一条 `projected_power_map()`，只有"哪些帧进入投影源、哪些帧进入目标"不同。
这一点是审计的核心，所以先把算子写清楚。

### 2.1 投影算子（Gram 形式，从不物化 V×V）

`scripts/evaluate_pair_statistics.py:239`

```
P = U U^H,  U = X_src B,  B = W_R Λ_R^{-1/2}   （W_R, Λ_R 来自 G_src = X_src^H X_src）

P X_tgt = U (U^H X_tgt) = (X_src B) (B^H G[src, tgt])
```

其中 `G = X^H X` 是 400×400 的全序列 Gram，**只算一次**并缓存为
`data/processed/crossfit/<seq>/gram_all.npy`。整个投影因此是两次矩阵乘，
且与 Phase 1 构造 D_A / D_B 用的是**同一段代码**——这不是"看起来像"的替代品。

`retained_basis()` 用 `U_R = X V_R S_R^{-1}` 而不是 `X V_R S_R^{-2} V_R^H X^H`：
后者把条件数平方，在 HSA64 上把幂等性检查推到 1.5e-6 的相对误差——小，但那是
"在平坦尾部悄悄丢精度"的指纹，而秩规则恰好就工作在平坦尾部。

### 2.2 P0 — 完全不投影

`raw_power(x) = mean(|x|^2, axis=1)`。臂是 A、B 两组帧各自原始 IQ 的功率均值。
**§5.4 明示 P0 不是噪声门槛的对照**，只作 sanity check（见 §6）。

### 2.3 P1 — 交叉拟合（提案本身）

```
src = C（该折的校准帧）
tgt = A  → D_A ；tgt = B → D_B
```

同一组 `(B, src=C)` 作用在两条臂上。**这是"交叉拟合"三个字的全部内容**：
投影算子只见过 C，A 与 B 都只是它的目标。

### 2.4 P2 — 各用自己的基（泄漏对照）

```
tgt = A, src = A → D_A^{P2} ；tgt = B, src = B → D_B^{P2}
```

没有 C 参与。它回答的是"P1 的吻合有多少只是'两边被同样地滤了一遍'"。

### 2.5 TETRIS — VALID 自己的配对

`tetris_pair()`（`evaluate_pair_statistics.py:265`）把一张**功率图**按 2×2×2 相位
切成两组互补体素：

```python
PATTERN_A = (0, 1, 2, 4)      # TETRIS_PATTERNS[0]
PATTERN_B = (3, 5, 6, 7)      # TETRIS_PATTERNS[7]
```

对每个 2×2×2 块，一条臂 = 属于该相位的 4 个体素的功率**均值**，然后**回填到整块**
（`np.repeat` 三次）。没有有效体素的块置 0 并掩掉。

三种设置，默认值是中间那个：

| 设置 | 投影源 | 帧 | 频带 | 本报告是否采用 |
|---|---|---|---|---|
| `--tetris-band official` | 全 400 帧的基 | — | 官方 [59,350) | 否 |
| **`--tetris-band rule`（默认）** | **C 的基** | — | **同该折该规则** | **是** |
| `--tetris-frames all` | 同上 | 全 400 帧 | 同上 | 否（敏感性检查） |
| **`--tetris-frames pair`（默认）** | **同上** | **A ∪ B** | **同上** | **是** |

默认组合的含义：Tetris 对照与 P1 用**同一个投影算子、同一条频带、同一批帧**，
唯一差别是"在帧上分组"还是"在体素相位上分组"。这正是 §5.4 的字面要求。

---

## 3. 每一个统计量的分母、掩膜、尺度、ROI

### 3.1 血管候选掩膜（等面积 top-5%）

`vessel_mask(power, valid, q=5.0)`：`thr = percentile(power[valid], 95)`，
掩膜 = `valid & (power >= thr)`。**分母是该图自身的有效体素**，所以
17.96% 的死体素不会稀释阈值。取 `>=` 使实际面积略超 5%，每一处使用都报出实际比例。

预注册草案用的是 `≥ −20 dB re 自身 p99.9`；**它在本数据上是退化的**
（SA64 上选中 100% 有效体素），因此按 §5.4 的"训练前一次标定"换成了等面积规则。
证据保留在输出里：`legacy_candidate_fraction`（SA64 = 1.0000，HSA64 = 0.9840）。

### 3.2 覆盖率与它的零分布

`centreline_coverage(arm_mask, ref_skel)` = 参考骨架体素中被臂掩膜
`binary_dilation(iterations=1)` 覆盖的比例。分母 = **参考骨架体素数**（SA64：2042）。

`centreline_coverage_null(...)`：**200 张等面积随机掩膜**，从有效体素里均匀抽，
同样膨胀、同样算覆盖率。输出 mean / p95 / max。
SA64 mean 0.727、p95 0.754；HSA64 mean 0.747、p95 0.780。
**没有这个零分布，5% 面积下的覆盖率数字读不出任何含义。**

### 3.3 骨架对比度（唯一不带跨管线归一化的结构量）

`skeleton_profile(arm_power, ref_skel, valid)`：
`10·log10(median(arm[骨架 ∩ valid]) / median(arm[valid \ 骨架]))`。
**分子分母来自同一张图**，所以不携带跨管线归一化。

这一点是必须的：官方参考对 400 帧**求和** `|·|^2`，一条臂对它自己的 ~133 帧**求均值**
且只占一条频带，两者按构造差 26 dB（`10·log10 400`）。跨这两者做任何电平陈述，
读到的只是约定差。实测 `ref/arm = 5397.56 = 37.32 dB`（含带宽差），
**因此本报告不给出任何"参考 vs 臂"的电平数字**，代码注释里也写明了这一点。

### 3.4 残差相关性：定义与网格

```python
a = a3d - normalised_box(a3d, valid3d, size=9)     # 9x9x9，死体素不拉低均值
b = b3d - normalised_box(b3d, valid3d, size=9)
r = pearson(a[mask], b[mask])
```

`normalised_box` 是 `uniform_filter` 的**归一化**版本（分子分母同时滤波），
所以死体素既不贡献也不拉低局部均值。

ROI 由**参考图**（不是臂）的阈值定义，`ROI_PCTS = (50, 99)`：

| ROI | 定义 | SA64 体素数 |
|---|---|---|
| `background` | `ref < p50` | 61 565 |
| `mid` | `p50 ≤ ref < p99` | 57 474 |
| `vessel` | `ref ≥ p99` | 1 193 |
| `all_valid` | 全部有效体素 | 120 232（裁剪后） |

**网格**：2×2×2 相位切分要求第一轴为偶数，数据是 37 层。四种配对**全部**统一在
z ∈ [0,36) 上比较（36×64×64），最后 1 层（1/37 = 2.7%）只从第 4 项排除，
第 2、3 项仍用全网格。`to_cropped()` 负责这个转换，避免在两个调用点各写一次。

### 3.5 主门槛的估计量与它的不确定度

`block_residual_diffs(a, b, valid, size=4)`：把裁剪网格切成**不重叠 4×4×4 块**
（`BOOT_BLOCK = 4`），每块内部算一次 Pearson r，**块**才是统计单元。
1881 个块，块内有效体素 < 32 的丢弃。

为什么必须是块而不是体素：同一根血管上相邻体素不是独立样本，体素级相关没有诚实的误差棒。

主门槛的差值是**配对**的（同一块、同一位置），检验用
`sign_flip_permutation_test(n_perm=20000)`，CI 用 `block_bootstrap_ci(n_boot=20000)`。

相对降低的定义（`analyse_fold`，`evaluate_pair_statistics.py:726`）：

```python
rel = (mean(block_r[P1]) - mean(block_r[TETRIS])) / abs(mean(block_r[TETRIS]))
passes = rel <= -0.10 and ci_hi < 0
```

**门槛是 ≤ −0.10，不是 ≥ +0.10。** 方向记反会让整个结论反转，所以这里明确写出。

---

## 4. 空间匹配疑点：确认成立

审计 §1.2 / §5 提出"P1 与 Tetris 的空间处理不同，比较不公平"。**这个疑点成立，而且是决定性的。**

代码层面的事实：`tetris_pair()` 对功率图做的是**块内均值 + 回填**，
这既平滑了图像（去掉块内高频），又把两条臂限制在互补的相位集上。
P1 两条臂则保留完整网格、完整高频。两者被送进**同一个** `residual_correlation()`。

`scripts/audit_pair_statistics.py` 把这件事拆成两个可分别测量的部分：

```python
def tetris_operator(power_map, pattern, shape3d, valid3d):
    # 平滑：块内均值 + 回填（与 tetris_pair 逐位相同）
    # 相位切分：只累加 pattern 里的相位
```

`tetris_operator(power, PATTERN_A, ...)` 与 `tetris_pair(power, ...)[0]` **逐位相等**
（`tests/test_phase2_statistics.py::test_complementary_operator_matches_tetris_pair_bitwise`，
用 `np.array_equal`，不是 `allclose`）。所以后续所有"变体"比较都建立在同一个算子上。

分解结果（`reports/phase2_audit_rerun_2026-10-19/variants.json`）：

| 变体 | 含义 | 与 Tetris 的差 |
|---|---|---|
| `v2_same_support` | 只平滑、**不**切相位（两条臂同一支撑） | ≈ `as_impl` |
| `v3_full_block` | 平滑 + 全 8 相位 | ≈ `as_impl` |
| `v1_compl` | 平滑 + 互补相位集 | ≈ `tetris` |

即：**平滑项的贡献是 +0.005…+0.015，相位切分项的贡献是 −0.30…−0.53。**
审计把矛头指向"块平均/回填改变了高频内容"，指向的是小的那一项。
真正制造差异的是**相位切分**——它让两条臂看到**不同的体素**，而不只是不同的频率内容。

这个修正不削弱审计的结论（算子必须匹配），只是把机制归到了正确的位置。
匹配后的重算见 Gate C。

---

## 5. frame 0 的角色：三种，不是两种

`data/processed/crossfit/splits/*.json` 的 `per_frame_roles` 是唯一真源。SA64 frame 0 的分布：

| split | frame 0 在 C | 在 A | 在 B |
|---|---|---|---|
| `sa64_int3` | 1 折 | 1 折 | 1 折 |
| `sa64_int5` | 6 折 | 4 折 | 0 折 |
| `sa64_cont3` | 1 折 | 1 折 | 1 折 |
| 合计（16 折） | **8** | **6** | **2** |

HSA64 3 折：C / A / B 各 1 折，但 HSA64 没有异常帧（frame 0 总 Gram 能量 1.005× 中位数），
所以分组角色对 HSA64 无意义。

**报告措辞的错误（审计 §4.2 指出，成立）。** 原报告把"frame 0 不在 A/B 的 8 折"称作
"干净折"。这个说法错在：这 8 折里 frame 0 全在 **C** 里，而 C 正是投影算子 P_C 的来源。
frame 0 在 C 里**仍然影响两条臂**——它参与估计被施加到 A 和 B 上的那个投影。
所以"不在 A/B"不等于"不参与"，"干净"二字不成立。

正确措辞：**"第 0 帧不在 A/B 的 8 折"**，并且必须同时说明它在 C 里。

**本轮新增的直接测量（审计未要求，但它改变了判断）。**
在 `sa64_int5_c1_2_4a0b3` 这一折（frame 0 在 A）里，逐帧投影功率为：

```
frame 0 的 ||P x_t||^2        2.59e9
A\{0} 的逐帧中位数            8.98e6       → frame 0 = 288 倍中位数
frame 0 占 D_A 分子的份额      76.6%
frame 0 在 400 帧中按投影功率排名第 1
```

**frame 0 不是"轻微偏离"，它在自己的臂里占据主导。** 这解释了 §5.1 里
"frame 0 落在哪一拨就决定那一折的结论"的全部量级，也解释了为什么
"frame 0 在 C 里"的那些折数字反而更温和：它在 C 里时，投影算子会把它当作
要保留的结构的一部分，而不是把它当作一条臂的一个主导样本。

---

## 6. 单元测试：设计与通过记录

`tests/test_phase2_statistics.py`，18 个测试，6 组。运行：

```
/data2/wjb/anaconda3/envs/threePM/bin/python -m pytest tests/ -q
```

当前结果：**26 passed**（18 个本文件 + 8 个既有 smoke 测试），4.8 秒。

| 组 | 测试 | 钉住的属性 |
|---|---|---|
| 1 估计量正确 | `test_identical_input_gives_r_one` | 非退化相同输入必须 r = 1（精确，不是"很高"） |
| | `test_independent_noise_gives_near_zero` | 独立噪声 r ≈ 0 |
| | `test_statistic_is_monotone_in_true_noise_correlation` | 对真实相关**单调**，含负相关 |
| | `test_noise_correlation_recovered_within_tolerance` | 数值大致对得上，不只是有序 |
| 2 估计量敏感 | `test_statistic_retains_dynamic_range_at_realistic_correlation_length` | 在**实测相关长度**（lag-1 自相关 0.44/0.77/0.50）下动态范围 > 1.2 |
| 3 算子可复现 | `test_complementary_operator_matches_tetris_pair_bitwise` | `tetris_operator` 与 `tetris_pair` **逐位相等** |
| | `test_operator_is_blockwise_constant` | 块均值回填后每块内标准差为 0 |
| | `test_operator_reduces_high_frequency_energy` | 算子确实平滑（所以不能不对齐） |
| 4 算子不是免费的 | `test_matching_the_operator_changes_the_statistic` | 匹配算子后完美对（r>0.95）必须显著变差（−0.1 以上） |
| | `test_matched_operator_preserves_ordering_but_compresses_range` | 匹配**保序**但**压缩**动态范围（两半都断言） |
| 5 退化输入 fail-fast | `test_empty_mask_returns_nan_not_a_number` | 空 ROI → NaN，不是 0 |
| | `test_constant_volume_returns_nan_not_zero` | 常数臂 → NaN；返回 0 会伪装成"完全独立" |
| | `test_pearson_of_zero_variance_is_nan` | 同上，底层 |
| | `test_block_diffs_skip_blocks_below_min_voxels` | 体素不足的块被丢弃，不是记作 r = 0 |
| | `test_dead_voxels_do_not_enter_the_statistic` | 改死体素的值不能动结果 |
| 6 bootstrap 配对 | `test_block_units_are_paired_by_location` | 单元数、单元掩膜跨配对种类一致；确定性 |
| | `test_block_size_changes_unit_count_as_expected` | 块大小与单元数的关系 |
| | `test_rms_rel_diff_scales_as_sqrt_2_over_n` | **两臂**差的标度是 σ√(2/N)，不是 σ/√N |

第 2 组和第 6 组是直接针对审计 §3 与 §5.3 写的：
审计说 `1/√N` 用错了（成立，正确是 `√2/√N`），
说合成数据要覆盖真实相关长度（成立，白噪声下的结论不可用）。

**测试过程中被自己抓到的问题。** 第一版合成生成器在 `SHAPE=(12,16,16)` 上让
平滑基结构泄漏进 9³ 去趋势残差，导致 5 个测试失败。诊断结论是**生成器的缺陷，不是被测代码的缺陷**；
换成 `(24,48,48)` + `base_sig=5.0` 后全部通过。这件事本身写进了测试注释，
因为"合成数据没代表性"正是审计 §5.3 警告的那类错误，我自己先犯了一次。

---

## 7. 本轮对代码的唯一改动

`scripts/evaluate_pair_statistics.py:722`，输出标签：

```python
# 改前
"unit": "non-overlapping 2x2x2 spatial blocks",
# 改后
"unit": f"non-overlapping {BOOT_BLOCK}x{BOOT_BLOCK}x{BOOT_BLOCK} spatial blocks",
```

`BOOT_BLOCK = 4`，所以报告里写 4×4×4、JSON 标签里写 2×2×2——两处不一致。
这是纯标签错误，**不影响任何数字**（统计量本身用的是 `size=BOOT_BLOCK`）。
改它的理由是：审计要求"代码和报告逐项对应"，而这种不一致恰好出现在最容易被
读者当作"块大小"依据的地方。改后 SHA 前缀 `c203df4915c1fc44`。

**除此之外，本轮没有修改任何预注册参数。** `configs/crossfit_pairs.yaml` 与
`configs/phase0_baseline.yaml` 的 SHA 与 HEAD `3309af4` 时相同，门槛
（`centreline_coverage_min = 0.85`、`width_median_change_max = 0.15`、
`noise_relative_reduction_max = −0.10`、`ci_must_exclude_zero = true`）一个字没动。

---

## 8. 逐项对照：报告声称 vs 代码事实

| 报告声称 | 代码事实 | 一致？ |
|---|---|---|
| 四种配对用同一个估计量 | 都走 `residual_correlation()` / `block_residual_diffs()` | 是 |
| Tetris 用同一频带、同一帧集 | `--tetris-band rule --tetris-frames pair`，且投影源同为 C 的基 | 是 |
| 门槛要求"相对降低 ≤ −0.10 且 CI 不含 0" | `rel <= -0.10 and hi < 0` | 是 |
| 单位是 4×4×4 不重叠块 | `BOOT_BLOCK = 4`，1881 块 | **改前标签写错，已修** |
| 结构/强度用全网格，残差用 z 裁剪 | `analyse_fold` 里 `shape3d` vs `cshape` 分开 | 是 |
| 不给"参考 vs 臂"的电平数字 | 代码注释显式拒绝，理由 26 dB 约定差 | 是 |
| P0 不是噪声门槛的对照 | `protocol.note` 明写 | 是 |
| "干净 8 折" | 这 8 折 frame 0 在 **C** 里，仍参与 P_C | **否 —— 措辞错误，见 §5** |

---

## 9. 通俗总结

**这一步在做什么。** 在相信任何一个统计数字之前，先把"这个数字到底是哪段代码、
用哪些帧、哪些体素、什么分母算出来的"逐条对上。审计要求的就是这件事，
而它恰好是最容易被跳过的一步。

**核对下来，代码本身没有错。** 四条配对用的是同一段投影代码、同一个相关估计量、
同一套掩膜规则；门槛的方向（要的是"降低"而不是"升高"）也没写反；
死体素、常数区域、空掩膜这些容易悄悄产生假数字的情况，都有测试钉住，
会明确返回 NaN 而不是返回一个看起来合理的数。26 个测试全部通过。

**有两处报告写错了，一处代码标签写错了。** 代码标签错的是"块大小写成 2×2×2、
实际是 4×4×4"，不影响数字，已改。报告错的两处更重要：

第一处是**把"第 0 帧不在两组里"的 8 折叫成"干净折"**。这个说法不成立，
因为第 0 帧虽然不在这两组里，却在第三组里——而第三组正是用来"算规矩"的那组，
它的影响会通过规矩传到两张图上。所以正确的说法只能是"第 0 帧不在这两组里的 8 折"。

第二处是**审计提出的那个疑点确实成立**：我们拿来当"及格线"的对照方法（按体素位置奇偶切），
它顺手把图像变模糊了，而我们的方法没有。两边被送进同一个尺子去量，
量的就不只是"噪点独不独立"，还量进了"谁更模糊"。这个疑点是对的，必须重算。

**但审计把原因说错了，这一点值得单独讲。** 那种切法其实做了两件事：
一是把小块内的像素取平均（图像变模糊），二是让两半看到**不同的像素**。
我们把这两件事分开量：变模糊只造成 0.005 到 0.015 的差别，
"看到不同像素"造成 0.30 到 0.53 的差别——**差了两个数量级**。
所以真正的问题不是"模糊度不同"，而是"两条臂看的根本不是同一批像素"。
这个区别决定了后面怎么修：光把模糊度对齐是不够的。

**还查出一件审计没预料到的事，而且它比预想的大得多。** SA64 的第 0 帧是个坏帧，
这一点之前就知道。但这次直接量了它在自己那条臂里占多大分量：
**它一帧就占了那条臂总功率的 77%，是其余帧中位数的 288 倍，在 400 帧里排第一。**
它不是"轻微偏离"，它在自己所在的那条臂里是**主导项**。
所以之前"它落在哪一拨就决定那一折的结论"这句话，现在有了确切的量级。

**这一步证明了什么、没证明什么。** 证明了：我们的代码路径是干净的，
报告里的数字确实是从它声称的那条路径出来的，唯一的例外是上面那两处措辞。
没证明：数字本身对不对——那是下一步（重算）的事。也**没有**开始训练，按规矩不能开始。
