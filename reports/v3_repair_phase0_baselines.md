# R01 / R02 整改：上游 VALID v1.0 源码对照与覆盖率口径审计

> **审查来源**：`Chatgpt/V3_PR1_external_review_2026-10-26.md` R01、R02
> **上游版本**：`FDU-donglab/VALID` tag **v1.0**，commit `34f5637a0f1150a980d1a7c8cd7a38c3e39dff6f`
> （`git ls-remote --tags` 实测；GPL-3.0，见 `THIRD_PARTY_LICENSES.md`）
> **克隆位置**：`/tmp/valid_upstream`（**临时目录，未入库**——上游是 GPL-3.0，不往本仓库拷代码）
> **本文件性质**：源码级对照与口径审计。**不含新实验结果**。凡未核验的写在 §4。

---

## 1. 采样契约：逐项对照表

| # | 上游 VALID v1.0 | 本项目 | 关系 | 核验方式 |
|---|---|---|---|---|
| 1 | `datasets/sampling.py::generate_mask_pair` 的 `idx_pair`：8 个 4 角点模式 | `datasets/sampling.py::TETRIS_PATTERNS` | **逐项相同，连顺序都相同** | 脚本比对，`集合相同=True, 顺序相同=True` ✔ |
| 2 | 每个 2×2×2 块随机抽 1 个模式（`torch.randint(0,8)`） | `valid_random_tetris`：同 | 一致 | 读码 ✔ |
| 3 | 角色的随机置换：`RandomSampler_0123` 洗牌 `[0,1,2,3]`，依次给 mask1..mask4 | `valid_random_tetris`：`torch.rand(...).argsort(dim=1)` | 同构（等概率置换），RNG 流不同但分布相同 | 读码 ✔ |
| 4 | **每块只取 4 个角点**（mask1..4 各取一个），块内 8 角点中另 4 个**不用** | 同 | 一致 | 读码 ✔ |
| 5 | 输入 = 2 个角点（输出 1、2），目标 = 同一 tet 的另外 2 个角点（输出 3、4） | `valid_random_tetris`：同 tet | 一致 | 读码 ✔ |
| 6 | 损失：`0.5·L2(out1,tgt1) + 0.5·L2(out2,tgt2) + L2(out1,out2) + w·Hessian(DWT 低频)` | `train.py:211-215`：`0.5·mse(ha,ya)+0.5·mse(hb,yb) + w_reg·hessian + w_idt·mse(ha,hb)` | 结构一致；**`weight_identity` 默认值需逐项核**（上游系数恒为 1） | 部分核验 🟡 |
| 7 | 每个 iteration **重新生成** mask（`train_pipeline.py:744` 在训练循环内） | `train.py:206` 同样在 step 循环内 | 一致 | 读码 ✔ |
| 8 | **推理**：`output = model(input)` 对**整 patch** 前向，再用 `reconstruct_dataset` 按索引拼回 | `infer.py`：滑窗 + 累加平均 | 一致（都是整卷前向） | 读码 ✔ |
| 9 | 上游**没有**相位回填推理 | `scripts/v3_phase3_eval.py::reconstruct` 8 相位各前向一次再散回各自格点 | **本项目独有，不是上游契约** | 读码 ✔ |
| 10 | 上游**没有** cross 配对 | `datasets/sampling.py::valid_random_tetris_cross`（目标取互补 tet） | **V3 自创**，不得称为 faithful | 读码 ✔ |

**结论（R01 部分）**：本项目 **`valid_random_tetris` + `infer.py`** 这条路径在上游契约上是**忠实**的；
**`valid_random_tetris_cross`** 与 **`reconstruct()`** 都不是上游行为。审查第 R01(1) 条
"不得悄悄将 cross 模式覆盖 faithful 模式"——**目前代码没有覆盖**，但报告与图注里确实有混称，
已在 `Project_progress_status.md` §3/§4 写明。

---

## 2. 口径审计：**"m=1 覆盖率 100%" 这个说法是错的**（R02 的根因之一）

这是本次对照最重要的发现，此前所有报告都没写清。

- **忠实 VALID 的训练采样，每个 2×2×2 块只用 4 个角点**（上方第 4 行）。所以 faithful 的
  训练覆盖率是 **4/8 = 50%**，不是 100%。
- 本项目 Phase 2/3 报告里的 "m=1 → 100%" 指的是**推理时 8 个相位全部扫一遍**（`reconstruct()`
  的 8 次前向），那是**推理口径**，不是**训练口径**。
- 于是 Phase 3 的核心对照被写成了 **m=2 (12.5%) vs m=1 (100%)，即 8 倍差**；
  而**同一口径下应是 12.5% vs 50%，即 4 倍差**。

| 口径 | m=1 | m=2 | 倍数 |
|---|---|---|---|
| 训练采样（忠实 VALID，4/8 角点） | **50%** | **12.5%** | **4×** |
| 推理扫描（8 相位全覆盖） | 100% | 12.5% | 8× |

**影响**：Phase 3 "m=2 落后 m=1 三到七倍"这个结论**方向不变**，但描述里隐含的"覆盖率差 8 倍"
**高估了一倍**。已按 R02 要求把该实验更名为**"稀疏覆盖 + 远距离耦合实验"**，不得再据此说
"同覆盖率下优化方向无效"。更正记录进 `Project_log.md` L54。

覆盖率已固化为单元测试，防止再混：
`tests/test_v3_paired_preprocess.py::test_m1_eight_phases_partition_the_volume`、
`::test_m2_coverage_is_one_eighth`。

---

## 3. 由此产生的两条明确待办（未运行）

1. **faithful VALID 基线必须用整卷推理（`infer.py`）复现，不能用 `reconstruct()`。**
   现有 Phase 3 的全部数字都是 `reconstruct()` 口径，因此**不能代表 faithful VALID**。
   → 实验 A 的第一件事。
2. **cross 配对臂必须在图上明确标注"V3 自创，非上游"**；与 faithful 的比较不得表述为
   "VALID 的变体优于 VALID"。

---

## 4. 未核验清单（不猜）

| 项 | 状态 |
|---|---|
| 上游 `weight_identity` 的默认系数（上游为固定 1，本仓库可调） | 🟡 需逐行核对 `train_pipeline.py` 的 `--weight_idt` 默认值与本仓库默认值 |
| 上游网络 `models/model_CRN.py` 与本项目 `models/network.py` 的逐层对照 | ⬜ 未做 |
| 上游 DWT/Hessian 与本项目实现的一致性 | ⬜ 未做（本项目有 `test_dwt3d_haar_energy`、`test_hessian_loss` 两项自测，但未与上游比对） |
| 上游 `data_patchify_indices.py` 的 patch 切分与本项目 `infer.py` 滑窗的边界处理 | ⬜ 未做 |
| 上游 PMID/数据集预处理与本项目的 SVD 带是否同口径 | ⬜ 未做（与本仓库 R13 的频带问题相关） |

---

## 5. 通俗总结

**这一步到底证明了什么。** 我们把原版 VALID 的源码真正下载下来逐项比对了一遍。结论有两半。
**好的一半**：我们训练时用的采样规则——每个 2×2×2 小块里选 4 个角点、随机挑一种"俄罗斯方块"
形状、再随机分配谁当输入谁当目标——**和原版一字不差，连八个模式的排列顺序都一样**；训练时
每步重新抽签这一点也一样。
**不好的一半**：我们在推理时用的那套"把八个相位各跑一遍再拼回去"，**原版根本没有**，是我们
自己发明的；另外那个"目标取互补方块"的配对方式也是我们自己造的，不能算原版行为。

**对项目主张意味着什么。** 更关键的是我们发现了一个**一直没写清楚的口径错误**：我们反复说
"完整采样覆盖 100%"，但那说的是**推理时把八个相位全扫一遍**。原版训练时每个小块只用 4 个
角点，也就是 **50%**。所以"稀疏版比完整版少八倍数据"这个说法要改成"少四倍"。方向没错
——稀疏版确实差得多——但我们把差距**夸大了一倍**。

**哪些话现在不能说。** 不能说"我们已经复现了原版 VALID"——训练规则对上了，但基线根本没重训过，
而且我们此前的所有数字都是用自己那套推理跑出来的，不是原版的。不能说"稀疏采样比原版差八倍"，
只能说"在同等口径下差四倍"，而且这个倍数只对当前这套构造成立。不能说"互补配对是 VALID 的
一种变体"，它是我们自己加的。
