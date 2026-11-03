# Round9 / Phase 1A：与上游 VALID v1.0 的逐项实现保真核对

日期：2026-11-03
计划：`reviews/V3_PR1_round9_shared_valid_restoration_plan_2026-11-03.md`（Phase 1A）
上游：`/tmp/valid_upstream`（FDU-donglab/VALID，GPL-3.0，不入库）
可复算证据：`scripts/v3_round9_upstream_audit.py` → `reports/v3_round9/upstream_audit.json`
单测：`tests/test_round9_phase1a.py`（8 条，全过）

---

## 0. 本文件的写法约束（先说清，免得又被当成"文档通过=实验通过"）

本轮计划要求：「**只有被测出相同的才能叫 faithful**，其余全部进差异表」。所以本文件把每一项
分成三档，**每档都必须有测量支撑**：

| 档位 | 含义 | 判定依据 |
|---|---|---|
| **实测相同** | 同一输入下数值/结构被量出相同 | `upstream_audit.json` 里的具体数字 |
| **已记录差异** | 不一样，且知道为什么、影响是什么 | 差异表 + 影响说明 |
| **无法核对** | 上游没有可比对象，或本轮没验 | 明写"无法核对"，不写成"一致" |

**本轮不改任何历史结论**：Round8 的 `INFERENCE_ARTIFACT_FIXED_METHOD_NO_GAIN` 与更早的负结果
全部保留，本文件只回答"我们的实现和上游差在哪"。

---

## 1. 已修代码（本轮真正动过的东西）

### 1.1 `--role-perm {per_block, global}` 透传与校验（计划点名的必改项）

问题（Round8 复核查出、已在 `datasets/sampling.py` 承认过）：上游 `RandomSampler_0123.__init__`
只 `random.shuffle([0,1,2,3])` 一次，四个 `draw()` 的结果套到**所有** 2×2×2 块上
（`datasets/sampling.py:264-268`），即**一个 patch 一个全局置换**；本仓库历史实现是
`torch.rand(nb,4).argsort(dim=1)`，**每块独立**。`train.py` 之前**根本没有把作用域透传出去**，
于是任何声称"VALID-faithful"的训练命令实际上都跑的是 per_block 变体。

本轮改动：

- `datasets/sampling.py`：`valid_random_tetris(..., role_perm="per_block"|"global")`、
  `valid_random_tetris_strided(..., role_perm=...)`（Round8 已加）。
- `train.py`：新增 `--role-perm`，在 `sample_views()` 的 random 分支透传给上面两个函数。
- `train.py`：**显式校验**——`--role-perm global` 与 `--mode fixed/guided/patch_random` 或
  `--pairing cross` 同时给会直接 `raise ValueError`。**不允许静默忽略**：静默忽略会产生
  "命令看起来 faithful、实际不是"的假象，这正是上一轮被指出的毛病。
- `train.py`：`--mode random` 的**默认语义不变**（`per_block`），以免旧 run 不可复现；
  Round9 的 faithful 臂必须在命令里显式写 `--role-perm global`（见 `prereg.md`）。
- `datasets/sampling.py:324`：`random_tetris_subvolumes` 的 docstring 原来写"用
  `valid_random_tetris` 获取 faithful baseline"，纠正为"上游每块重抽 tet，本函数整个 patch
  只抽一个 pattern，**不是**上游采样器"。

**未改**：`train.py` 里 `--lr-schedule` 的说明改过两次，见 §3 第 12 项（先改错、再改对，
两处都留在文件里，因为第一版判断本身也是一条需要记录的教训）。

### 1.2 这次改动**没有**碰的东西

loss、优化器、网络结构、数据管线、推理脚本、`mode=random` 的默认行为、任何既有 run 的产物。
所以 Round1–Round8 的所有数字仍然可复现。

---

## 2. 实测相同的项（每项都有数字）

| # | 项 | 上游位置 | 我们的位置 | 实测结果 |
|---|---|---|---|---|
| 1 | 8 个合法 Tetris 模板 | `datasets/sampling.py: idx_pair` | `TETRIS_PATTERNS` | `torch.equal == True`；且**每个 template 都不含任何体对角对**（{0,7},{1,6},{2,5},{3,4}）——这是上游"互补 tet"配对成立的前提 |
| 2 | 每块四个角色互不相同 | `generate_mask_pair` | `valid_random_tetris` | 5 次抽样 × 8 块全部是上游 `idx_pair` 里的合法 tet，四个角互不重复（`upstream_every_block_is_a_legal_tet=[True]*5`） |
| 3 | 角色置换作用域 | `RandomSampler_0123` | `role_perm="global"` | **实测**上游一个 patch 内所有块的 rank 元组唯一（5 次抽样均为 1 种；例：`(3,1,0,2)`），而每块抽到的 tet 有 5–6 种不同 → 置换全局共享、tet 每块独立。我们的 `global` 模式同样只有 1 种 rank 元组（`(0,3,2,1)` @seed 5），`per_block` 模式 8 块 8 种 |
| 4 | 视图形状 D/2×H/2×W/2 | `generate_subimages` | 同上 | 6×8×8 卷 → 4 × (3,4,4)；逐体素坐标编码验证：视图体素 (bz,by,bx) 必落在原卷 2×2×2 块 (bz,by,bx) 内，且坐标算出的角编号 == 视图实际取到的角编号 |
| 5 | 四视图取自**同一个** tet | `train_pipeline.py:753-757` 全用 `noisy_output_1` 的 DWT、目标来自同一次抽样的 `sub3/sub4` | `valid_random_tetris` | 实现上四个视图由同一次 `rd_pair_idx` 派生；我们的 random 臂同样由一个 tet 的四个角构成 |
| 6 | 两条分支共用一个网络 | `noisy_output_1 = model(sub1)`、`noisy_output_2 = model(sub2)`，同一 `model` | `ha, hb = model(xa), model(xb)` | 单测 T4：forward hook 计数 == 2、参数集唯一且等于优化器参数集、`len(param_groups)==1` |
| 7 | CRN 结构 | `models/model_CRN.py: CRN` | `models/model_CRN.py: NetworkCNR` | 112 个参数张量各一份、形状序列逐元素相同、总参数 **1,543,654**；**权重互传后前向 `max|Δ| = 0.0`（逐位相同）** |
| 8 | `HessianConstraintLoss3D` | `models/network.py` | `models/network.py` | 结构一致；端到端在**同一批随机张量**上，我们的 reg 与"上游把 typo 修掉"的 reg **差 = 0.0** |
| 9 | 邻项/恒等项 | `0.5·MSE(o1,t3) + 0.5·MSE(o2,t4)`、`MSE(o1,o2)` | `loss_terms()` | 同一批张量上 `neighbor`、`identity` 的 `max|Δ| = 0.0` |
| 10 | 总损失组合与权重 | `neighbor + identity + weight_reg·reg`，`weight_reg=1e-4` | 同 | 逐项相加顺序一致；默认权重一致 |
| 11 | 推理方式 | `test_pipeline.py` 满 patch 前向 + `reconstruct_image` 累加/计数平均 + 去 padding | `infer.py` 满 patch 滑窗、`acc/count` 平均 | 同构。单测 T6：喂给网络的必须是**整个** 32×64×64 patch（不是相位子视图），覆盖率最小 1（无空洞），恒等模型下 `max|Δ| = 0.00e+00`；`infer.py` 源码内不含 `[::2]`/`decim`/`backfill`/`phase_views` |
| 12 | 有效学习率 schedule | `train_pipeline.py:741` 每 batch `adjust_learning_rate`（poly, power=0.9） | `--lr-schedule poly`（默认） | 复算 5 个 epoch：上游有效值 `(1e-4, 9.91e-5, 9.549e-5, 9.095e-5, 5.359e-5)` vs 我们 `(1e-4, 9.90995e-5, 9.54885e-5, 9.09533e-5, 5.35887e-5)`，差异仅为上游那一步 `round(·, 8)` |
| 13 | 优化器 | `Adam(lr, betas=(0.9,0.999))`；`amsgrad` 传进 argparse 但**从未传给优化器** | `Adam(lr, betas=(0.9,0.999))` | 相同；我们同样**没有**用 amsgrad（这里"没抄错"= 都没用） |
| 14 | 默认超参 | `base_features=16`、`n_groups=4`、`weight_reg=1e-4`、`seed=3407`、`batch_size=1` | `--features 16 --groups 4 --weight-reg 1e-4 --seed 3407`，每步 1 个 patch | 相同 |

---

## 3. 差异表（逐项，含影响与处置）

| # | 项 | 上游 VALID v1.0 | 本仓库 | 影响 | 处置 |
|---|---|---|---|---|---|
| 1 | **角色置换作用域默认值** | 一个 patch 一个全局置换 | 默认 `per_block`（历史行为） | 旧 run 的采样几何与上游不同；这正是 Round8 被指出的点 | 已加 `--role-perm`；Round9 faithful 臂**必须**显式 `--role-perm global`（`prereg.md` 已写死） |
| 2 | **输入→目标配对** | 四个视图全取自**同一个** tet 的四个角（`mask1..4` ← 同一次 `rd_pair_idx`） | `--pairing within`（默认）符合；但仓库另有 `--pairing cross`（目标取**互补** tet 的体对角位） | `cross` 是**本仓库自创**，与上游不可比；把它当成"VALID baseline"会算错 | `cross` 保留给历史 run 与引导采样实验；**Round9 禁止** `--pairing cross`（已由参数校验挡住 `global`+`cross` 组合） |
| 3 | **上游 DWT typo** | `train_pipeline.py:753-754` 两行都是 `DWT3D(noisy_output_1)`，即 reg 的 8 个通道里后 4 个来自 output_1，`noisy_output_2` 的频带从未进入正则项 | `wavelet_hessian_loss` 用 `[band(ha), band(hb)]` 对称拼接 | 在同一批随机张量上 reg：上游（带 typo）`36.6044` vs 我们 `36.7816`，差 `0.177`（≈0.5%）；换算到总损失 3.9474 vs 3.9474（`weight_reg=1e-4` 压得很小） | **有意修掉**，且已验证"把 typo 改掉的上游"与我们**差 0.0** → 差异**只**来自 typo |
| 4 | **DWT 频带符号约定** | `DWT_3D("haar")` | `dwt3d_haar` | 8 个频带里 **LLH、LHL、HLL、HHH 与上游反号**（`max|Δ|` 全在 2–5e-7，即纯符号），等价于我们的高通滤波器 = 上游高通的相反数。包含 H 个数为奇数的频带全部反号，与此一致 | 对 reg **无影响**（每通道独立取二阶导再平方）：单测 T5 把 LLH/LHL/HLL 取负后 reg 完全不变；端到端 reg 差 0.0（第 2 项）。**记录为差异，不改**（改了没有任何行为差别，只会让历史权重对不上） |
| 5 | **reg 用哪四个频带** | `753-757` 取 `0,1,2,4` 号 = LLL/LLH/LHL/**HLL** | `VALID_REG_BANDS = (LLL, LLH, LHL, HLL)` | 相同。**这条是本轮第一次写错、被测量纠正的**：审计脚本初版误用"前四个"（3 号是 LHH），导致 reg 差 0.544；改成 `(0,1,2,4)` 后差 0.0 | 已在 `scripts/v3_round9_upstream_audit.py` 里注释说明，避免下次再错 |
| 6 | **参数名/state_dict 键** | `pool`/`act`/`gate` | `global_pool`/`relu`/`sigmoid` | `set(state_dict) 相同 == False`：**上游权重不能直接跨仓库 `load_state_dict`**，必须先按形状顺序搬运 | 本轮审计与单测都按"形状序列 + 顺序拷贝"比对（`crn_forward_max_abs_diff_after_weight_transfer = 0.0` 证明搬运正确） |
| 7 | **梯度裁剪** | `clip_grad_norm_(max_norm=20.0)` | `--clip-grad` 默认 `100.0` | 上游更紧。默认不同 ⇒ 旧 run 的更新幅度与上游不同 | Round9 训练命令**显式写 `--clip-grad 20.0`**；默认值不改（旧 run 可复现） |
| 8 | **patch 尺寸** | `z_patch=64, w_patch=256, h_patch=256` | `--patch 32 64 64` | 我们的数据体只有 37×64×64（帧数 37），无法用上游尺寸；patch 小 ⇒ 相位子视图 16×32×32，CRN 的 `AvgPool3d(8)` 第一分支刚好可用 | 记为**规模差异**，不是算法差异。代价：感受野相对血管尺度更小，会影响"能否恢复长程结构"，写进结论的适用范围 |
| 9 | **输入表示** | 直接读 TIF 灰度（论文用 log 压缩后的 PD 图），数据侧自己归一化 | 训练输入 = `log1p(PD/s)`，`s` 取自真实数据标定（`reports/v3_round4/b1_norm.json` 那套用于真实数据的标准化；S0 正对照用各场景自己的 `s`） | 我们比上游多一层"显式 log 域"；在 S0 上这也让**噪声是加在 log 域的**（与上游"缺噪图"不同） | 记为**领域设计差异**（本项目就是要做 PD 去噪）。Round9 的对照组（恒等/高斯/Wiener）在同一 log 域 → 同一真值域里评，不引入额外不公平 |
| 10 | **训练步数与数据量** | 100 epoch × 每 epoch `len(loader)` 步 | Round9 固定 ≤10,000 optimizer steps | 上游没给"步数"这个量，只能给"步数预算" | `prereg.md` 写死 10,000 步（含 2,000 步检查点）；两臂同预算 |
| 11 | **`--lr-schedule` 说明文本曾写错** | 686 行建 `LambdaLR(warmup+余弦)`；741 行每 batch 再直接赋值 poly | 说明文本先写"poly 是上游"，Round8 又被我改成"cosine 才是上游"，**两次都不完整** | 最终结论：**实际生效的是 poly**（741 行每步覆盖 686 行），我们 `cosine` 是自造 | 说明文本已改为完整叙述两套 schedule 与覆盖关系；默认 `poly` 不变 |
| 12 | **推理时的归一化回写** | `reconstruct_dataset` 内部乘回 `std`/加回 `mean` | `infer.py` 只输出网络域结果，反变换在下游脚本做 | 不影响数值，只影响"谁负责反变换" | 记录，不视为差异 |
| 13 | **多卡** | `DataParallel`（多卡时） | 单卡 | 只影响吞吐 | 记录 |
| 14 | **控制台/交互** | 依赖 PyQt5 GUI 线程 | 纯 CLI | 无算法影响 | 记录 |

---

## 4. 单测清单与结果（`tests/test_round9_phase1a.py`）

| 测试 | 验什么 | 结果 |
|---|---|---|
| T1 | 坐标编码卷（6×8×8 → 48 块）：每块 4 个不同角色、属一个合法 tet、rank 是置换、视图形状 D/2×H/2×W/2、**逐体素**坐标映射正确；`per_block` 与 `global` 两种都验 | ok |
| T1b | `stride>1` 的随机臂保留同样块几何，且 `global` 置换真的全局 | ok |
| T2 | 作用域：`global` 全块共用 1 种 rank 元组 / `per_block` 8 块 8 种；同 seed 逐块可复现（seed=5 记录在输出里） | ok |
| T3 | **直接调用上游** `generate_mask_pair` 量出：一个 patch 只有 1 种 rank 元组、8 块里 5–6 种 tet | ok |
| T4 | 单参数集（1,543,654）/ 单优化器 / 两输入走同一 `f_θ`（hook 计数 2）/ neighbor·identity·reg 对 `ha,hb` 的梯度**都非零** | ok |
| T5 | 把 LLH/LHL/HLL 取负后 reg 数值不变（`37.072365`）⇒ 符号差异不是行为差异 | ok |
| T6 | 推理满 patch、无空洞、无相位抽稀/回填，恒等模型逐体素还原（`max|Δ| = 0.00e+00`） | ok |
| T7 | `--role-perm global` 与 `fixed`/`cross` 组合**真的报错**；`random + global` 一路走到读文件 | ok |
| T7b | CRN 与上游同构（112 张量 / 1,543,654 参数），权重互传后 `max|Δ| = 0.0` | ok |

跑法：

```bash
/data2/wjb/anaconda3/envs/threePM/bin/python tests/test_round9_phase1a.py
/data2/wjb/anaconda3/envs/threePM/bin/python tests/test_smoke.py      # 旧回归，也全过
/data2/wjb/anaconda3/envs/threePM/bin/python scripts/v3_round9_upstream_audit.py
```

---

## 5. 现在还**不能**说的话

1. **不能说"我们完全复现了 VALID"**。能说的是：CRN、Hessian 正则、loss 组合与权重、
   有效学习率、tet 模板、置换作用域、推理方式这七项被**测量**为相同或等价；
   patch 尺寸、输入域、梯度裁剪默认值、配对方式（`cross` 臂）这四项**就是不同**。
2. **不能说"采样忠实就等于上游性能"**。上游 patch 64×256×256、100 epoch 规模远大于本轮
   ≤10,000 步 × 16×32×32 子视图；规模差异会改变可恢复的结构尺度。
3. **不能说"已验证上游训练是稳定/有效的"**。本文件只核"实现是否一样"，没有重跑上游原仓库。
4. **不能说 `--role-perm per_block` 是错的**。它只是**不是上游**；Round1–8 的结论挂在它上面，
   本轮不回收。
5. **不能说 DWT 符号差异已被消除**。它被记录、被证明对 reg 无影响，但**代码里仍然是反号的**。

---

## 6. 通俗总结

这一轮先把"我们到底是不是照着 VALID 原仓库做的"这件事查实。查的办法不是读一遍源码然后声称
"我抄对了"，而是把上游那个仓库真的跑起来，用同一批张量去比数字，比完把每一项分成"量出来一样"、
"确实不一样但知道为什么"、"没法比"三类。

结论是：**网络骨架完全一样**——上游模型和我们的模型参数量一样、形状一样，把上游权重搬过来以后
同一输入下的输出差**恰好是 0**，一个比特都不差。**损失函数的三块也基本一样**——目标项和恒等项
差 0；正则项在我们把上游一个明显的笔误（同一份输出做了两次小波变换，另一条分支的频带根本没进
正则项）改掉之后，也和上游**差 0**。**采样器的规则现在也对上了**：上游每个小块随机挑一套四个角，
但"这四个角分别当输入A、输入B、目标A、目标B"这个顺序，整张图只抽一次、所有小块共用——这一点
是我们这次真的去跑上游代码量出来的，不是猜的。我们以前默认是每个小块各自抽顺序，现在加了一个
开关，本轮会显式打开"整张图共用一个顺序"。

确实不一样的地方有四类，都要如实写着：一是训练的图块尺寸（上游 64×256×256，我们只有
32×64×64，因为我们的数据本来就只有 37 帧 × 64 × 64）；二是输入表示（我们是明确的对数域，
上游是直接读图）；三是梯度裁剪的默认值（上游 20，我们历史默认 100，本轮会显式用 20）；
四是仓库里另有一条"目标取互补小块"的配对方式，那是我们自己加的，不是 VALID 的，本轮不许用。
还有一个小波符号问题：八个频带里有四个和上游反号，但因为正则项是把每个频带取二阶导再平方，
反号完全不影响数值——这一点也做了单测验证，两条路的正则项数值一模一样。

**这一步证明了什么**：我们终于可以**有资格**说"这一臂是照着 VALID 做的"，而且这句话是有数字
背书的。**这一步没有证明什么**：它完全没说 VALID 这种做法在我们的数据上能不能去掉噪声——
那要下一步真训练两条臂（一条照上游的随机采样、一条固定最强调度）然后和恒等、高斯、维纳三个
平凡滤波比。**现在还不能说的话**：不能说"我们复现了 VALID 的效果"，也说不清上游那种小波正则
在我们这个更小的图块上是否还起同样作用——这些都要等训练结果。
