# Round4 预注册（Round4-0 / B / D / C 的冻结设定）

> 依据外部任务书 [V3_PR1_round4_last_chance_stop_loss_protocol_2026-10-29.md](V3_PR1_round4_last_chance_stop_loss_protocol_2026-10-29.md)
> （commit `2fba290`），审查基准 SHA `469935d`。**本文件在跑任何 Round4 数字之前写出**；
> 一旦写出，判据、split、阈值、预算不得回改。若有必须修改处，新建 `V3_PR1_round4_prereg_v2.md`。
>
> **不覆盖**：Round1–3 的报告、原始 JSON、旧 Gate（`G-B1`、`G-S0-2` 等）全部原样保留。
> 旧 Gate 若被本轮的终点取代，只做引用说明，不回填历史阈值。

## 0. 一句话目标与止损

本轮只回答三个可决策问题（任务书 §0）：(1) 结构保真约束下模型是否有对 Gaussian/Wiener 的独立优势（0）；
(2) 换表示域（linear / `log1p` / `log(PD+ε)`）能否让基础训练出现清晰正向信号（B）；
(3) 若基础链有改善，物理先验是否比最强固定采样与打乱先验有额外价值（C，前置条件不满足则 `NOT_RUN`）。
**硬预算：全部新增训练 ≤8 GPU·h 且 ≤8 作业**（B pilot 最多 3 次）。达到任一上限即停止并出决策。

## 1. 冻结的集合、split 与预处理

| 项 | 冻结值 | 来源/理由 |
|---|---|---|
| 低噪声档 `low` | `LEVEL_SIGMA['low']`，数据 `data/processed/v3_round2_s0/low*/` | Round2 已冻结 |
| 高噪声档 `vlow` | `LEVEL_SIGMA['vlow']`，数据 `data/processed/v3_round2_s0/vlow*/` | 同上 |
| 校准集（选参用） | **e900, e901** | 与 A5 一致；选参只允许在这两卷上做 |
| 留出集（主结果） | **e902, e903, e904, e905** | 独立于选参；主结论只允许用它 |
| 确认集（最终确认） | **e906–e911**（6 卷，本次新生成，split 先写死） | 生成用 CPU、不计 GPU 预算；**在 P0 主结果表冻结之前不做任何评估** |
| 训练集 | `low/train/` 12 卷（s200…s211），`vlow/train/` 12 卷 | B1 未改 |
| 预处理尺度 | 每卷 `s = manifest 里的 s`；log 域 `log1p(PD/s)` | 与 Round2/3 完全一致 |
| `log(PD+ε)` 的 ε | **ε = 1e-3 · s**（每卷各自 s） | 见 §3.1；只在训练/校准集上确定，不在留出集重算 |
| 有效 mask | npz 内 `valid` | 三臂共用同一 mask |
| 评价域 | 主报 **反变换到线性 PD** 后的误差 + log 域描述指标；两域单列 | 任务书 §B1 要求 |
| 显示窗 | 固定窗宽窗位（线性 PD 与 dB 各一套，所有方法一致） | 防"挑图" |

**PSF 与分辨率映射（必须先讲清，否则几何指标无意义）**：`PSF σ = (3.0, 2.37, 3.97)` 体素，
PD 图像的等效核 `PSF_PD = PSF·√2 = (4.24, 3.35, 5.61)` 体素；仿真血管半径 **0.75 / 1.1 / 1.5 体素**。
⇒ **管径（宽度）指标在本仿真里不可解析**：成像后的半高宽由 PSF 决定，与去噪方法无关。
故 `width` 状态固定为 **`NOT_RESOLVABLE`**（不是"没做"，是有明确理由的撤除），
替代指标见 §2.2 的 PSF 匹配口径。这条与 R12 报告的撤除理由一致，本轮不推翻。

## 2. Round4-0（P0，无训练）：公平 baseline 与结构安全对照

### 2.1 参赛臂（各档只用同档权重，不混名）

| 臂 | 定义 | 选参（只在 e900/e901） |
|---|---|---|
| `input` | 观测 `obs1` 原样 | 无 |
| `gauss` | masked Gaussian，`σ ∈ {0.2,0.3,0.4,0.5,0.6,0.8,1.0}` | 复用 **A5 已冻结** σ*：low **0.6**、vlow **0.5**（**不重选**） |
| `wiener` | 非局部 Wiener，窗口 `w ∈ {3,5,7,9,11}` | `w*` = e900/e901 上 n2n 最小的 w（本轮新选，**先选再看留出**） |
| `net_direct` | `runs/v3_round2_s0/<lvl>/S_dense_s3407` | 无（B1 训练早已冻结） |
| `net_residual` | `runs/v3_round3/b1_<lvl>/B1_<lvl>_residual_s3407/latest.pt` | 同上 |
| `net_zeroinit` | `runs/v3_round3/b1_<lvl>/B1_<lvl>_zeroinit_s3407/latest.pt` | 同上 |
| `truth_img` | `log1p(gt_imaging_pd/s)`（无噪成像真值，PSF 已含） | 参照（已知答案，不参与排名） |

跨档组合（low 权重评 vlow 等）**允许**，但只标 `cross_level`、**不进任何判定**。

### 2.2 指标（冻结定义，P0 主报这六项）

1. `n2n = mse(pred, obs2) / mse(obs1, obs2)`（沿用旧端点，仅作连续性对照）。
2. `rmse_img`：**对 `log1p(gt_imaging_pd/s)` 的 RMSE**（PSF 匹配的无噪真值；旧 R12 用
   `gt_flow_pd` 的锐利真值，这是"真值在检测指标上系统性输给含噪图"的直接原因，本轮改用成像真值）。
3. `recall_at_fpr5`：弱血管 mask `weak = valid & (0.1 ≤ rho < 0.5)`，阈值取 `quantile(pred[bg], 0.95)`
   （`bg = valid & rho ≤ 0`），即**背景假阳性率匹配到 5%** 后比召回。
4. `ncomp_ratio`：预测前景（取与 `core` 同体素预算的 top-k）连通分量数 / `core` 连通分量数（`MIN_COMP_VOX=8`）。
5. `fp_blobs`（新）：`pred[bg]` 上面阈值（同第 3 项阈值）的连通分量中体素数 ≥8 的**个数**——
   这是"假血管计数"的显式替代，用来做"造血管"负控制。
6. `bg_std_ratio`：`std(pred[bg]) / std(input[bg])` —— 背景抑制程度的统一尺度。

`width` → `NOT_RESOLVABLE`（见 §1）。`psnr_db`/`ssim` 只作描述性记录（R06/R12 已证伪其单调性）。

### 2.3 结构指标的正负控制（**闸门前置**，未过则一切结构结论作废）

已知答案臂：`truth_img` / `input` / `gauss(σ*)` / `cut`（把 `truth_img` 里一条血管置零）/
`fake`（在 `truth_img` 的 `bg` 区加一条合成血管）。

**通过条件（全部满足才算 `METRIC_RESOLVED`）**：

| 控制 | 期望 | 判定 |
|---|---|---|
| 正控制 1 | 无噪成像真值必须优于含噪输入 | `recall(truth_img) > recall(input)` 在 **e900–e905 共 6/6 卷**成立 |
| 正控制 2 | `rmse_img(truth_img) < rmse_img(input)` | 6/6 卷 |
| 正控制 3 | 连通性单调 | `abs(ncomp_ratio(truth_img) − 1) ≤ abs(ncomp_ratio(input) − 1)`，6/6 |
| 负控制 A（断枝） | 断血管必须被检测为退化 | `recall(cut) < recall(truth_img)` **并** `ncomp_ratio(cut) < ncomp_ratio(truth_img)`，6/6 |
| 负控制 B（造假） | 造血管必须被计入假阳性 | `fp_blobs(fake) > fp_blobs(truth_img)`，6/6 |

任一条不满足 → 记录 **`METRIC_UNRESOLVED`**，此时**只报 `n2n`/`rmse_img`/`bg_std_ratio` 三个可核验量**，
**不允许**用结构指标宣告任何"结构安全"/"结构受损"，B pilot 的"结构增益"判据同步失效（见 §3.3）。

### 2.4 Pareto 口径（"结构保真—背景抑制"）

- 每臂一条曲线，横轴 `bg_std_ratio`（越小=背景抑制越强），纵轴 `recall_at_fpr5` 与 `rmse_img`（两条）。
- 旋钮（**同一族内的单参数扫描，不是重新训练**）：
  - `gauss`：σ 网格（§2.1 七个值）；
  - `wiener`：w 网格；
  - 网络臂：**输出混合** `y_β = x + β·(net(x) − x)`，`β ∈ {0, 0.25, 0.5, 0.75, 1}`（β=0 即输入，β=1 即原输出）。
- **选点在 calibration 上，曲线在 held-out 上画**。允许在留出集上画整条曲线（曲线不是选择），
  但**不允许**在留出集上挑 β / σ / w 后当"最优臂"报。
- 支配定义（冻结）：在某个留出体上，若存在基线的 `bg_std_ratio ≤` 网络臂且 `rmse_img ≤` 网络臂
  且 `recall_at_fpr5 ≥` 网络臂，并且至少一项严格更优 → 该体上"网络臂被支配"。

### 2.5 0 Gate（冻结）

- 4 个留出体（e902–e905）中，若网络臂**每个体**都被某条 baseline 支配 → **`BASELINE_DOMINATED`**：
  停止"仅改网络架构"分支；仍允许执行 §3 的**最后一次 B 表示 pilot**，但不得再加 CRN 模块、不得改一堆损失。
- 若 `METRIC_UNRESOLVED` → 不得以 `n2n` 单项宣告任何胜利；B 只做表示域本身（方差/条件均值），
  B pilot 的训练**只在 §3.3 的"非结构判据"下进行**，且结论标签固定为探索。
- 若存在非支配区 → 精确写出区间与该区间内血管/噪声代价，交用户决定，不擅自扩大预算。

### 2.6 交付

`reports/v3_round4/baseline_pareto.md`、`baseline_pareto.json`、`fig_baseline_pareto.png`、
`metric_controls.json`（已知答案表原样落盘）、以及生成 e906–e911 的 `manifest` 与 seed 清单。

## 3. Round4-B：表示域 pilot

### 3.1 B0（无训练）：真实数据三域统计

- 数据：SA64 / HSA64，**只在 calibration 帧**上分箱（与 R0-v2 同一协变量分箱），
  比较 `linear PD`、`log1p(PD/s)`、`log(PD+ε)`（ε = 1e-3·s）的**条件方差、条件残差均值、空间相关**。
- 明确分层写：**已测事实**（本次实测）/ **代理假设**（N2N 有效性所需条件）/ **训练有效性**（未测，不得由前两者推出）。
- R0-v2 只测过 `log(PD)`，**不得**把它的斜率 0.033/0.104 直接当作 `log1p` 的结论。
- 加一个**乘性噪声正控制**（合成乘性扰动），检验 log 统计量的方向；并写明奇偶块在共同 SVD 投影下
  仍可能相关，非共享参考残差相关 0.017/0.036 只是差分代理。

### 3.2 B1（≤3 次短训练）：单变量表示对照

| 项 | 冻结值 |
|---|---|
| 档位 | **low**（理由：高斯优势最大 0.618 vs 网络 0.825，表示域若真有效必须在此显现；vlow 只作后续可选扩展） |
| 网络/头 | `--mode dense --arch-head residual --zero-init-final`（B1 最佳正控制，不变） |
| 骨干 | `features=16, groups=4, patch=32×64×64`（不变） |
| 优化 | `--lr 1e-4 --lr-schedule poly --pairing within --seed 3407`，`15 epoch × 500 steps` |
| 三臂 | `linear`（`PD/s`）、`log1p`（`log1p(PD/s)`）、`logeps`（`log(PD·1e-3/s + 1e-3)` 的等价形式，见下） |
| 目标 | 同一变换作用于 `gt_imaging_pd`（**PSF 匹配的成像真值**，不是锐利的 `gt_flow_pd`） |
| 归一 | 每域用**训练集**统计量做一次仿射标准化（减均值/除标准差），常数写进 `args` 冻结；**不在留出集重算** |
| 评价 | 统一**反变换回线性 PD** 后与 `gt_imaging_pd` 比；另列转换域指标 |
| 选参 | 只看 e900/e901；e902–e905 报结果 |

`log(PD+ε)` 的精确定义（冻结）：`x = log(PD·ε/s + ε)`，其中 `ε = 1e-3`，
反变换 `PD = s·(exp(x) − ε)/ε`。取这个形式是为了让 `x≥0`、零/低亮背景下反变换数值稳定，
且保留与 `log1p` 的单调可比性；**该定义在跑第一臂之前写死，不因某臂好训而调整**。

**公平性护栏（针对"某个表示更容易训练"）**：三臂同 optimizer/lr/steps/seed/patch/数据划分，
且各自做了训练集仿射标准化 → 有效步长量级一致。仍在报告里记一条保留：
"标准化只对齐了一阶/二阶尺度，不能保证三域的可优化性完全相同"。

### 3.3 B Gate（冻结）

- 判据只看 **e902–e905 四体的 pairwise**（场景为单位，不把体素当独立样本）：
  某表示臂"有清晰正向信号"需同时满足：
  1. 相对 B1 现正控制（`log1p` 即当前训练域）`rmse_img` 改善 **≥3/4 体**；
  2. 且不劣于 `gauss(σ*)` 的 `rmse_img`（4/4 体）；
  3. 若 `METRIC_RESOLVED`，还要求 `recall_at_fpr5` 不比 `gauss(σ*)` 差超过 **0.02**（4/4 体）。
- 三臂全不满足 → **`B_NO_GAIN`**：立即停止 CRN 结构/损失调参；D 只做文献核对，
  不做昂贵全量复现；I1 不进入大规模训练。
- 只有 `logeps` 满足 → 必须先证明收益来自条件方差/偏置改善（B0 的对应量），再谈 C；**不得**归因 I1。
- 追加训练只允许**一条**：最优臂从 15 epoch 续训到 40 epoch（同一个 checkpoint 续训，不换种子）。

## 4. Round4-D：SVD-SN2N（2026, Zhang et al.）核对

- 唯一对象：*SVD-SN2N: Self-Inspired Noise2Noise Learning for Denoising Log-Compressed SVD-Filtered
  Ultrasound Imaging*，bioRxiv DOI `10.64898/2026.09.10.750668`。**不用** 2024 荧光 SN2N 替代。
- 逐项列出：输入表示、log 操作、对角子视图构造、Fourier 插值、双向损失、网络结构、训练/推理处理、
  隐含假设；每项标 **原文确定 / 推断 / 未知** 并附出处（无法访问原文的项一律标 `未知`，不得脑补）。
- 数值检验（无训练）：在现模拟正控制上比较"对角子视图构造"与"当前 Tetris 视图"在
  信号几何保持、残余依赖上的差异（视图强度相关 ≠ 纯噪声相关）。
- 若 B 全败且 D 无积极机制证据 → **不做**完整 SVD-SN2N 训练复现，直接出决策报告。
- 交付：`reports/v3_round4/svd_sn2n_fidelity_and_decision.md`（含 `未找到/无法核验` 的显式标注）。

## 5. Round4-C：触发条件（默认 `NOT_RUN`）

- 触发需**同时**满足：(a) §2.5 判定为存在非支配区且 `METRIC_RESOLVED`；(b) §3.3 至少一臂"清晰正向信号"。
- 触发后先做**无训练采样几何筛选**：检查 8 个 tetra/Tetris 候选中，正确先验所选模式/角色是否与
  fixed-4 实质不同、打乱先验是否真的改变实际采样。若退化为同一固定模式 →
  **`I1_NO_DISTINCT_INTERVENTION`**，不训练。
- 否则最多 3 臂 × 1 seed 最小 pilot（最强 fixed / 正确先验 / 打乱先验），同体素、同覆盖率、同网络、
  同表示域、同预算；无独立结构安全收益 → **`STOP_I1`**。
- 未触发时状态写 **`I1_UNTESTED_BECAUSE_BASELINE_FAILED`**，**不得**写成"I1 已被证伪"。

## 6. 预算与消耗记录（记账用，每作业一行）

| # | 作业 | 类型 | 估计 | 实测 | 备注 |
|---|---|---|---|---|---|
| 1 | P0 推理（含 Pareto 扫描、已知答案控制） | 推理 | — | 待填 | 不计 GPU·h 上限，但记墙钟 |
| 2 | B0 三域统计（SA64/HSA64） | 无训练 | — | 待填 | |
| 3 | D 数值检验 + 文献核对 | 无训练 | — | 待填 | |
| 4 | B1 pilot `linear` 15ep×500 | 训练 | 0.33 GPU·h | 待填 | job 1/8 |
| 5 | B1 pilot `log1p` 15ep×500 | 训练 | 0.33 GPU·h | 待填 | job 2/8 |
| 6 | B1 pilot `logeps` 15ep×500 | 训练 | 0.33 GPU·h | 待填 | job 3/8 |
| 7 | 最优臂续训至 40ep（若触发） | 训练 | 0.55 GPU·h | 待填 | job 4/8，上限 8 |

**累计上限**：8 GPU·h / 8 jobs。**不含**程序错误重跑（重跑同样记账）。达上限即停，交出 `DECISION.md`。

## 7. 逐项状态（执行中更新本表，不改上面任何设定）

| 任务 | 状态 |
|---|---|
| 预注册（本文件） | **READY** |
| 生成 e906–e911（两档，seed 先冻结） | READY |
| P0 已知答案/结构指标控制 | READY |
| P0 baseline + Pareto（留出 e902–e905） | READY |
| B0 三域方差/条件均值 | READY |
| B1 三域 pilot（≤3 作业） | READY（受 §3.3 与预算约束） |
| D 文献核对 + 数值检验 | READY |
| C（I1） | **NOT_RUN**（触发条件未满足时按 §5 标 `I1_UNTESTED_BECAUSE_BASELINE_FAILED`） |
| 真实数据大规模训练 | **NOT_RUN**（本协议不允许，需另行申请） |

## 8. 不改的东西（防止事后翻账）

- 旧 Gate `G-B1 = 0.80`、`G-S0-2`、A5 的 σ*、Round3 的 `--tag` 产物，全部原样保留；
- 不因 `n2n = 0.825`"只差 0.025"申请加预算；不因单张图好看、单个 seed 过线而改判；
- 不在确认集（e906–e911）上重选任何参数；不删除任何历史负结果。
