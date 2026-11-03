# SVD-VALID / PR #1 — Round9 共享网络 VALID 主线恢复与快速止损执行计划

> **日期**：2026-11-03。**适用分支**：`v3/judge-acceptance`。**基准提交**：`808070af6114b23a8e6bcedf1921b142f9f1a897`。  
> **执行指令**：**从阶段 1 直接开始**。用户明确取消原先“阶段 0 / 水波纹与显示专项诊断”，不要执行水波纹 FFT、额外 DC 探针、推理尺度归因或新增 Phase 0 实验。只保留保障代码正确性的短单元测试。  
> **文档性质**：新增预注册与修复任务书，不代表实验已经完成。旧 V1–V3/Round1–8 负结果、旧判据与权重全部保留。**PR #1 不合并。**

## 0. 执行依据与研究目标

用户关心的是：**与原始 VALID 相比，PSF/散斑或残余噪声相关性感知的四角色训练视图选择，能否在不牺牲弱微血管结构的情况下，改善三维 SVD–Power Doppler 质量？**

Round8 的已确定结果：八相位拼装人为造成棋盘；全分辨率完整 patch 推理后棋盘消失，但 Round7 的四个独立 cross-tet 训练模型在 SA64/HSA64 上仍有明显输出失真，且没有胜过固定 Gaussian/Wiener。结论仅是**终止“固定 cross-tet + 4 个网络 + 八相位回填”的实现分支**，**不是**“原始 VALID 基线已正确验证并失效”，也不是“I1 被证伪”。见 [Round8 原始报告](../reports/v3_round8_zero_train/REPORT.md)。

**核心方法应为**：
1. **采样只在训练侧**。在每个 `2×2×2` 小块随机抽一个合法的四角 Tetris；同一训练采样调用内四个角色采用上游兼容的**一次共享排列**，不同小块可选择不同 Tetris 构型。四角色构成四个 `D/2×H/2×W/2` 视图，每个角色只取每小块的一个体素（单视图总覆盖原图 1/8、四个合计 1/2）。
2. **单个共享 CRN**：`xa,xb` 分别通过相同参数的 `f_θ`，目标 `ya,yb` 取自**同一个** Tetris 四角内互补角色；使用上游定义的 neighbor MSE、预测一致性和 wavelet/Hessian 项。不能改为跨互补四面体体对角目标，也不能四个角色分别训练四个网络。
3. **推理不再重新采样**：每个模型直接对完整分辨率三维 patch 前向，重叠区融合。不能使用 `v3_round7_ours_recon.py` 的八相位回填，也不新增子视图推理拼图/插值后处理来规避失败。
4. **I1 的唯一允许方法差异**：在固定网络、损失、预处理、推理和覆盖率下，由 calibration 中的 PSF/空间噪声统计与信号保真约束改变**训练侧合法 Tetris 几何/角色分配**；并和随机、最强固定、错误/打乱先验比较。若合法候选实际上等价、物理先验不能改变训练采样决策，则应直接停止 I1。

**别将 SN2N/SVD-SN2N 的插值机制错误套给 VALID**。这轮不是复现 SN2N；若需文献比较，单独注明对照来源及实现，不改变本轮核心变量。

## 1. 阶段 1A：实现上游一致的单网络 VALID baseline（先修代码，后开训练）

**必改代码（执行者修，不由本计划伪称已修）**：
- `datasets/sampling.py` 已有 `valid_random_tetris(..., role_perm="global")`；历史默认值仍是 `"per_block"`，必须不破坏历史结果地让**新的 faithful 运行显式启用** `"global"`。
- `train.py` 当前在 `sample_views()` 的 random 分支调用 `valid_random_tetris(arr, generator=gen)`，未传 `role_perm`；补 `--role-perm {global,per_block}` 透传到 `valid_random_tetris` / strided 版本，并在本轮命令写明 `--role-perm global`。对于固定或 cross 模式，若该参数不适用，做参数校验，不得静默忽略导致假 faithful。
- 与冻结的上游 VALID v1.0 源码逐项对照：每小块 Tetris 选择概率、角色全局 permute 范围、同-tet 内输入目标映射、两路是否共享同一个 CRN、CRN 结构、各损失项权重、输入表示、采样出的 `D/2×H/2×W/2` 尺寸及 **`infer.py` 完整 patch 推理**。只把**实测一致的组件**叫 faithful；不同之处写入差异表。
- 不修改 `mode=random` 既有默认含义，避免历史 checkpoint 失去复现性；需要显式参数/单独配置让新方法忠实性可追溯。固定组采用相同共享网络与**within-tet 配对**，不允许 `--pairing cross`。
- 当前 `--sampler-config` 主要做检查/存档，不代表物理 sampler 已驱动训练；不得把读取配置等同于 I1 实现。

**仅做快速单元测试（这是阶段 1 代码验收，不是阶段 0 研究诊断）**：
- 用带空间坐标编码的 3D 偶数尺寸 tensor 检查每个小块四角色互异、来自同一个合法 Tetris、四个视图形状和逐体素坐标映射正确；
- `role_perm=global` 在一次采样调用内每块使用同一角色排列，和上游固定版本的行为对应；历史 `per_block` 保持不变，记录 RNG/seed；
- 模型有且只有**一个**可训练参数集合/optimizer；两个训练输入调用同一个对象；`ha/hb` 的 MSE/identity/正则梯度在适用样本上非零，不能忽略某个 loss；
- 推理的完整 patch 输入/输出一致、无孔洞/固定尺度标定，不调用相位抽稀与回填。只测试 shape/数值可用性，不要重新开展水波纹专项分析。

**交付**：`reports/v3_round9/implementation_fidelity.md`、相应 tests、配置及实际运行命令。若未通过则**不进入训练**，也不生成替代结果。

## 2. 阶段 1B：仅两组最小训练，检验正确的 VALID 是否有使用价值

**新训练最多 2 jobs**（两个组各 1 seed，**同一 CRN 架构，各组训练独立权重**，不是四个相位网络）。

| Arm | 网络与训练 | 相同项 | 唯一差异 |
|---|---|---|---|
| **A — VALID-faithful random** | 单个共享 CRN，两路共享权重，within-tet 四视图 | 同一 log-PD 预处理、loss、预算、patch、同一训练/验证/留出 split、完整 patch 推理 | 每个小块随机 Tetris，`role_perm=global` |
| **B — strongest feasible fixed-tet** | 同一共享 CRN，within-tet | 同上 | 事前 calibration 选定合法固定 Tetris/角色，绝不按测试结果反选 |

- **先在有 clean imaging truth 的仿真正控制**上跑，选可恢复噪声而非原 v2 的近噪声地板场景；保留血管几何真值作为辅助，最终比较**成像真值**，不要错把 flow-only PD 真值当完整观测期望。
- 训练预算：每组**最多 10,000 optimizer steps**（可先 2,000 steps 检查数值及趋势；只从同 checkpoint 续训，不能为选种子重新跑）；总量不超过 **3 GPU·h**。若阶段 1A 单测不过，GPU 支出应为 0。记录训练时间、GPU、seed、checkpoint/hash。
- 事前固定独立场景的 calibration/validation/held-out。不能用同一体积 patch 充当独立样本；已用于选高斯 σ 的 e900/e901 只作 calibration，原 e902–e905 用于 held-out，若实验已多次依据其结果改方案，须另留**未曾用于选参的独立测试场景**作为严格确认。既有 held-out 可用于透明的“历史探索诊断”，不得称首次盲测。
- Baselines 至少：identity、固定校准 Gaussian、Wiener。输出在相同线性 PD 真值域评价：MSE/PSNR、背景、弱血管/分支保真；若分割指标无法通过干净图与断枝/假血管正反控制，不用它们决定成功，只标“结构评价未建立”。
- 真实 SA64/HSA64 使用现有处理链**只读推理**作为外部检验；禁止无 GT 画面单独决定成功、禁止用它们调整归一化或阈值。固定同一参考 dB 显示及 ROI，输出输入、A、B、Gaussian/Wiener。若输入和输出动态范围差异明显，原始浮点数、同参考 dB 和线性 PD 的强度统计必须一起报告，不许逐图自动拉伸。

**阶段 1 硬 Gate**：
- **PASS（仅允许进入阶段 2）**：A 在 ≥4 个独立评估场景中多数场景改善 clean-imaging 主误差（如 MSE 下降），且没有明确弱血管/分支损伤，并在至少部分有意义的噪声—结构权衡区间不被固定 Gaussian/Wiener 完全支配；B 的变化可被解释且使用同一协议。当前只有一个训练 seed，PASS 只意味着**基础链有探索性希望**，不等于完成论文验证。
- **STOP_BASELINE**：A、B 在可信有真值场景仍无结构安全增益，或均被平凡滤波器支配。**立刻停止这一 VALID/CRN 迁移路线，不启动物理感知采样，不追加网络/损失/训练种子**。报告不能以“差一点过门”为借口申请重复试验。
- **BLOCKED**：采样正确性、评价真值或数据分组无法审计。优先报告阻断而非消耗预算掩盖。

## 3. 阶段 2：仅当阶段 1 PASS，再做 I1 最小物理采样对照

**先做无 GPU 检查**：实际训练 sampler 能从冻结的 calibration 输出**读到并使用** `N(k)`（噪声依赖）与 `S(k)`（信号可预测性），选出的合法同-tet 角色/构型必须与阶段 1 的最强 fixed **存在实际分布差异**；错误/打乱先验也必须产生不同角色选择。否则 `STOP_I1_NO_DISTINCT_INTERVENTION`，**不训练**。

只允许 **新增 2 jobs**（每个 ≤10,000 steps；全 Round9 总 GPU 时间 ≤6 GPU·h）：

| Arm | 训练与推理 | 变量 |
|---|---|---|
| **C — physically guided Tetris** | 与阶段 1 A/B 相同单共享 CRN、loss、输入域、推理、有效体素预算 | 仅依据 calibration 的物理信息，合法地选择同-tet 训练角色 |
| **D — shuffled/wrong prior** | 与 C 完全相同 | 仅打乱或错误先验；必须实际改变采样分布 |

复用阶段 1 A/B 的固定 checkpoint 与数值做四臂对照。若与阶段 1 的最强 fixed 不同模型训练预算或输入几何，则必须显示并修正混杂，不得直接归因 I1。候选间若改变 stride/m，不允许同时改变体素覆盖率、视图稀疏度、配准/评价范围；**本轮默认 stride=1**，除非相同信息量的严格对照预先得到验证。不恢复 Round7 跨互补 tet 的体对角强制配对。

**阶段 2 Gate**：C 必须在独立场景中表现出超出 A/B/D 的**结构安全、方向一致的收益**；仅赢 random、只改善 noise proxy、或正确/打乱先验相同均为 `STOP_I1`。单 seed 的正面结果只记 `PROMISING_REQUEST_APPROVAL`，后续多 seed/真实采集需用户另行明确批准；未通过则归档终止，禁止自动引入新网络或长训练。

## 4. 总计算与时间纪律

- **最多 4 个新增训练 jobs，累计 ≤6 GPU·h**：阶段 1 A/B ≤2 jobs、≤3 GPU·h；阶段 2 C/D 仅 Gate 通过时 ≤2 jobs，累计不超过 6 GPU·h。达到任何一个上限立即提交决策，不自动继续。
- **不做独立阶段 0 的水波纹/直流/频谱诊断**；也不做 Round7 四个相位模型继续微调或八相位输出后处理。必要 correctness 单测属于阶段 1A。
- 不允许把更多模型、更大网络、增加自由参数或重新划分测试数据作为“修复”后继续烧算力。
- 若不满足 Gate，则先写 STOP/BLOCKED 决策，不为了凑齐四组训练而继续。
- 所有新内容单独版本化，**保留 Round8 `INFERENCE_ARTIFACT_FIXED_METHOD_NO_GAIN` 及前轮负结果**，不重标旧 Gate。

## 5. 给 DeepSeek 的确定性交付

1. **先提交**：修复 `train.py` 的 `role_perm` 传递及对应 tests；`reports/v3_round9/implementation_fidelity.md` 的逐项上游差异矩阵；`reviews/V3_PR1_round9_prereg.md` 写死 split/seed/steps、图像真值、Gaussian/Wiener 参数、GPU 上限。必须明确“已修代码”和“已运行实验”的区别。
2. **再执行** A 与 B；提交 `reports/v3_round9/baseline_shared_crn.md`、数值 JSON、同标度有真值与 SA/HSA 对照图、checkpoint hash、耗时及真实结构变化。**阶段 1 不通过就停止。**
3. 仅 Gate 通过，才提交 `reports/v3_round9/physical_sampler_intervention.md` 和 C/D；否则该文件写 `NOT_RUN` 并说明未触发。
4. **最终**提交 `reports/v3_round9/DECISION.md`：从 `STOP_BASELINE` / `STOP_I1_NO_DISTINCT_INTERVENTION` / `STOP_I1` / `PROMISING_REQUEST_APPROVAL` / `BLOCKED` 中选择符合证据的状态，具体列出 commit、配置、命令、GPU 时间、效果和错误案例。更新 `Project log/Project_progress_status.md` 的 Round9 状态，保留旧 Round8 全部记录。
5. 在 **PR #1** 回复上述文件路径与结果；PR 保持 open，不合并、不改 main。**任何缺失数据或未运行实验都标为 NOT_RUN，不允许“文档通过=实验通过”。**

### DeepSeek 一句话执行指令

> 从阶段 1A 直接开始：先把上游一致的 random Tetris（含 global 角色置换）真正接到 `train.py`，恢复**一个共享 CRN、同-tet 双训练对、完整分辨率滑窗推理**；通过快速单元测试后只训练忠实 VALID random 与最强合法 fixed 两臂，在有真值正控制及 SA/HSA 固定显示上对照 Gaussian/Wiener。**不再做阶段 0 水波纹专项诊断**。若基础链无结构安全收益，立即 STOP_BASELINE；只有有可信信号时再启动最多两臂的正确/错误物理先验采样。训练总计最多 4 jobs、6 GPU·h，不自动追加预算，不合并 PR。
