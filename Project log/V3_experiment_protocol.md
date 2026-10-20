# V3.0 实验执行规范、对照矩阵与论文图表

> 版本 2026-10-20 ｜ 附属于 [Project_plan.md](Project_plan.md)；**所有参数与阈值均为预设建议，不代表结果**。正式实验前在配置中冻结，不得在测试集上后验挑选。

## A. 输入和复现契约

1. 统一数据记录：acquisition_id、sample_id(可用时)、空间 axis name/spacing(物理单位)、序列/帧、valid_mask、源文件 SHA256、SVD 秩带、输入物理域(linear PD/log PD/remap)、强度参数。
2. SA64/HSA64 官方 SVD MATLAB 1-based rank 60:350 = Python `[59,350)`，核对实现与公开脚本；`official_pipeline.py` 之前只与仓库 V1 实现高相关比对，不能叫 MATLAB 全等。
3. 固定 calibration/train/validation/test；按独立 acquisition/subject 分组并报告各组个数。两个独立序列不意味着 19 个独立样本。
4. 原始 `datasets/sampling.py` 与训练/推理按 faithful VALID 核对。固定八模式时保存每个 subvolume 的 2×2×2 位置、角色分配、输入—目标三维物理位移，建立 exhaustive 小数组测试。
5. 训练方法只改变采样机制；optimizer、loss 及其权重、网络、训练步数、batch/patch、数据表示和验证标准固定。若不同采样计算成本有别，报告 step 与 FLOPs/前向次数。
6. 所有训练和评估保存 commit SHA、配置、seed、模型权重、raw float output、完整评价数值、绘图脚本。多模型对比必须共享 color limits 和 ROI，禁止单独裁剪/重拉伸。

## B. PSF/噪声统计的区别

- **PSF**：由点目标、标定体模或已知仿真成像核估计；不将血管纹理的自相关当作实测 PSF。
- **Speckle**：相干成像导致的强度/纹理相关，可能包含真实组织/血流结构。
- **Residual noise**：SVD 后残留的成分，来源可能是噪声、残余杂波、运动及真实微细血管；只能在明确假设下做代理。
- **ACF**：在各物理轴和对角位移下测定，注意有效 mask、均值移除、不同深度、采样尺度、局部统计及不确定性。不同代理作敏感性分析。
- **N(k)**：尽量来自独立标定或真值仿真中的噪声-噪声视图依赖；**不**直接使用总体像素相关性等同噪声相关性。
- **S(k)**：已知结构参考/体模/模拟的视图信号保真；对细血管、弱血管提高权重，避免只保留亮血管核心。
- 候选选择规则先采用 `argmin N(k)` subject to `S(k)>=S_min` 和几何合法；预注册 `S_min`，验证中检验对阈值变化的稳定性。

## C. 实验矩阵

| ID | 配置 | 唯一科学问题 | 必要控制 |
|---|---|---|---|
| E0 | 原始输入/传统平滑 | 是否优于不处理或直接模糊 | 相同显示与物理域 |
| E1 | faithful VALID random | 基线质量 | 原 VALID 随 block 采样+角色随机 |
| E2 | fixed-4 | V1 提示的关键固定基线 | 相同 seed/budget |
| E3 | 八 fixed patterns | 其它采样模式差异 | 同一网络+训练协议 |
| E4 | 相同 pattern 的角色置换 | 位移角色是否关键 | 保持位置集合不变 |
| E5 | 物理固定方向 | 最强可解释固定策略 | 与结合方法公平比较 |
| E6 | PSF-only | PSF 的独立作用 | 不用噪声先验 |
| E7 | correlation-only | 噪声相关先验的独立作用 | 不用 PSF |
| E8 | PSF+correlation | 核心新方法 | E1/E2/E5/E6/E7 |
| E9 | 错误/随机先验 | 物理因果归因 | 相同训练成本 |
| E10 | 全局 vs 局部选择（可选） | 自适应有无额外收益 | 先通过物理异质性 Gate |
| E11 | 深度/SNR/PSF/血管宽度变化 | 适用范围 | 物理真值仿真和真实采集 |
| E12 | SA64→HSA64/独立采集 | 泛化 | 严格采集级拆分 |

优先执行 E1-E4，再 E6-E9；局部 E10 不在初版投入。

## D. 主要终点与安全约束

**主终点（建议）**：在预先确定的假阳性率 FPR 下，真值体积中弱血管的 Recall；必须预注册血管标记方式、FPR、弱血管阈值和真实评估单位。如体模/仿真不支持该指标，则在 Phase 0 中改为明确且有真值支持的唯一主要终点（例如中心线 F1），正式 Phase 4 以后不允许更换。

**补充**：PSNR/SSIM、血管体素 F1、分支连续性、宽度误差、体素残差、背景噪声；真实图像以独立参考/重复采集/固定 ROI 作为有限证据。不以高频减少、归一化相关性或全帧 PD 伪真值作为独立真值。

**结构安全**：相同或更低 FPR 条件下不能显著降低血管召回、连接和细分支；不得增加伪血管，宽度误差不得因过平滑明显恶化。

**建议挑战值（非既成成绩）**：相对最强固定采样 PSNR 提升约 +0.5 dB 或弱血管 Recall +3 个百分点；宽度中位误差恶化不超过参考宽度的约 10%；这些是附加目标，真正验收需主终点独立增益+结构安全，按采集/独立仿真实例做置信区间，核心至少 3 个 seed。

**Pareto 分析**：画出噪声抑制与结构保真的前沿；同等背景抑制比较召回/宽度，同等结构保真比较噪声，不得通过单一“平滑越多越好”的打分决定胜者。

## E. 物理与统计严谨性检查

- 不等数量的帧分别取**均值**，只要同分布预期均值相同；不同帧数主要造成方差和波动差异，而不是必然的 `T_B/T_A` 固定强度偏差。只有帧**求和**才天然按数量缩放。
- 空间分块计算时若各组分别丢弃低有效率/零方差块，配对比较必须用 block ID 取交集。
- patch、空间相邻 block、时间重叠折、同一 acquisition 派生的样本均不得默认为统计独立。Bootstrap 按真正独立采集或采用合适的层次方法。
- Tetris 的 4 相位子视图 != 先平均再复写回 full grid 的 block-smooth 对照，不能作为严格等价的 VALID 去噪实验。
- 防止泄漏：任何受测试图像影响的模式筛选、ROI 选择、阈值拟合都要从 held-out 测试之前完成并冻结。

## F. 每阶段执行清单/产物与闸门

| Phase | 需要提交的产物 | Gate |
|---|---|---|
| 0 | `v3_phase0_baseline.md`、基线/评估测试、分组协议 | 可复现且不泄漏 |
| 1 | `v3_phase1_physics.md`、3 方向/深度 PSF/ACF 图 | 相关性代理可靠，提出可检验差异 |
| 2 | `v3_phase2_sampling.md`、模式几何/统计选择与负控制 | 独立模式预测可靠、信号约束达标 |
| 3 | `v3_phase3_training.md`、训练日志、raw outputs | 正向验证趋势且不伤结构 |
| 4 | `v3_phase4_image_quality.md`、统计主表、真值/真实图和消融 | 超过最佳固定采样，多 seed 与结构安全 |
| 5 | `v3_phase5_generalization.md`、参数敏感性/独立采集 | 明确跨域效果与失败边界 |
| 6 | `v3_phase6_paper_map.md`、主/补图目录、复现清单 | 论断逐一与证据对应 |

阶段报告统一包括：配置/样本数→数值结果及误差→与预设 Gate 对照→可能替代解释→**通俗总结**→进入下阶段/停止判定。

## G. 预期论文图（待实测，不预建假数据）

- **Fig.1** 方法总览：原 VALID 随机采样、PSF/残余散斑方向相关、物理约束采样、CRN 和重建。
- **Fig.2** PSF 和空间相关性：三方向 FWHM、ACF、深度/位移及置信区间。
- **Fig.3** 模式几何/预测：八模式、角色/位移、噪声相关与信号约束、预测对照、错误先验负控制。
- **Fig.4** 有真值恢复：仿真/体模 GT + 输入/random/fixed-4/new；血管召回@FPR、PSNR/SSIM、管径和连接。
- **Fig.5** 真实三维恢复：相同窗口 MIP、三切片、弱血管 ROI、宽度/背景/假阳性。
- **Fig.6** 消融与泛化：PSF-only/noise-only/combined、各 seed、跨深度与独立采集、失败案例。

## H. 建议配置字段（尚未创建可运行 YAML）

```yaml
project_version: v3
data:
  representation: log_pd  # only after Phase0 freezes the choice
  valid_mask: required
  axis_spacing_mm: null   # MUST be measured, not guessed
splits:
  group_by: acquisition_id
  test_ids: []             # freeze before evaluating
sampling:
  candidate_set: valid_8_patterns
  mode: faithful_random     # fixed4 | psf_only | corr_only | combined
  signal_floor: null       # pre-register in calibrated metric units
model:
  backbone: valid_crn
  loss: original_valid
evaluation:
  primary_endpoint: weak_vessel_recall_at_fixed_fpr
  seeds: [11, 23, 47]
  shared_display_limits: required
```

此文件是实验契约与开发路线，不是可直接运行的配置；`null` 和 `required` 项必须由 Phase 0 明确后才能训练。
