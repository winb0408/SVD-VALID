# V2 / SVD-CrossFit：停止实验路线归档

**归档日期**：2026-10-20；**补齐日期**：2026-10-20；**原计划截止于**：2026-10-19（最终版 v0.5 + 三轮回填）。
**状态**：V2 Phase 2 未通过验收，Phase 3 未启动，路线中止。归档保留原样，不因研究策略改变而抹去负结果。

> **2026-10-20 补齐说明（为什么改这份文件）**
> 首次归档是在 `main@3309af4` 上做的逐字快照，而 V2 在 2026-10-19 的那轮工作当时只存在于分支
> `phase2-gate-t-review`，尚未并入 `main`，因此首次归档漏掉了 V2 的最后三轮实测：
> 指标标定（残差 proxy 被否证）、Phase 2 审计（Gate A–D）、Gate N 实现与归因、Gate T 与基可辨识性。
> 结果就是本文件原先写的"Phase 2 预设的残差相关闸门失败"并不是 V2 的最终结论——那个闸门后来被
> **撤回**（它测的是图像相似度，不是噪声独立性）。补齐后：计划快照由 v0.3 换成最终 v0.5，日志快照
> 由 1762 行续到 2426 行（补入 L32–L34），阶段状态由 116 行续到 176 行，证据报告由 5 份增至 12 份。
> **只补不删，原有的每一份文件都保留，未改写任何一份报告的文字。**

## 原始文档（逐 blob 复制，不改文字）

- [Project_plan.md](Project_plan.md) — 先前 `Project log/Project_plan.md` 的**最终版 v0.5** 原文（含 v0.5 实现回填、
  标定归因回填、Gate T 回填，以及 §5.6.7–5.6.10、§5.7）。
- [Project_progress_status.md](Project_progress_status.md) — 先前阶段完成/失败/停止结论（含 2026-10-19 审计裁决）。
- [Project_log_snapshot.md](Project_log_snapshot.md) — 截止切换时原项目日志完整快照，2426 行（含 L32 Gate N、L33 标定归因、L34 Gate T）。与切换当刻的 `Project log/Project_log.md` 逐字相同；V3 之后继续追加时，本快照不再跟随。

## 阶段证据

| 报告 | 内容 |
|---|---|
| [Phase0 数据审计](reports/phase0_data_audit.md) | 轴序、哈希、有效掩膜、环境 |
| [Phase1 观测对](reports/phase1_crossfit_pairs.md) | 交叉拟合 SVD 观测对构造 |
| [Phase2 噪声自相关](reports/phase2_noise_autocorrelation.md) | 噪声带时间自相关首测 |
| [Phase2 子空间迁移](reports/phase2_basis_transfer.md) | 子空间在不同帧子集间的迁移性 |
| [Phase2 观测对统计闸门](reports/phase2_pair_statistics.md) | 结构/强度/残差相关三类门槛的主报告 |
| [Phase2 指标标定](reports/phase2_metric_calibration.md) | **残差相关 proxy 的构念效度被否证**（对任意切分都成立，随机切帧同样好） |
| [Phase2 审计 · 代码链路](reports/phase2_audit_codepath.md) | Gate A：环境/SHA、四条数据流、frame 0 三角色 |
| [Phase2 审计 · 裁决](reports/phase2_audit_verdict.md) | Gate D：裁决第 2 项「匹配后结论不确定」 |
| [Gate N · 实现与归因](reports/phase2_gate_n.md) | 体素协方差噪声独立性闸门的落地与首轮标定 |
| [Gate N · 标定失败的归因](reports/phase2_gate_n_calibration_analysis.md) | 增益归一缺陷、k≥4、响应曲线已知答案、**分辨率 0.3–0.4 够不到 0.10** |
| [Gate T · 执行报告](reports/phase2_gate_t.md) | 时间维判据：A0/A1 锚点通过，但主判据在 A/B 二分下代数退化 |
| [SVD 基可辨识性](reports/phase2_basis_identifiability.md) | **根因**：1/3 标定集估出的投影子空间两两重叠仅 0.02（顶部 10 模式为 0.88） |

配置：[crossfit_pairs](configs/crossfit_pairs.yaml)、[phase0_baseline](configs/phase0_baseline.yaml)。

上述报告与 YAML 均按旧仓库 blob 内容复制；原 `reports/`、`configs/` 文件未删除。图仍在
[../../reports/figures/](../../reports/figures/)；脚本仍在 `../../scripts/`（含
`gate_noise_independence.py`、`gate_temporal.py` 与各 `audit_*.py`）；原始结果 JSON 仍在其原目录
`reports/phase2_*_2026-10-19/`。权重和部分数据因未跟踪不包含于归档。**未重跑历史实验。**

```text
# 最终版的定量状态（摘要，细节见上面各报告）
Phase 0  通过        数据审计、基线冻结
Phase 1  通过        交叉拟合观测对构造
Phase 2  未通过      见下
Phase 3  未启动      按 §0.5 未获准
```

**V2 Phase 2 的最终判断（此段替换首次归档的"残差相关闸门失败"表述，原因见上）**

1. **主门槛（残差相关）已撤回。** 标定证明它测的是"两条臂来自不同帧"，不是噪声独立性：
   随机切帧得到同样甚至更好的数值，SA64 匹配后落在随机零分布的第 60–64 百分位。
   因此这条门槛的失败**不能**作为"时间 N2N 不可行"的证据。
2. **替代判据（Gate N，体素协方差）实现了，但在真实数据上分辨率只有 0.3–0.4，够不到 0.10 的刻度。**
   既不能判通过，也不能判不通过。估计量在模型成立时是精确的（合成数据 0.007/0.504/1.000）。
3. **第二替代判据（Gate T，时间维）锚点通过，但主判据在 A/B 二分下代数退化**，
   且单帧核随标定相位在 0.467 / 0.00 / 0.02 之间翻转。
4. **根因在采样设计层，不在判据层。** 官方波段 60:350 在 400 帧上保留 72.75% 的时间维；
   把它按比例映射到 134 帧的标定集（保留 72.4% 的自身维度）后，两个互不相交的标定集估出的
   子空间重叠只有 **0.02**。也就是说"从 C 估基、再对 A/B 施加同一个投影器"这个前提，
   在 1/3 标定率下名义成立而实质不成立。
5. **这个根因不否定 V3。** V3 用全体积 SVD（单一、确定），不涉及用帧子集复现基；
   本节记录的是"不能拿子集去估它"，不是"官方管线的带取错了"。

**重要审阅限制**：不同帧数的均值并不数学上自动产生帧数比的强度系统偏差；不同方法的空间块若被分别过滤应按 block ID 对齐；
残差相关和特制的 Tetris 代理不能被当作直接测得纯噪声相关或等价于原 VALID 的训练。
相关问题留给 V3 Phase 0 重新审计，不改写当时的 V2 报告。

**新方案**：[V3 计划](../../Project%20log/Project_plan.md)。
