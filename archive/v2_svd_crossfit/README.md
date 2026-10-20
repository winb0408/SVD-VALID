# V2 / SVD-CrossFit：停止实验路线归档

**归档日期**：2026-10-20；**原计划截止于**：2026-10-16；**状态**：V2 Phase 2 统计闸门不通过，按原计划中止后续 N2N/混合网络训练。归档保留原样，不因研究策略改变而抹去负结果。

## 原始文档（逐 blob 复制，不改文字）
- [Project_plan.md](Project_plan.md) — 先前 `Project log/Project_plan.md` 的原文。
- [Project_progress_status.md](Project_progress_status.md) — 先前阶段完成/失败/停止结论。
- [Project_log_snapshot.md](Project_log_snapshot.md) — 截止切换时原项目日志完整快照（包含 V1/V2 叙事）。

## 阶段证据
- [Phase0 数据审计](reports/phase0_data_audit.md)
- [Phase1 观测对](reports/phase1_crossfit_pairs.md)
- [Phase2 噪声自相关](reports/phase2_noise_autocorrelation.md)
- [Phase2 子空间迁移](reports/phase2_basis_transfer.md)
- [Phase2 观测对统计闸门](reports/phase2_pair_statistics.md)
- [配置 crossfit_pairs](configs/crossfit_pairs.yaml)
- [配置 phase0_baseline](configs/phase0_baseline.yaml)

上述报告与 YAML 均按旧仓库 blob 内容复制；原 `reports/`、`configs/` 文件未删除。图仍在 [../../reports/figures/](../../reports/figures/)；脚本仍在 `../../scripts/`，权重和部分数据因未跟踪不包含于本次归档。**未重跑历史实验**。

**重要审阅限制**：不同帧数的均值并不数学上自动产生帧数比的强度系统偏差；不同方法的空间块若被分别过滤应按 block ID 对齐；残差相关和特制的 Tetris 代理不能被当作直接测得纯噪声相关或等价于原 VALID 的训练。相关问题留给 V3 Phase0 重新审计，不改写当时的 V2 报告。

**新方案**：[V3 计划](../../Project%20log/Project_plan.md)。
