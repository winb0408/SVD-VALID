# 历史归档索引（截至 2026-10-20）

> **V3 当前计划**：[Project_plan.md](../Project%20log/Project_plan.md) ｜ **V3 当前状态**：[Project_progress_status.md](../Project%20log/Project_progress_status.md)

本目录保存已收束研究路线的原始证据，不因 V3 启动而作废或被重新解释为 V3 已完成数据。

## V1 — SVD-guided Tetris（已归档）

完整内容见 [v1_svd_guided_sampling/README](v1_svd_guided_sampling/Project_plan.md) 及该子目录的原始计划、状态、runs、图和日志。V1 关键教训：SVD-guided 选择器没有在固定 pattern-4 基线之外证明独立贡献；旧输出结构评价需谨慎看待伪真值与高频指标。**没有修改 V1 现有文件。**

## V2 — SVD-CrossFit（本次归档截止快照）

- [v2_svd_crossfit/Project_plan.md](v2_svd_crossfit/Project_plan.md)：V2 v0.3 **原文完整快照**。
- [v2_svd_crossfit/Project_progress_status.md](v2_svd_crossfit/Project_progress_status.md)：V2 截止状态 **原文完整快照**。
- [v2_svd_crossfit/Project_log_snapshot.md](v2_svd_crossfit/Project_log_snapshot.md)：截至切换时的连续研究日志快照（含早期 V1）。
- [v2_svd_crossfit/README.md](v2_svd_crossfit/README.md)：V2 闸门与证据索引。
- [v2_svd_crossfit/reports/](v2_svd_crossfit/reports/)：Phase 0/1/2 **原始 Markdown 报告副本**。
- [v2_svd_crossfit/configs/](v2_svd_crossfit/configs/)：历史 YAML 配置快照。

仓库现有 `reports/figures/`、`scripts/`、`configs/` 和原始研究日志仍保留原位置，防止历史引用或复现脚本失效。Github 的 git history 继续提供逐提交追溯。实验权重与本地大型产物若未跟踪在 Git 中，本次归档也不会凭空补齐。

**科学状态**：V2 Phase 0/1 通过，Phase 2 预设的残差相关闸门失败而停止，Phase 3 未启动。该“失败”是 V2 特定测量/协议结论，不能推广为所有时序 Noise2Noise 方法失效。

## V3 — 新研究（进行中计划，非历史）

见现行 `Project log/Project_plan.md`、`Project log/Project_progress_status.md` 和 `Project log/V3_experiment_protocol.md`。以最终三维血管成像质量为核心，测量物理空间相关性并约束正交采样；直到完成 Gate 0–4 才能声称新方法有效。
