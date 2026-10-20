# 历史归档索引（截至 2026-10-20）

> **V3 当前计划**：[Project_plan.md](../Project%20log/Project_plan.md) ｜ **V3 当前状态**：[Project_progress_status.md](../Project%20log/Project_progress_status.md)

本目录保存已收束研究路线的原始证据，不因 V3 启动而作废或被重新解释为 V3 已完成数据。

## V1 — SVD-guided Tetris（已归档）

完整内容见 [v1_svd_guided_sampling/README](v1_svd_guided_sampling/Project_plan.md) 及该子目录的原始计划、状态、runs、图和日志。V1 关键教训：SVD-guided 选择器没有在固定 pattern-4 基线之外证明独立贡献；旧输出结构评价需谨慎看待伪真值与高频指标。**没有修改 V1 现有文件。**

## V2 — SVD-CrossFit（本次归档截止快照）

- [v2_svd_crossfit/Project_plan.md](v2_svd_crossfit/Project_plan.md)：V2 **最终版 v0.5 原文完整快照**（2026-10-20 补齐，原为 v0.3）。
- [v2_svd_crossfit/Project_progress_status.md](v2_svd_crossfit/Project_progress_status.md)：V2 截止状态 **原文完整快照**（含 2026-10-19 审计裁决）。
- [v2_svd_crossfit/Project_log_snapshot.md](v2_svd_crossfit/Project_log_snapshot.md)：截至切换时的连续研究日志快照，2426 行（含早期 V1 与 L32–L34）。
- [v2_svd_crossfit/README.md](v2_svd_crossfit/README.md)：V2 闸门与证据索引（含 2026-10-20 补齐说明）。
- [v2_svd_crossfit/reports/](v2_svd_crossfit/reports/)：Phase 0/1/2 **原始 Markdown 报告副本**，共 12 份。
- [v2_svd_crossfit/configs/](v2_svd_crossfit/configs/)：历史 YAML 配置快照。

> **2026-10-20 补齐说明**：首次归档取自 `main@3309af4`，漏掉 V2 在 2026-10-19 的最后一轮
> （指标标定、Phase 2 审计 Gate A–D、Gate N 归因、Gate T 与 SVD 基可辨识性）——那轮成果当时只在
> 分支 `phase2-gate-t-review` 上，未并入 `main`。补齐只增不删，未改写任何一份报告的文字。

仓库现有 `reports/figures/`、`scripts/`、`configs/` 和原始研究日志仍保留原位置，防止历史引用或复现脚本失效。Github 的 git history 继续提供逐提交追溯。实验权重与本地大型产物若未跟踪在 Git 中，本次归档也不会凭空补齐。

**科学状态**：V2 Phase 0/1 通过，Phase 2 未通过验收而中止，Phase 3 未启动。
**注意（2026-10-20 更正）**：首次归档写作”预设的残差相关闸门失败”，该表述已被 V2 自身后续证据取代——
主门槛（残差相关）经标定后**撤回**（它测的是图像相似度，随机切帧同样通过）；替代判据 Gate N 在真实数据上
分辨率仅 0.3–0.4、够不到 0.10 的刻度；Gate T 的主判据在 A/B 二分下代数退化；**根因在采样设计层**：
官方波段占 72.75% 的时间维，1/3 标定集估出的投影子空间两两重叠仅 0.02。
该结论是 V2 特定设计/测量结论，**不能**推广为所有时序 Noise2Noise 方法失效；也**不**意味着官方 SVD 带取错
（V3 用全体积 SVD，不涉及用子集复现基）。细节见 [v2_svd_crossfit/README.md](v2_svd_crossfit/README.md)。

## V3 — 新研究（进行中计划，非历史）

见现行 `Project log/Project_plan.md`、`Project log/Project_progress_status.md` 和 `Project log/V3_experiment_protocol.md`。以最终三维血管成像质量为核心，测量物理空间相关性并约束正交采样；直到完成 Gate 0–4 才能声称新方法有效。
