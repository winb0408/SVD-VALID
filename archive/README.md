# 归档区

本目录保存**已收束项目线 v1「SVD 引导采样」(SVD-guided sampling)** 的产物。
该线于 2026-10-10 判定创新点 1 的架构前提不成立(见 `Project log/Project_log.md` 的 L23/L24),
主线转入 CrossFit 方案(见 `Project log/Project_plan.md`)。

**这里的文件不是废品**:它们是那份否定结论的审计轨迹,正在被外部复审。
权重、中间产物仍留在磁盘上(未入库),但**不建议删除** —— 整个 v1 run 树只有 201 MB,
而 `/ssd1` 尚有 1.2 TB 可用,删除换不到空间,只会毁掉可复核性。
外部复审结束后如需清理,按下表"状态"列执行即可。

---

## run → 结论 映射表

状态取值:**复用**(新方案直接用的基线) | **证据**(支撑已记录结论,保留) | **中间**(被后续实验取代,仅过程留档) | **废弃**(管线错误,结论已作废)

| run / 产物 | 内容 | 支撑的结论 | 状态 |
|---|---|---|---|
| `A_sa64_full_random` | 全带(rank 0–400)随机 Tetris | 无(秩带错误) | 废弃 |
| `B_sa64_svd_random` | rank 10–80 随机 | 无(秩带错误) | 废弃 |
| `C_sa64_svd_guided` | rank 10–80 引导 | 无(秩带错误,见 L9「A/B/C 结论作废」) | 废弃 |
| `B2_sa64_blood_random` | 官方带 60–350 + log,随机 | L20 前身 | 中间 |
| `C2_sa64_blood_guided` | 官方带 60–350 + log,引导 | L20 前身 | 中间 |
| **`B3_sa64_remap_random`** | 官方带 60–350 + remap,**faithful random** | VALID 原版基线 | **复用**(新方案 M1) |
| `C3_sa64_remap_guided` | 官方带 + remap,引导 | L20「C3 优势来自 SVD 指导」→ 后被 P3 推翻 | 证据 |
| `B3p_sa64_remap_patchrandom` | patch 级随机控制组 | L20 归因三重奏 | 证据 |
| `B3h0_remap_random` / `C3h0_remap_guided` | Hessian 正则置零消融 | `figure_denoised_hessian_ablation.png` | 证据 |
| `B3id0_remap_random` / `C3id0_remap_guided` | identity 损失置零消融 | L23 消融 | 证据 |
| **`B4_hsa64_random`** / **`C4_hsa64_guided`** | HSA64 复现 | 跨序列复现 | **复用**(新方案 HSA64 复现) |
| `SA64e_random` / `SA64e_patchrandom` / `SA64e_guided` | 仅偶帧(200 帧)子采样 | 体素数稳健性 | 证据 |
| `F0`–`F3`,`F5`–`F7` `_sa64_remap_fixed` | 8 模式全扫描(其余 7 个模式) | **P3:逐模式去噪强度离散** | 证据 |
| **`F4_sa64_remap_fixed`** | 固定 pattern 4 | 唯一真去噪的模式 | **复用**(新方案 M2) |
| `F4r1`–`F4r4_sa64_role` | 角色置换(输入对配对) | **P3:位移方向是唯一有内部对照的变量** | 证据 |
| `eval_structure_*.json`(6 个) | core / pseudoGT / keep / `hf_noise_reduction` | 全部结构指标 | 证据 |
| `phase3/`,`phase3_hsa64/` | ρ_signal / ρ_noise 分带统计 | **P3:判据近乎常数** | 证据 |
| `pattern_scan/pattern_scan.json` | 描述子-质量相关性 | **P3:判据无预测力且符号相反** | 证据 |
| `selector_crosscheck/` | 偶/奇帧选择器独立核验 | **P3:决定性否定(一致率 0.438)** | 证据 |
| `official_mip/figure_*.png` | 同尺度对照图、扫描总览图 | 全部可视化证据 | 证据 |
| `official_mip/*.npy` | 全序列官方 PD 参考(dB、偶/奇帧) | Phase 0 参考体 | 中间(新方案会重生成) |
| `*.log`(28 个) | 训练日志(逐 epoch loss) | 训练可复核性 | 证据 |

## 文档

| 文件 | 内容 |
|---|---|
| `Project_plan.md` | v1 项目计划(2026-10-06 创建,止于 P3 修订) |
| `Project_progress_status.md` | v1 阶段验收状态(更新至 2026-10-09) |

v1 的完整历史记录在 `Project log/Project_log.md`(L1–L24),该文件继续沿用、不归档。
