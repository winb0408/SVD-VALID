# SVD-VALID Round8：零训练推理对照与原始 VALID 采样忠实度审查

## 当前运行状态

本地已完成**合成模型的伪影机制自检**，并非研究数据的 trained-checkpoint 结果。由于 GitHub 仓库中没有 Round7 `latest.pt` 权重或 `/data/processed/...` 输入体积，不能在当前环境完成 SA64/HSA64 真实权重推理。不得将该自检称为真实数据实验通过。

## 核心方法对照

1. 原始 VALID：每个 2×2×2 块选择一个 4 点 Tetris 构型，四角色抽成四个 (D/2,H/2,W/2) 子视图；两路共享同一 CRN，neighbor loss 加预测一致性和正则；推理完整 patch+重叠平均。四子视图各含原体积 1/8 的体素，合计 1/2。
2. 当前 V3 `valid_random_tetris()`：同样按块选构型，但**每块随机 permute 四角色**；上游 VALID 的 `RandomSampler_0123` 是**一次全局 permute，再对所有块采用该顺序**。所以 `faithful` 声称需修正与测试。
3. 当前 Round7：跨互补四面体 cross-pairing，固定体对角、四次单独训练得 P0–P3 权重；八相位回填用于部署。这不是上游 VALID 的共享网络及整幅 patch 推理路径。该网格伪影不能归因于 Tetris 方法本身。
4. 实验中**不训练**：同一批现有 P0–P3 checkpoint；同一 logeps volume；比较 (A) 八相位回填、(B) 每个 P0–P3 分别完整 3D patch 推理与重叠平均、(C) 四个完整推理结果的均值。必须同时比较输入 SVD。完整推理可能不产生拼接缝，但仍可能因训练目标的固定空间错位而扭曲细血管。
5. 用同一 SVD 输入导出的**固定线性 PD 标定参考**与共同 dB 显示窗口，不能为每列单独重新缩放。固定评价 ROI：亮血管与背景像素都按 SVD 输入选取而非按输出挑选。统计：8 相位的残差均值/方差，Y 向 lag-1 与 lag-2 接缝，亮区 core mean 比，背景 Std 比，输入输出配准。

## 立即执行命令（在原数据所在的实验电脑）

```bash
python scripts/v3_round8_zero_train_full_infer.py --self-test
python scripts/v3_round8_zero_train_full_infer.py --seqs SA64 HSA64 --device cuda --patch 32 64 64 --overlap 0.25
```

前置资产：
- `runs/v3_round7/{SA64,HSA64}_{P0,P1,P2,P3}/latest.pt`，8 个 checkpoint；
- `data/processed/v3_round7_b1/logeps/{SA64,HSA64}_full_x.npy`；
- `reports/v3_round4/b1_norm.json`、`data/processed/v3_round6/{seq}/metadata.json`、`data/processed/official/{seq}/valid_mask.npy`。

若这些文件不存在，返回明确缺失信息、状态 BLOCKED，不自动重训，不生成假结果。

## 成功/止损判定

- **至少**：完整推理无额外奇偶接缝，完整体积输出稳定且不丢有效体素，真实图的血管/背景并行比较。
- **不能**：单凭棋盘消失、全场平均变亮或相位相似度提高就宣布质量提升；需要有真值模拟或谨慎的真实重复观测/结构评价。
- 若棋盘消失但完整推理仍输给固定 Gaussian/Wiener，判为 `INFERENCE_ARTIFACT_FIXED_METHOD_NO_GAIN`，不要继续堆网络。
- 若棋盘仍存在，检查 checkpoint 单模型输出、padding/groupnorm、错位输出和物理轴；如果严重位移或弱血管破坏，则暂停所有 I1 采样优化。
- 即使修好推理，也**没有完成 I1**：实际物理先验选择器尚未端到端控制 TRAINING sampler。后续如继续，必须真正检验正确/错误先验与强固定采样，保持一个共享网络、统一推理。

## 结果交付（DeepSeek）

保留原报告不覆盖，新建 `reports/v3_round8_zero_train/`：每序列 8 个输出的 JSON 指标、固定参考图、训练权重哈希、计算耗时、并排定性图；回复本审查中的所有方法差异，并据结果明确是否止损。整个过程新训练 jobs = 0。