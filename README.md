# SVD-VALID

**Current research direction (V3.0, planned): PSF- and Speckle-Correlation-Aware Orthogonal Self-Supervised Denoising for 3D Ultrasound Microvascular Imaging.**

**中文简介**：项目以最终三维微血管图像恢复质量为首要目标；拟用独立测量的 PSF、散斑/残余噪声空间相关性指导 VALID 三维正交采样，并在不同时修改 CRN 网络和主要损失函数的条件下，对照 faithful random Tetris、fixed-4 与最强固定方向，检验背景抑制与微血管结构保真能否得到真实改善。

> **Status / 研究状态：** V3 研究方案已建立，Phase 0 仍待执行；**尚未实现或验证新采样算法**。当前 `train.py`、`infer.py`、`datasets/sampling.py` 等仍是历史 V1 的可用代码基础；不要将当前命令行选项视作 V3 实现。

## Project documents / 项目文档

- **[V3 detailed plan](Project%20log/Project_plan.md)** — 目标、假设、创新、分阶段任务、验收门槛、论文实验。
- **[V3 progress tracker](Project%20log/Project_progress_status.md)** — 当前真实状态、Gate 与待办。
- **[V3 experiment protocol](Project%20log/V3_experiment_protocol.md)** — 固定数据/网络条件、矩阵、评价、图表及负控制。
- **[Research log](Project%20log/Project_log.md)** — V1/V2 逐次历史实验日志；旧内容保留，今后追加 V3 日志。
- **[Historical archive](archive/README.md)** — V1、V2 的项目计划、状态、V2 原始报告/配置快照。

## Research pipeline (planned / 待实现)

```text
Complex IQ data
  -> validated SVD clutter filter -> Power Doppler (fixed representation)
  -> separately calibrated 3D PSF / speckle & residual-noise covariance
  -> VALID-compatible orthogonal sampling (pattern / role / distance)
  -> same VALID CRN and loss
  -> full 3D denoised volume
  -> independent quality evaluation (noise + weak vessels + connectivity)
```

**Primary scientific test:** exceed both faithful VALID random and the strongest reasonable **fixed sampling geometry**, without sacrificing weak vessels, branches or width. The primary test must use reliable truth/reference and non-leaking held-out acquisitions; metrics such as high-frequency suppression alone are insufficient.

## Historical lines

- **V1** SVD-guided Tetris: the SVD selector did not demonstrate improvement independent of fixed pattern 4. Historical code is retained.
- **V2** SVD-CrossFit: Phase 0/1 passed, Phase 2 predeclared residual-statistical gate failed; no Phase 3 model training was initiated. Original documents and reports are archived rather than discarded.

## Reproducibility and existing code

Original implementation references Gu et al., *Science Advances* (2026), VALID, and related self-supervised ultrasound methods. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [docs/datasets.md](docs/datasets.md) and historical configs/scripts. Existing CLI options belong to previous research phases; **there is not yet a V3 quick-start command**.

Install dependencies from `requirements.txt` if using the historical implementation. Data and weights not committed to GitHub must be supplied separately. Maintain the original sample/acquisition IDs, the valid spatial mask, SVD parameters and physical voxel spacing when running new experiments.

## Decision discipline

Phase 0 baseline/evaluation -> Phase 1 physical measurement -> Phase 2 sampling prediction -> Phase 3 minimal training -> Phase 4 true image-quality evidence -> Phase 5 generalization -> Phase 6 paper. **Do not advance through a failed gate by increasing algorithm complexity or by changing the held-out test selection.**
