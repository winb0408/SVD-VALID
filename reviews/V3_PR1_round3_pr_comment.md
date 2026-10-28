# PR #1 评论：Round3（A 段 + B1）执行完毕 —— **两段都是负结果**

分支 `v3/judge-acceptance`，提交 `74d96ab → 371e452 → a8b79ae`（已 rebase 到外部复审文档提交 `3b48443` 之上）。
本轮只做了 A 段（只读探针）与 B1（残差旁路三臂 + low 档追加两臂）。**C/I1、D 仍冻结**，
`runs/v3_chain_after_T2.sh` 未运行，**没有合并本 PR**，**没有改动任何阈值**。

---

## 1. 结论先行

| 项 | 结果 |
|---|---|
| A1 误差地板律 | ✅ 成立：14 个格子 0.93% 内吻合 `n2n ≈ (err²+σ²)/(2σ²)` |
| A2 α 混合解析式 | ✅ 修正为 `R(α)=((1−α)²+1)/2`，α=0.5 应为 0.625（旧文写 0.125，交叉项符号错） |
| A5 平凡基线 | ✅ 盲高斯模糊 low6 **0.618**、vlow6 **0.822**（σ* 只在 e900/e901 上选） |
| B1 vlow 三臂 | ⛔ `direct` 11.103 / `residual` 0.851 / `zeroinit` 0.980（门槛 0.80）**全部未过** |
| B1 low 三臂 | ⛔ 1.939 / 1.430 / **0.825**（门槛 0.80）**全部未过** |
| 判读规则 1–4 | 全部执行：规则 1 未通过（不记"部分通过"）；规则 2 有独立贡献但**两档符号相反**；规则 3 未触发；规则 4 **命中一例** |

**一句话**：残差旁路把"网络加噪"（`n2n` 11.10）改成了"网络减噪"（0.851），增益塌陷确实治好了；
但**四个臂没有一个打得过一条盲高斯模糊**（low 档 0.618）。因此本轮只能声明"没再破坏"，**不能声明"学到了去噪"**。

---

## 2. 新增 / 修改文件

**预注册与答复（先写规则、再跑数）**
- `reviews/V3_PR1_round3_amended_plan.md`（新）：修订版预注册，逐条处置 R3-P01–P07。
- `reviews/V3_PR1_round3_external_review_response.md`（新）：P01–P07 逐条答复 + 附 1/2/3 执行结果。
- `reviews/V3_PR1_round2_next_plan.md`（仅顶部加一条指向修订版的说明）：**旧预注册正文原样保留、不改写**。
- `reviews/V3_PR1_round3_pr_comment.md`（新）：本评论的存档副本。

**代码**
- `models/network.py`、`train.py`、`infer.py`：`--arch-head {direct,residual}`、`--zero-init-final`；
  残差旁路是模块级 bypass，**不改 state_dict 键名**，`infer.py` 从 checkpoint 的 `arch_head` 重建。
- `scripts/v3_round3_a_probes.py`、`v3_round3_a5_trivial_baselines.py`、`v3_round3_b1_collect.sh`、
  `v3_round3_b1_low_chain.sh`、`v3_round3_b1_summary.py`、`v3_round3_figure.py`（新）。
- `scripts/v3_round2_r0_audit.py`、`v3_round2_r0_gate.py`、`v3_expA_v2_eval.py`（改）：R0 判据 v2。
- 判据版本化：`G-R0-b` 标 historical（它奖励"测得出斜率"，与同方差化目标相反；log 分支口径不一致），
  新增 `G-R0-b2`（|log 斜率| ≤ 0.20 且 log 域方差跨箱 max/min ≤ 2.0）与 `G-R0-d`（共用参考伪像核查）。
  **旧数值整份存档**为 `reports/v3_round2/*_v1_2026-10-27.json`，逐位对账可用。

**测试**：`tests/test_v3_round3_*.py` 7 个文件，**66 passed**（全仓 149 passed），覆盖共同参考 1/3、
双向 log、α 解析式 0.625、dense 恒等、几何编码、桥容差、残差头 state_dict 兼容。

**证据**
- `reports/v3_round3/r0_noise_independence_v2.md`、`r0_log_variance_v2.md`、`geometry_and_loss_gradient_audit.md`、
  `a_probes.md`、`b_minimal_repair.md`（新）。
- 原始 JSON：`a_probes_{vlow6,low,vlow}_*.json`、`a5_trivial_baselines.json`、
  `b1_{direct,residual,zeroinit}{,_low}_probe.json`、`b1_summary.json`、`b1_summary_low.json`。
- 图：`fig_round3_{low,vlow}_inDist.png`（A 段）、`fig_round3_b1_{low_zeroinit,vlow_residual}.png`（B1 段）。

---

## 3. 复现命令

```bash
# 0) 环境
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
cd /ssd1/wjb/SVD-VALID

# 1) A 段只读探针（不改权重）
$PY -u scripts/v3_round3_a_probes.py --level vlow --arm S_dense --seed 3407 \
  --runs runs/v3_round2_s0/vlow --gpu 0 \
  --manifest data/processed/v3_round2_s0/vlow6/manifest.json \
  --out reports/v3_round3/a_probes_vlow6.json
$PY -u scripts/v3_round3_a5_trivial_baselines.py        # 盲模糊/中值基线，σ* 只在 e900/e901 上选

# 2) B1 四臂训练（同一套超参，只换输出头）
D=data/processed/v3_round2_s0/vlow
VOLS=$(ls $D/train/*_log1p.npy | tr '\n' ' ')
CLEANS=$(ls $D/train/*_clean.npy | tr '\n' ' ')
COMMON="--patch 32 64 64 --epochs 40 --steps-per-epoch 500 --features 16 --seed 3407 \
        --lr-schedule poly --pairing within --mode dense \
        --train-volumes $VOLS --train-clean-volumes $CLEANS"
CUDA_VISIBLE_DEVICES=0 $PY -u train.py $COMMON --out runs/v3_round3/b1_vlow/B1_residual_s3407 \
  --arch-head residual
CUDA_VISIBLE_DEVICES=1 $PY -u train.py $COMMON --out runs/v3_round3/b1_vlow/B1_zeroinit_s3407 \
  --arch-head residual --zero-init-final
# low 档两臂：bash scripts/v3_round3_b1_low_chain.sh（同 COMMON，D 换成 .../low）

# 3) 采集六份探针 + 两份汇总（本次实际执行的一条命令）
nohup bash scripts/v3_round3_b1_collect.sh    # PID 3933931，日志 runs/v3_round3/b1_collect.log

# 4) 出图
$PY scripts/v3_round3_figure.py --level-dir data/processed/v3_round2_s0/low6 \
  --ckpt runs/v3_round3/b1_low/B1_low_zeroinit_s3407 --sigma 0.6 --scenes e900 e903 \
  --out reports/v3_round3/fig_round3_b1_low_zeroinit.png --gpu 0
```

**种子与切分**：训练种子一律 `s3407`；训练集 12 卷；评估集一律 `e900–e905`（六卷独立场景）；
评估档 manifest `data/processed/v3_round2_s0/{vlow6,low6}/manifest.json` **未变**。
权重一律取训练 40 epoch 结束后的 `latest.pt`（`history.json` 落盘后才评测，**未评测任何中途 checkpoint**）。

**哈希（sha256）**

```
9e1f27a0…eb90192e  reports/v3_round3/b1_summary.json
27d50460…7f7dc574  reports/v3_round3/b1_summary_low.json
dcc08bfe…7269b6d  reports/v3_round3/a5_trivial_baselines.json
0f55d8c9…ff3e272b  b1_direct_probe.json        e55f8040…ce8779d6  b1_residual_low_probe.json
f1a6cc3c…a1bf5d25  b1_residual_probe.json      d2b1b760…fdac5024  b1_zeroinit_low_probe.json
fa644069…d3cb279f4 b1_zeroinit_probe.json      2c4f6e63…06d3e9a8686 b1_direct_low_probe.json
9e110f9b…e5b5e74526 fig_round3_b1_low_zeroinit.png
186fd710…a3f937d6659 fig_round3_b1_vlow_residual.png
```

---

## 4. B1 数字（`n2n` = mse(pred,obs2)/mse(obs1,obs2)，1 = 不动）

| 档 | 臂 | `n2n` 均值 | 场景区间 | `r_roi`↑ | 最优混合（α* 中位） | 隐含 err/σ | 过 0.80 |
|---|---|---|---|---|---|---|---|
| vlow6 | `direct`（旧基线） | 11.103 | 4.970–30.298 | 0/6 | 0.973（0.05） | 4.61σ | 否 |
| vlow6 | `residual` | **0.851** | 0.795–0.915 | 6/6 | 0.779（0.62） | 0.838σ | 否 |
| vlow6 | `zeroinit` | 0.980 | 0.893–1.047 | 5/6 | 0.789（0.50） | 0.980σ | 否 |
| vlow6 | 盲高斯模糊 σ=0.5 | **0.822** | 0.803–0.848 | — | — | 0.803σ | 否 |
| low6 | `direct` | 1.939 | 1.171–4.237 | 4/6 | 0.854（0.30） | 1.696σ | 否 |
| low6 | `residual` | 1.430 | 1.104–1.830 | 3/6 | 0.840（0.35） | 1.364σ | 否 |
| low6 | `zeroinit` | **0.825** | 0.777–0.858 | 6/6 | **0.721**（0.60） | 0.806σ | 否（差 3%） |
| low6 | 盲高斯模糊 σ=0.6 | **0.618** | 0.607–0.628 | — | — | 0.486σ | **是** |
| low6 | 中值 3×3 | 0.853 | — | — | — | 0.833σ | 否 |

`err/σ = sqrt(2·n2n − 1)`；过 0.80 需 `err ≤ 0.775σ`。

---

## 5. 遗留问题（按优先级）

1. **为什么一条盲高斯模糊能赢？** low 档 0.618 vs 最好网络 0.825（误差 0.486σ vs 0.806σ，差 66%）。
   在讲清这件事之前，任何新模块的效果都无法归因到本项目方法本身——这也是本轮之后**暂停追加架构复杂度**的理由
   （审查 §6 的中止条件）。
2. **`vlow6` 的 0.80 门槛本身不具分辨力**：连盲模糊都只到 0.822。该档以后只应作"是否 ≤ 0.822"的对照，
   或直接改评测档位；**本轮不改门**（不在结果出来后调阈值）。
3. **`r_roi` 子判据几乎没有判别力**：输入臂读数 low 0.9947–0.9978 / vlow 0.9994–0.9998，天花板距离 0.002–0.005，
   而三臂增益仅 +1e-4～+4e-3；连 `n2n`=1.939 的 `direct` 在 low6 都拿到 4/6"上升"。建议下一版预注册替换此条，
   **本轮未改**。
4. **初始化效应在两档方向相反**（vlow `zeroinit−residual` = +0.128 [+0.104,+0.153]；low = −0.605 [−0.847,−0.362]）。
   "零初始化是关键修复"这句话不能笼统说。
5. **规则 4 命中**：vlow `zeroinit` 训练损失 5.66e-5 → 1.34e-5（降为 1/4），eval `n2n` 却 0.980。
   监督 MSE 的主要下降在信号区，而 `n2n` 的地板由纯背景噪声决定。
6. **检查点不可混用**：A5 里 low6 的"1.697"来自 vlow 训练的权重（跨档探针），B1 `direct` low 的 1.939 才是
   low 训练的权重；两者不得混引。
7. **未运行**：A1 带 B 臂的重跑、log 表征小试、C/I1、D、以及 `runs/v3_chain_after_T2.sh`。

---

## 6. 通俗总结

这一步做的事很简单：我们怀疑网络"越练越糟"是因为它输出的是"整张图"，而不是"噪声"，所以换了个接法试两种，
加上下游低噪声档各补一臂，一共四个版本。

结果分两层，要分开说。**好消息是病治好了**：以前网络输出会把输入弄坏十倍（数值 11 倍于"什么都不做"），
换接法后降到 0.85，也就是虽然几乎没变好，但至少不再把图弄坏——网络确实在减噪，不是在加噪。
**坏消息是没有竞争力**：我们同时让一条最朴素的"糊一下"（固定程度的高斯平滑）做同样的题，它在低噪声档拿了 0.618，
比我们四个版本里最好的 0.825 还好三成左右。数值越低越接近"干净答案"，1.0 等于"原样不动"，所以 0.618 好于 0.825。

因此这一轮**不能**说"我们的方法学到了去噪"——只能说"我们的网络不再破坏数据"。也不能说"某个新模块有效"，
因为我们没有赢过一个不需要训练的对照。还有一句必须提醒：高噪声那一档设的合格线（0.80）连那条盲平滑都过不了
（它 0.822），说明那条线在这一档上没有区分能力，不能拿它当"我们差一点就过了"的证据。

下一步按事先约定该做的不是再加模块，而是停下来回答一个具体问题：**为什么一次不学习的平滑，比训练四十轮的卷积网络还好？**
这个问题答不出来，后面加什么都会被同样一句反问推翻。
