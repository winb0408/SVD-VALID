# PR #1 评论：Round3（A 段 + B1）执行完毕 —— **两段都是负结果**

分支 `v3/judge-acceptance`，提交 `74d96ab → 371e452 → a8b79ae`（已 rebase 到外部复审文档提交 `3b48443` 之上）。
本轮只做了 A 段（只读探针）与 B1（残差旁路三臂 + low 档追加两臂）。**C/I1、D 仍冻结**，
`runs/v3_chain_after_T2.sh` 未运行，**没有合并本 PR**，**没有改动任何阈值**。

> **2026-10-29 追加（第二次推送，§7）**：本文之后又追加了一节 `§7. 真实数据阶段的解锁条件`，
> 记在 [V3_PR1_round3_amended_plan.md](V3_PR1_round3_amended_plan.md) §7A。**没有新实验、没有改阈值**。
>
> **2026-10-29 追加（第三次推送，§8）**：`§8. 真实数据零样本评估已执行`——用 B1 四臂在真实 SA64/HSA64 上
> 做了一次**不训练**的零样本评估（提交 `7eaadfa`）。判据在跑之前写死：`n2n ≤ ①Wiener` 且 `core ≥ 0.95`；
> 结果 **SA64 0/4、HSA64 0/4**。§7 条件 ④ 已完成，条件 ①②③ 仍未满足，真实数据训练仍不启动。
>
> **2026-10-29 追加（第四次推送，§9）**：`§9. Round4-0（P0，零训练）已执行`——按 Round4 计划先重建结构尺子
> （v1/v2 失败留档、v3 校验通过），再做公平 baseline 与 Pareto，最后在**未见过的 e906–e911** 上跑预注册确认。
> **vlow 档 β=0.5 的 net_residual / net_zeroinit 确认通过**（对 gauss 的 `|bias_ves|` 6/6 更小、`var_bg` 5/6 不劣）；
> **low 档三臂 `NO_FEASIBLE_BETA`，旧 Gate 结论不变**。零训练，GPU·h ≈ 0.15。

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

---

## 7. 真实数据阶段的解锁条件（2026-10-29 补记，写于开跑前）

**为什么加这一节**：用户问"到底什么时候能进入下一阶段真实数据的训练"，并且要一张
input / our method / GT 的三联图来判断方法有没有用。这两问都需要落到可核验的条件上，
且必须在看到真实数据训练结果**之前**写死，否则事后一定按结果倒推理由。

**硬条件（全部满足才启动；不满足就写"未满足"）**，完整版见
[V3_PR1_round3_amended_plan.md](V3_PR1_round3_amended_plan.md) §7A：

1. low6 上有一臂 `n2n ≤ 0.80` **且** 6/6 场景优于输入臂 **且击败同档盲高斯模糊的 0.618**；
2. 同一改动在第二个独立档位（vlow6/self）方向一致（不要求同幅度）；
3. 有一个与 `n2n` 正交的机制量能解释"它为什么比盲模糊强"，否则按 `G-S0-2` 不做数；
4. 先有一次**不训练**的真实数据零样本评估，记录全秩链退化标记与 dark 区保留倍数。

**当前状态：①未满足**（vlow 0.851/0.980/11.103；low 0.825/1.430/1.939）。故真实数据训练不启动。

**一条不可绕过的事实**：SA64/HSA64 **没有**干净微血管 GT（phase4 §8.1：全秩链 n2n 退化 241.8×/5180.4×）。
所以"input / our method / **GT**"三联图**只在模拟数据上成立**（Round3 四张图）；
真实数据上只能出"input / 我们的方法 / 独立参考（奇偶帧另一支、或 SVD+①Wiener）"，且必须标"无 GT"。
已有的真实数据可视化：`reports/figures/v3_real_network_{SA64,HSA64}.png`（零样本 4 seed，**0/4 通过**，
dark 区注入结构 2.7–7.3 倍）、`v3_real_denoise_{SA64,HSA64}.png`（全秩/SVD/SVD+Wiener 三链）。

**本次改动文件**：`reviews/V3_PR1_round3_amended_plan.md`（新增 §7A）、
`Project log/Project_progress_status.md`（Round3 补记）、本文件（§7 与顶部指针）。
**未运行**：任何训练、任何新探针；C/I1、D、`runs/v3_chain_after_T2.sh` 仍未动。

---

## 8. 真实数据零样本评估已执行（2026-10-29 追加，无训练）

**为什么加这一节**：§7 的条件 ④ 要求"先有一次不训练的真实数据零样本评估"。用户选择了现在就做，
所以用 Round3 的 B1 四臂 checkpoint，在真实 SA64/HSA64 上跑了一遍（**未训练、未调参、未改阈值**）。
这次执行只用来回答"新输出头有没有把 Phase 5 的病灶修掉"，**不作为真实数据训练启动的依据**。

**协议与 §7 / Phase 5 完全一致**：血流带 `[0.15, 0.875]×400`、`log1p(PD/s)` 归一化、网格 `[32,64,64]`、
判定 `n2n ≤ n2n(①Wiener w=7)` 且 `core ≥ 0.95`（预注册写在输出 JSON 的 `prereg.written_before_run=true`）。
命令：

```bash
python -u scripts/v3_phase5_real_network.py --dataset SA64  --ckpt-glob 'runs/v3_round3/b1_*/B1_*/latest.pt' \
  --out-dir reports/v3_round3_real --tag round3
python -u scripts/v3_phase5_real_network.py --dataset HSA64 --ckpt-glob 'runs/v3_round3/b1_*/B1_*/latest.pt' \
  --out-dir reports/v3_round3_real --tag round3
```

**结果：SA64 0/4、HSA64 0/4 通过。**

| 臂 | SA64 n2n / core / dark | HSA64 n2n / core / dark |
|---|---|---|
| 不处理（SVD 输出） | 1.0000 / 1.000 / 1.000 | 1.0000 / 1.000 / 1.000 |
| ①Wiener w=7 | 0.2759 / 0.976 / 1.047 | 0.3774 / 0.933 / 1.240 |
| B1_low_residual | **0.0591** / 0.738 / 1.051 | **0.1512** / 0.814 / **1.111** |
| B1_low_zeroinit | 0.1619 / 0.877 / 1.035 | 0.5262 / 0.949 / 1.042 |
| B1_vlow_residual | 0.2651 / 0.873 / 1.033 | 0.3543 / 0.917 / 1.066 |
| B1_vlow_zeroinit | 0.4777 / **0.960** / 1.022 | 0.7697 / **0.986** / 1.027 |

**对比 Phase 5（Round2 权重）**：最好 n2n 0.9536（SA64）/ 0.9503（HSA64），dark 注入 4.9–7.3× / 2.7–3.8×。
所以 B1 的输出头把两个病灶都修掉了；**但没有换来对 ① 的胜利**。

**读法与限制**

1. 真实数据上可观测到一条 **n2n ↔ 结构权衡**：低 n2n 一律伴随 `core` 0.738–0.917（过度平滑），
   唯一 `core ≥ 0.95` 的 `B1_vlow_zeroinit` n2n 0.478/0.770 反超 ①。与仿真 low6 上"盲模糊 0.618 靠平滑取胜"同源。
2. HSA64 上 `B1_low_residual` 的 n2n（0.1512）与 dark（1.111）双双优于 ①（0.3774 / 1.240），但 core 0.814 < 0.933，
   故只记"一维更好、核心结构更差"，**不写"胜过 Wiener"**。
3. 真实数据 **无 GT**，全秩链 n2n 退化 ×241.8/×5180.4（phase4 §8.1），**真实数据 n2n 没有校准地板**；
   0.059 只做臂间相对比较，不能解读为"接近完美"。
4. 因此 §7 条件 ④ **已完成并留档**，条件 ①②③ **仍全部未满足**，真实数据训练**仍不启动**。

**脚本改动（两处）**：① `net_estimator` 接受 `mode=dense` + `arch_head` 的 checkpoint 约定（B1 checkpoint 无
`infer`/`residual` 键，原版直接拒绝）；② 新增 `--tag`，输出写到 `reports/v3_round3_real/` 与 `*_round3.png`，
**Phase 5 既有产物未被覆盖**。

**产物**：`reports/v3_round3_real/{SA64,HSA64}_network_round3.json`、
`reports/figures/v3_real_network_{SA64,HSA64}_round3.png`、[reports/v3_round3_real_zeroshot.md](reports/v3_round3_real_zeroshot.md)。
**未运行**：任何训练、任何新探针；C/I1、D、`runs/v3_chain_after_T2.sh` 仍未动。

## 9. Round4-0（P0，零训练）已执行：先重建尺子，再拿到一个预注册确认的正向结果（2026-10-29 追加）

**预算消耗**：GPU·h ≈ 0.15，新增训练作业 **0**（Round4 上限 8 GPU·h / 8 作业）。

### 9.1 三把尺子，前两把失败

| 版本 | 指标 | 已知答案控制 | 处置 |
|---|---|---|---|
| v1 | voxel `recall@fpr5` / `iou_core` / `ncomp` | **失败**：完美真值自己只有 0.36–0.57，输入/高斯同量级；断枝 0/6、造假 1/6 | 留 `metric_controls.json` |
| v2 | 段级 CNR | **失败**：真值/输入/高斯两位小数相同；断枝 0/6 | 留 `metric_controls_v2.json` |
| v3 | 对 PSF 匹配真值的 `rmse² = bias² + var` 分解 | **通过**：恒等式 + 五个构造臂，两档各 6/6 | `metric_checks_v3.json` |

**失败根因（一条硬结论）**：`PSF_PD σ=(4.24,3.35,5.61)` 体素 ≫ 血管半径 0.75–1.5 体素，
且段尺度上噪声被平均掉 ⇒ **任何阈值化检测型结构指标在本仿真器上都没有判别区间**。
v3 不阈值化；v3.1 另记一条数据事实：vlow 档 σ*=0.5 的盲高斯 n2n 好（0.81–0.85），
但对真值 RMSE 只略优于不动，且压低血管幅度 0.0045–0.0051。

### 9.2 主表（held e902–e905；操作点只在 e900/e901 上选）

| 臂 | low rmse / bias_ves / var_bg | vlow rmse / bias_ves / var_bg |
|---|---|---|
| 不处理 | 0.0199–0.0205 / ±7e-4 / 3.8–4.1e-4 | 0.0066–0.0068 / ±2e-4 / 4.3–4.5e-5 |
| gauss(σ*) | **0.0097–0.0102** / −0.0070…−0.0083 / **0.91–1.02e-4** | **0.0053–0.0057** / −0.0045…−0.0051 / **2.7–3.1e-5** |
| wiener(w=3) | 0.0130–0.0141 / −0.0115…−0.0141 / 1.66–1.94e-4 | 0.0100–0.0113 / −0.0121…−0.0145 / 9.7e-5–1.23e-4 |
| net_direct | 0.0244–0.0559 / −0.0426…+0.0184 / 5.9e-4–1.08e-3 | 0.0208–0.0527 / −0.0417…+0.0178 / 4.3e-4–8.2e-4 |
| net_residual | 0.0235–0.0334 / −0.0016…−0.0090 / 5.4e-4–1.04e-3 | 0.0053–0.0062 / **−0.0004…−0.0011** / 2.7–3.7e-5 |
| net_zeroinit | 0.0154–0.0171 / **−0.0017…−0.0028** / 2.4–2.9e-4 | 0.0061–0.0071 / **−0.0003…−0.0006** / 3.8–4.9e-5 |

**0 Gate**：`BASELINE_DOMINATED = False`，两档三臂"非支配"。**但这个 Gate 无判别力**：
`bg_std_ratio` 全部落在 0.983–1.002（背景波动由 PSF 旁瓣主导），该轴在本仿真器上退化。
明细里 `net_direct` 在 4 卷中的 3 卷被 gauss 与 wiener 同时支配，只因规则要求"全部 held 卷"才未判支配。
按预注册"不得因某轴无判别力就宣告胜利"，**本轮不使用这个 Gate 声称任何结论**。

### 9.3 确认实验（e906–e911，六卷；β 规则在看数据之前写死）

β 选点规则（`V3_PR1_round4_prereg_v4_confirm.md` §1，只在 e900/e901 上算）：
`β* = argmin |bias_ves(β)| s.t. var_bg(β) ≤ var_bg(gauss σ*)`，空集 → `NO_FEASIBLE_BETA`。
通过线：对 gauss 与 wiener 各三条指标（`var_bg` / `|bias_ves|` / `overall_rmse`）都不劣，≥5/6 卷。

| 档 | 臂 | β* | vs gauss（var/bias/rmse） | vs wiener | 结论 |
|---|---|---|---|---|---|
| low | net_direct / net_residual / net_zeroinit | — | — | — | `NO_FEASIBLE_BETA` |
| vlow | net_residual | 0.5 | 5/6 · **6/6** · 5/6 | 6/6 · 6/6 · 6/6 | **`STRUCT_SAFE_GAIN_CONFIRMED`** |
| vlow | net_zeroinit | 0.5 | 5/6 · **6/6** · 5/6 | 6/6 · 6/6 · 6/6 | **`STRUCT_SAFE_GAIN_CONFIRMED`** |

未通过的 1 卷是 **e910 的 `var_bg`（2.42e-5 > 2.27e-5）**，如实记录、不做解释性豁免。

### 9.4 边界（请审查方重点看这一节）

1. **low 档不变**：三臂全部 `NO_FEASIBLE_BETA` ⇒ 旧 Gate"网络 0.825 输给盲高斯 0.618"在该档依然成立。
2. **不是"全局最优"**：β 是 Round4 引入的后处理旋钮；选点规则虽在确认集之前写死，
   但"网络在匹配噪声下结构偏差更小"这一**假设**是从 e902–e905 看出来的 ⇒ **探索+确认两段式**，
   强度上限是"在该仿真器、该 6 卷上复现"，**不可推广**。
3. **不可外推到真实数据**：SA64/HSA64 无 clean GT，本节结论只在仿真器内成立。
4. **`bg_std_ratio` 不可用于任何比较**。

### 9.5 产物

- 报告：`reports/v3_round4/baseline_pareto.md`；数据：`baseline_pareto.json`、`p0_confirm.json`、
  `metric_controls.json`(v1) / `metric_controls_v2.json`(v2) / `metric_checks_v3.json`(v3/v3.1)
- 图：`reports/figures/v3_round4_pareto.png`（两档 `var_bg` vs `|bias_ves|` Pareto 前沿 +
  确认集 `bias_ves(net)/bias_ves(gauss)` 逐卷比值）
- 预注册：`V3_PR1_round4_prereg.md` / `_v2_metric` / `_v3_metric` / `_v3_1_metric_control_fix` / `_v4_confirm`；
  审阅答复：`V3_PR1_round4_review_response.md`
- 脚本：`scripts/v3_round4_{metrics,metrics_v2,metrics_v3,metric_controls,metric_controls_v2,metric_checks_v3,p0,p0_confirm,figure}.py`

**下一步（未启动）**：Round4-B0 真实数据三域统计 → B1 三臂短 pilot（≤3 训练）。C/I1 与 D 未动，
`runs/v3_chain_after_T2.sh` 未运行，未改任何阈值，未合并本 PR。

> **后续状态（2026-11-02 补记）**：本节写的"下一步"——B0 / B1 / C 前置筛 / D——**已全部执行完毕**，
> 结果与判定见 §10。`runs/v3_chain_after_T2.sh` 仍未运行，阈值未改，PR 仍未合并。

---

## 10. Round4-B / C / D 已执行：一次止损结案 + 一个探索级正向信号（2026-11-02 追加）

### 10.1 判定（一页式：`reports/v3_round4/DECISION.md`）

> ## `I1_UNTESTED_BECAUSE_BASELINE_FAILED`

触发依据：任务书 §5 停止条件 **1**（结构评价的正负控制无法修复 ⇒ `METRIC_UNRESOLVED`）。
C 的三臂对照需要结构主指标，而 v1/v2 判据的已知答案控制失败、v3 的 0-Gate（`bg_std_ratio` 0.983–1.002）
在本仿真器上退化 ⇒ **不得为了"必须验证 I1"强行训练**。
**这不是"I1 被证伪"**：C 一次也没跑，正确/打乱先验的对照从未成立。

为什么不是另外三个标签：

| 标签 | 成立？ | 依据 |
|---|---|---|
| `PROMISING_REQUEST_APPROVAL` | ✗ | 结构安全增益✗（尺子失效）、物理先验差异✗（§10.4 精确并列）|
| `STOP_CURRENT_CHAIN` | ✗ | 基础链**确实**优于强基线（§10.3），"不优于强基线"的前提是假的 |
| `STOP_I1` | ✗（时间上） | 该标签要求"可信且直接的 C 对照里正确先验无优势"；C 未运行 |

### 10.2 B0（零训练）：真实数据三表示域的条件方差斜率

`d log Var / d x`（x 已在 log 域；越大 = 噪声随信号涨得越快）：

| 表示域 | SA64 | HSA64 |
|---|---|---|
| `linear: x = u` | +2.147 | +2.141 |
| `log1p: x = log1p(u)` | +1.371 | +1.459 |
| `logeps: x = log(u+1e-3)` | **+0.035** | **+0.107** |

乘性噪声正控制复现同一方向；块差分 ACF（SA64 −0.006/0.425/0.169、HSA64 0.050/0.505/0.235）
说明相邻噪声相关、不能假设逐点独立。**B0 只到"统计量确实不同"，不回答"训练会不会更好"**
（报告里明确标为 `NOT_MEASURED_IN_B0`）。

### 10.3 B1（训练 3 臂，只差表示域）：15 epoch × 500 卷，单 seed

`rmse_img`（对 PSF 匹配真值）：

| 场景 | 输入 | `gauss(σ*=0.6)` | `linear` | `log1p` | `logeps` |
|---|---|---|---|---|---|
| e902 | 0.0202 | 0.0097 | 0.0089 | 0.0093 | **0.0086** |
| e903 | 0.0205 | 0.0102 | 0.0094 | 0.0099 | **0.0091** |
| e904 | 0.0199 | 0.0101 | 0.0073 | 0.0077 | **0.0070** |
| e905 | 0.0200 | 0.0098 | 0.0085 | 0.0082 | **0.0077** |
| 均值 | 0.0201 | 0.0099 | 0.0085 | 0.0088 | **0.0081** |

`logeps` vs `gauss` **+11.9% / +12.7% / +43.6% / +26.5%（4/4）**；vs `log1p` 6.1–9.1%（4/4）；
`linear` vs `log1p` 3/4（e905 差 3.7%）。
其余：三臂 `bias_ves` 负偏只有高斯的 1/4–1/5（−0.0009…−0.0018 vs −0.0077）、
`var_bg` 6.57e-05…7.73e-05 vs 9.65e-05、`ret_p10` ≈0.997 vs 0.991。
⇒ **臂间差异几乎全在 `var_bg`**，`bias_ves` 三臂同量级。

**冻结的 B Gate 原样执行**（预注册 §3.3，未事后修改）：`linear` 3/4·4/4·4/4、`log1p` 4/4·4/4·4/4、
`logeps` 4/4·4/4·4/4 ⇒ `B_NO_GAIN = false`。两条保留：
① 第 1 条判据对 `log1p` 臂是**同义反复**（预注册文本漏洞，不追溯修改，结论只用跨臂比较）；
② 第 3 条判据因 `METRIC_RESOLVED` 失败而**形式上失效**，通过也不算"结构安全"证据。

**机制核对**：支持的一半——稳定化最彻底的 `logeps` 同时最低 `var_bg` 与最低 `rmse`；
不支持的一半——`linear`(斜率 2.14) 反而略优于 `log1p`(1.4)，且三域 loss 尺度不同（`logeps` 1.94e-4 最大），
"收敛更快"与"表示更好"未分离。⇒ 只能说**与方差稳定化一致**，不构成因果。
**预注册允许的唯一一条追加**：`logeps` 15→40 epoch 续训（同 checkpoint、同 seed，
`train.py` 新增 `--resume`，只加载权重 + 全局 epoch 学习率衰减，先经烟测核对），实测 1.30 GPU·h。
**结果是负的**：

| 场景 | 15 ep | 40 ep | 变化 |
|---|---|---|---|
| e902 | 0.00864 | 0.00901 | +4.3%（差） |
| e903 | 0.00908 | 0.00948 | +4.4%（差） |
| e904 | 0.00702 | 0.00736 | +4.8%（差） |
| e905 | 0.00774 | 0.00809 | +4.5%（差） |

4/4 体一致变差，变差集中在 `var_bg`（+9.1%）与 `bias_ves`（负偏幅度 +33%）⇒ 更久训练只是把输出压得更平。
**Gate 判定不回填、不重跑**；排序结论不变（40 ep 的 `logeps` 仍优于 `gauss` 4/4：+6.8%/+7.4%/+27.0%/+17.3%，
也仍优于 15 ep 的 `linear`）；但"再多训会更好"这个解释被删掉。明细 `b1_pilot_40ep.json`、`log_domain_pilot.md` §2.8。

### 10.4 C（I1）：按止损规则 `NOT_RUN`，外加一道零训练的几何筛

C 触发要求 (a) 非支配区 **且** `METRIC_RESOLVED`，(b) clear positive signal；(a) 已在 P0 失败。
为避免"不跑就是躲"，另做零训练前置筛（`scripts/v3_round4_c_screen.py` → `c_screen.json`）：

1. **几何恒等式**：2×2×2 角块任意 4+4 分割，组内距离多重集恒为 {1,1,1,√2,√2,√2}，
   而预测跨伴图相关 =(总协方差和 − A 内 − B 内)/2 ⇒ 任何可分离自相关模型下**预测值精确并列**
   （实测极差 **6.9e-18**）。相关性感知先验在 VALID 家族内**没有区分力** ⇒ 不存在"随数据变化的干预"。
2. 实测族内跨伴图相关确有 0.07–0.17 差异，但无法由相关模型解释、argmin 在场景间跳动 ⇒ 不可复现的选择规则（任务书 §4 条款）。
3. 真正的结构代价是半个体素错位：族内信号失配中位 **0.125** vs 共享质心分法 **0.00056（223×）**。
   ⇒ 结构安全增益只能靠**离开 VALID 家族**（固定采样重设计），**不能**记在"相关性自适应先验"名下。
4. 机械判定 `I1_NO_DISTINCT_INTERVENTION`。

### 10.5 D（零训练）：核 SVD-SN2N（Zhang et al. 2026）

来源可核验（bioRxiv v1 2026-09-13，JATS XML 可取；**官方代码未检出**、Table 2/3/4 在 XML 内为图片 → 记"未知"）。
逐项核对结论：log 压缩是**操作性成分**（其自身对照：线性输入只 +1.1–2.4 dB、不缩窄血管），
与我们 B0/B1 的方向一致；其"子视图"是**像素域对角拆分**，与"按采样几何挑样点"不是一回事
⇒ **不能**说 I1 已被他人验证。
在有真值的 low 档上实测六种伴图构造：**VALID 8 个 tetris 模式无一共享质心**（`any_shared=false`），
默认配对带 (0.5,0.5,0.5) 体素位移 ⇒ 信号失配 **0.122**；换成共享质心 `{0,3,5,6}/{1,2,4,7}` 降到 **0.00056**，
噪声方差基本不变（1.75e-4 → 1.98e-4）；Fourier 插值未引入伴图噪声相关（0.114 → 0.064）。
**决策：不投入完整复现**（任务书 §3 条件 + 预算），两条候选改动登记为待训练验证。
明确不支持的推论：**噪声相关更低 ≠ 训练更好**，本轮没训任何"SN2N 式"网络。

### 10.6 预算记账

| # | 作业 | 实测 | 训练作业计数 |
|---|---|---|---|
| 1 | P0 推理（Pareto + 控制 + v3 指标） | ≈0.15 GPU·h | — |
| 2 | B0 三域统计 | 0 | 0 |
| 3 | D 文献核对 + 视图构造数值检验 | 0 | 0 |
| 4 | C 前置筛 | 0 | 0 |
| 5–7 | B1 pilot `linear` / `log1p` / `logeps`（各 15 ep × 500 卷） | 23:55 / 26:28 / 23:34 | 1–3 / 8 |
| 8 | `logeps` 续训 15→40 ep | **1.30 GPU·h**（25 ep，01:04–02:21；GPU 争用致 180–188 s/ep） | 4 / 8 |

**累计训练 ≈3.3 GPU·h / 8，训练作业 4 / 8。** 停止原因是 §5-1，**不是**预算耗尽。

### 10.7 请审查方重点看的边界

1. B1 是 **15 epoch / 单 seed / 4 场景**的探索级证据，不构成确认；确认轮需另行预注册（3 seed × **≤20 ep + 在 e900/e901 上 early-stop 选点** × 6 卷，≈1.1–1.7 GPU·h / 6 作业，**未启动**）。
2. 增益**集中在背景方差项**，不能说"血管结构被保住"；结构类主张在本仿真分辨率下**不可解**。
3. 机制未分离（表示 vs 可优化性），不能说"因为是方差稳定化"。
4. 不能说 I1 被证伪；也不能把 §10.4 的"失败"当成"I1 输了"——那是"做不出合法对照"。
5. 所有阈值/Gate **未事后修改**；历史负结果全部保留；`runs/v3_chain_after_T2.sh` 未运行；PR 未合并。

### 10.8 产物

- 报告：`reports/v3_round4/{DECISION.md, log_domain_pilot.md, i1_last_pilot.md, svd_sn2n_fidelity_and_decision.md}`
- 数据：`reports/v3_round4/{b0_logstats.json, b1_norm.json, b1_pilot.json, b1_pilot_40ep.json, d_subview.json, c_screen.json}`
- 图：`reports/figures/v3_round4_b1_domains.png`（三臂逐场景 rmse 柱 / `bias_ves`–`var_bg` 散点 / e904 六格切片条）、
  `reports/v3_round4/fig_b0_logstats_{SA64,HSA64}.png`
- 脚本：`scripts/v3_round4_{b0_logstats, b1_eval, b1_figure, c_screen, d_subview}.py`；`train.py` 新增 `--resume`

---

## 11. Round5 已执行：BD-2 与 D1 的**直接**对照（2026-11-02 追加；2026-11-05 改按几何命名）

> 这一节是对 §10.4 / §10.7-4 的一处**口径更正**，请在读时把它当作 Round4 结论的补充而非推翻。

### 11.1 为什么补这一步（一处自我更正）

§10.4 写"C 未运行 ⇒ I1 未检验"，容易被读成"采样创新本身没被检验"。这是**过度外推**，两条都查得清：

1. Round4-B1 的四臂是 `--mode dense` + `--train-clean-volumes`：`dense_views` 对四个视图全部返回
   全分辨率同块，目标又被同位置干净块替换（`xa is xb`，identity 项恒为零）⇒ **它们是监督上界臂，不是 SN2N**。
2. Round4-C 的前置筛只证明了"8 种 tetris 模式里选哪一种"是几何恒等式（极差 6e-18），
   **不覆盖** `--stride` 与 `--role-order/--slots`——而这两个才是我方采样的实际变量。

因此"采样创新有没有用"可以用误差类指标直接回答，不需要结构尺子。本节就是这次直接对照。

### 11.2 设计与预注册

预注册（**跑之前**写死判据）：`reviews/V3_PR1_round5_prereg_sampling_direct.md`。
三臂全部**自监督**、同网络/损失/预算/seed，唯一变量是采样：

| 臂 | 域 | 采样 |
|---|---|---|
| `S_log` | `logeps` | 标准 VALID：`--stride 1 --mode random --pairing cross`（m=1 随机 tetris） |
| `O_log` | `logeps` | 我们的：`--stride 2 --mode fixed --slots 1 4 6 3 --pairing cross`（m=2 + 体对角角色对） |
| `O_lin` | `linear` | 同 `O_log`，域换线性 |

`15 ep × 500 步`、`patch 32×64×64`、`features 16`、`seed 3407`、`residual` 头 + 零初始化；
参照 `input` 与 `gauss(σ*=0.6)`。判据：C1 实质差异（逐场景 ≥5% 且 ≥3/4 同向）/ C2 无实质差异（<2%）/ C3 否则。
预注册里**预先写下** Phase 3 的负先验（m=2 落后 3–7 倍）。

### 11.3 结果：`C1_实质差异 / OURS_WORSE`

留出 e902–e905，`rmse_img`（主口径 `own_pairing`：按各臂真实被训练的映射回填，所有臂同一 25% 掩膜评分）：

| 臂 | e902 | e903 | e904 | e905 |
|---|---|---|---|---|
| **`S_log` = D1（间距 1）** | **0.0166** | **0.0190** | **0.0193** | **0.0173** |
| `O_log`（我们的） | 0.2156 | 0.2061 | 0.2001 | 0.2027 |
| `O_lin`（我们的·线性） | 0.1423 | 0.1844 | 0.1724 | 0.1291 |

相对变化 **+1200% / +983% / +939% / +1069%**（4/4 更差）。`fold_all` 字面口径同向（+832% ~ +1005%）。

**机制**：m=2 下跨相位信号传输 `r_sig` 仅 0.27–0.29 ⇒ MSE 最优解退化为输出均值
（`bias_ves` −0.10~−0.16、`var_bg` 高两个数量级、`n2n` 50–58）。与 `reports/v3_phase3_probe.md` 方向一致。
**部署代价**：单个固定角色对只能填 **25%** 体素；推理前向 **8×**（8 相位 × m³=8 平移）。

### 11.4 口径披露（请按此打折）

- 预注册 §3 字面是"8 相位回填"。对随机角色臂（`S_log`）正确；对固定角色臂会被问没学过的映射
  （Phase 3 `reconstruct` 的 docstring 早已写明）。`own_pairing` 是**看到 `fold_all` 结果之后**补的口径。
  它不改判据、数据、臂定义；**两口径结论同向**，故不影响判定。
- 三个脚本 bug 在执行中自查修正（收窄 `valid` 后喂网络致 `S_log` 虚高到 0.59；
  `SCATTER` 传外层字典致预测退化为常数、会读出"灾难性更差"的假数字；37→36 裁剪只做在指标侧致 `IndexError`；
  `gauss` 参考行在窄掩膜内做归一化卷积致分母退化、虚高到 0.0195，已改并重跑）。
  修前修后的数字都保留在 `reports/v3_round5/sampling_direct.md` §2。

### 11.5 结论边界

**能说**：同网络/损失/预算/数据下，**D1**（对角配对、间距 1，SN2N 原型）显著优于 **BD-2**（体对角固定配对、间距 2，本方案），9.4–12×，4/4 场景。
（2026-11-05 命名修订：旧写法"标准 SN2N 采样 / 我们的采样"改为按几何命名，**改动原因**是前者只是对角配对这一条具体几何、后者没有名字无法作图例；数字与判据未变。）
**不能说**："采样创新在原理上一定无效"（只测了 m=2 + 体对角这一个配置；m=1 换角色、m=3、
数据驱动间距均未测）；不能据此说"物理先验被证伪"；结构保真仍未被验证。
**对 §10 的影响**：`I1_UNTESTED_BECAUSE_BASELINE_FAILED` 不变；但**"采样几何创新"不应再作为已获支持的贡献**。

### 11.6 预算与产物

训练 ≈0.52 GPU·h、推理 ≈0.05 ⇒ 本轮 ≈0.57 GPU·h / 3 作业；**Round4+5 累计 ≈3.9 GPU·h / 8、作业 7 / 8**。
产物：`reports/v3_round5/{sampling_direct.md, sampling_direct.json, sampling_direct_ownpair.json}`、
`reports/figures/v3_round5_sampling_direct.png`、`scripts/v3_round5_{sampling_pilot.sh, eval.py, figure.py}`。
未改任何阈值/Gate；PR 保持 open、未合并。

**通俗总结**：之前说"采样创新没被检验"其实只意味着"那个大对照没跑"，不意味着测不了。这一轮用完全相同的条件
（同网络、同损失、同步数、同数据，只换"怎么切数据"）把两种配对方式正面对了一次——**D1**（相邻层配对，
SN2N 的做法）和 **BD-2**（我们设计的体对角、间距 2 配对）。结果是 **D1 赢，而且差 9–12 倍，四个场景全输**。
原因是间隔拉大后输入与目标几乎无关，网络学会了输出平均值，血管被抹平、背景噪声留着。
所以"BD-2 这套采样几何有效"这句话不能再说了；能站住的只剩"噪声清洗有用"和"对数压缩是最好的表示选择"。
但也不能说"物理先验被证伪"——只输了这一种配置。
