"""偏置感知验收判据 J1.2 —— **只在此处定义，标定与真实数据共用同一份实现**。

## 版本变更记录（改了什么、为什么；不许悄悄改）

**J1.0（作废，标定失败）**：两路 `N_rel + B_rel`，`B_rel` 为暗区"总能量减方差"。标定显示该
偏置项在幻影上对所有平滑估计器都 ≈1.0、**无区分度**。作废原因不是阈值没调好，而是构造
错误：模糊这个缺陷根本不在暗区能量里体现（它是**削掉亮区结构**），而暗区被"共享静态杂波"
主导，各向同性平滑都去不掉它。

**J1.1（作废，标定 3/5 通过）**：改为三路 N/B/S，`B_rel` 仍按**输入**归一且设零容忍
（`B_rel<=1.0`）。标定暴露两个问题：
1. 零容忍**连项目自己的参考估计器都过不了**——温和收缩 w=5 的 `B_rel=1.0120`。原因是
   `out=mu+b(x−mu)` 里 `mu` 是 log 值的局部均值，在平坦背景上会略微抬高能量，
   "完全不添背景"这个要求本身不可实现。故零容忍不成立。
2. 按输入归一让门槛跨数据集不可比：输入本身背景越脏，同样的绝对添加反而显得越轻。

**J1.2（本版）**：`B` 改为**对血管能量归一**（`B_frac = b_dark / core_energy`），阈值由
幻影标定给出（见下）。这是唯一一处从标定里取数的阈值，规则在跑之前写死。

## 三路（全部在 log1p 域，与全项目其它指标同域）

- **N（噪声）** `N_rel = mse(p,q)/mse(xo,xe)` —— 即 n2n。挡"什么都不做"与"注噪"。
- **B（暗区假阳性）** `B_frac = b_dark(p,q,dark) / mean_core((p²+q²)/2)`。
  `b_dark(a,b) = mean_dark((a²+b²)/2) − ½mean_dark((a−b)²)`：依据暗区真值为 0
  （`v3_simulator` 的真值定义 = 只有血流那一项的带通 PD），`E[a²]=bias²+var`、
  `½E[(a−b)²]=var`，**相减只剩 bias²**——这是唯一能真正测出偏置的通道。挡"整片抬亮/往
  背景添东西"（④ 的病；n2n 对它恒为不变，见标定 offset 扫描）。
- **S（结构保留）** `S_rel = mean_core(p)/mean_core(xo)`（log1p 域，与项目既有 `keep_ratio`
  逐字同一实现）。挡"抹平换分"（n2n 的病，见标定 blur 扫描）。

## 预注册门槛

- `N_rel <= 0.5`   —— 必须把奇/偶不一致度**至少砍半**（不处理 = 1.0）。含义定死，不看结果。
- `S_rel >= 0.95`  —— 沿用项目已预注册的 `THR_CORE = 0.95`，不新造数字。
- `B_db <= THR_B_DB` —— **由幻影标定给出**：`THR_B_DB = (幻影上真值误差最小的估计器 = w=7
  收缩) 的 B_db + 3 dB`。3 dB 是**事先声明**的松弛，理由：估计器的尺度常数（`s`、掩膜边界
  效应）不确定性就在这个量级；加到参考估计器上，保证"已知良好"的估计器留有余量。
- 三条**同时**满足才算通过。

## 已知局限（必须与结论并列写出）

1. 三道门都是**必要条件不是充分条件**。全过只说明"没犯这三类错"，不证明去噪变好。
2. `B` 只在"暗区真值为 0"的先验下成立。真实数据无真值，该先验依据是 Power Doppler 的物理
   语义 + 掩膜低参考档（沿用 `DARK_THR=0.05`，不另调）。若暗区真有低速血流，会误判。
3. 亮区**内部**的过冲（把血管增强到超真值）三路都测不到，因为亮区无先验。
4. 不设加权总分。加权系数一旦可调就给了事后挪门槛的空间；三路并列、全过才过。

## R09 更正（2026-10-26，外部审查）：`B` 到底在测什么，必须写清

**恒等式**：`b_dark(a,b) = mean((a²+b²)/2) − ½mean((a−b)²) ≡ mean(a·b)`（代数恒等，已数值验证
到 1e-16）。所以 `B` 实际测的是**跨两次处理的平均乘积**，不是直接的 bias²。

只有在下面三条**同时成立**时，`mean(pq)` 才约化为 `bias²`：
(i) 两次处理的**信号成分相同**；(ii) 噪声各自**零均值**；(iii) **两次的噪声互不相关**。
三条中任何一条破了，读数都会偏：

- 若两次噪声**正相关**（真实 SA64/HSA64 的奇/偶半卷**共享静态杂波**，正是这种情况），
  `mean(pq)` 含 `+cov(n1,n2)`，**读数偏高（更不易通过）**。也就是说：真实数据上的 `B` 在
  这个方向上是**保守的**，它把"共享杂波"也算进了"偏置"。④ 在真实数据上不过 B，不是因为
  恒等式把它算轻了。
- 若暗区真有**非零真实血流**，`mean(pq)` 含 `+s²`，同样偏高。
- 反向的风险是：若两次处理的噪声**负相关**（构造上可能出现），`B` 会被低估。本项目未观察到。

**因此**：`B_db <= −7.3` 是**对这套实现的工程门槛**，不是"假阳性能量低于某个物理常数"。
`N_rel <= 0.5` 同理。跨数据集直接套用这两个数需要另行标定，不得因某次通过就宣布图像质量提升。
`S_rel` 只是亮核均值保持，**不能**单独识别断裂、细分支损失、局部过冲或假管——这些要靠
真值体模的结构指标（弱血管 recall@固定 FPR、宽度、连通性），本项目**尚未实现**。
若将来要做更强的判据，另建 **J2** 并预注册；J1.2 的旧数值与旧判定必须能原样复现。
"""
from __future__ import annotations

import numpy as np

# 预注册门槛
THR_N_REL = 0.5          # 含义定死：奇/偶不一致度至少砍半
THR_S_REL = 0.95         # 沿用项目既有 THR_CORE
# 由幻影标定给出：THR_B_DB = B_db(幻影上真值误差最小的估计器 = w=7) + 3 dB
# 实测 w=7 的 B_db = −10.28 → 门槛 −7.28，取 −7.3。推导见 reports/v3_judge/JUDGE_v1.md
THR_B_DB = -7.3

JUDGE_VERSION = "J1.2"
THR_TEXT = "N_rel<=0.5 且 S_rel>=0.95 且 B_db<=-7.3dB（= 幻影 w=7 的 B_db −10.28 + 3dB 松弛）"


def mse(a, b, mask):
    return float(np.mean((a[mask] - b[mask]) ** 2))


def b_dark(a, b, mask):
    """暗区假阳性能量：mean((a²+b²)/2) − ½mean((a−b)²)，已消去 var，只剩 bias²。"""
    if mask.sum() == 0:
        return float("nan")
    var = 0.5 * float(np.mean((a[mask] - b[mask]) ** 2))
    tot = float(np.mean(0.5 * (a[mask] ** 2 + b[mask] ** 2)))
    return tot - var


def judge(p, q, xo, xe, masks, valid):
    """三路判据。返回各分量与判定（N/B 越低越好，S 越高越好）。"""
    ref = mse(xo, xe, valid)
    n_rel = mse(p, q, valid) / max(ref, 1e-30)

    d = masks["dark"] & valid
    b_out = b_dark(p, q, d)
    core_e = float(np.mean(0.5 * (p[masks["core"] & valid] ** 2
                                  + q[masks["core"] & valid] ** 2)))
    b_frac = b_out / max(core_e, 1e-12)

    c = masks["core"] & valid
    s_rel = float(np.mean(p[c]) / max(float(np.mean(xo[c])), 1e-12))

    ok_n = bool(n_rel <= THR_N_REL)
    ok_b = bool(10.0 * np.log10(max(b_frac, 1e-30)) <= THR_B_DB)
    ok_s = bool(s_rel >= THR_S_REL)
    return {
        "n_rel": float(n_rel), "n_db": float(10.0 * np.log10(max(n_rel, 1e-30))),
        "b_out": float(b_out), "b_frac": float(b_frac),
        "b_db": float(10.0 * np.log10(max(b_frac, 1e-30))),
        "s_rel": s_rel, "s_db": float(10.0 * np.log10(max(s_rel, 1e-30))),
        "ok_n": ok_n, "ok_b": ok_b, "ok_s": ok_s,
        "n_fail": int((not ok_n) + (not ok_b) + (not ok_s)),
        "pass": bool(ok_n and ok_b and ok_s),
    }
