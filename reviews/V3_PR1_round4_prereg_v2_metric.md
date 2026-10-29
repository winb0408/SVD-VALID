# Round4 预注册 v2：结构指标改为**段级可检测性**（v1 已记录失败）

> **性质**：对 [V3_PR1_round4_prereg.md](V3_PR1_round4_prereg.md) §2.2–2.3 的**版本化修订**，
> 依据是 v1 的已知答案控制**没有通过**这一实测结果（`reports/v3_round4/metric_controls.json`）。
> 按 Plan §0.5：撤换端点必须留证；新判据换版本，旧判据原样保留，不回填历史。

## 1. v1 为什么被判失败（实测，不是猜）

`reports/v3_round4/metric_controls.json`（已在跑完当时落盘，未改动）显示，即使把参照换成 PSF 匹配的
`gt_imaging_pd`，v1 的五个控制仍是：

| 控制 | 通过 |
|---|---|
| 正控制 1 `recall(truth) > recall(input)` | **4/6** |
| 正控制 2 `rmse(truth) < rmse(input)` | 6/6（近乎恒真，因为 truth 与自己是 0 距离） |
| 正控制 3 连通性 | 5/6 |
| 负控制 A 断枝（recall 与 ncomp 都要降） | **0/6 与 0/6** |
| 负控制 B 造假（`fp_blobs` 要升） | **1/6** |

根因（两个，都可直接读出）：

1. **voxel 级 `recall_at_fpr5` 在本仿真里没有判别区间**：完美真值自己只有 0.36–0.57，
   而输入/高斯也是 0.36–0.57，差距 ≤0.005。原因不是实现错，是分辨率：`PSF_PD σ=(4.24,3.35,5.61)` 体素、
   血管半径 0.75–1.5 体素，**体素级**上"弱血管"与背景在无噪真值里就已经重叠。
   一个在完美答案上都只有 0.47 的量，无法判断任何方法是否损伤结构。
2. **负控制实现不到位**：`cut` 抹掉的是 `core`（`rho≥0.5`）的连通域，而 `recall` 只在 `weak`
   （`rho∈[0.1,0.5)`）上算，两者不相交 → 断枝当然测不出来；`fake` 管的强度取弱血管中位数，
   落在阈值以下 → 造血管当然多数不触发。

第 2 条是**控制设计缺陷**（可以且必须修）；第 1 条是**指标本身的判别力问题**（修控制也救不回来）。
因此不是"再调调阈值让它过"，而是换一个**与被测对象同尺度**的指标。

## 2. v2 指标：段级可检测性（segment-level detectability）

**为什么换尺度**：血管是一根根管子，管子直径小于 PSF，但其**轴向长度远大于 PSF**。
在整条管子上取平均，PSF 造成的分辨率限制就被平均掉了——这就是匹配滤波/CNR 的常规做法。
体素级评价等于要求"逐点分辨"，那是 PSF 不允许的；段级评价要求"这条管子能不能被看出来"，
那是成像任务真正关心的。

### 2.1 冻结定义

**段（segment）的分割只来自 phantom 几何，不看任何观测**：
`M_seg = valid & (rho ≥ 0.1)`，3D 连通分量，保留体素数 ≥ `MIN_SEG_VOX = 30` 的分量。
记第 s 段体素集合为 `V_s`，共 `N_seg` 段。背景 `bg = valid & (rho ≤ 0)`。

对任一候选图像 `pred`（log1p 域，与 `E["vol"]` 同尺度）：

```
bg_mu  = mean(pred[bg]);           bg_sd = std(pred[bg])
snr_s  = (mean(pred[V_s]) − bg_mu) / bg_sd            # 段级对比噪声比（每段一个数）
```

| 端点 | 定义 | 方向 |
|---|---|---|
| `seg_cnr_median` | `median_s(snr_s)` | 越大越好（连续量，主判据） |
| `seg_recall_z2` | `mean_s(snr_s ≥ 2)`（z 固定 2，不调） | 越大越好 |
| `fp_segments` | `pred > bg_mu + 2·bg_sd` 在 `bg` 内的连通分量数（≥30 体素） | 越小越好 |
| `seg_cnr_p10` | `10% 分位`（最弱那批管子） | 越大越好（"弱血管"就藏在这里） |

保留 v1 的三个**可核验标量**：`n2n`、`rmse_img`、`bg_std_ratio`（不依赖结构解释）。
`width` 仍 `NOT_RESOLVABLE`。voxel 级 `recall_at_fpr5` / `iou_core` / SSIM 一并保留为**描述性记录**，
不得作判据（v1 的实测就是它制定不了判据的证据）。

### 2.2 v2 已知答案控制（**先写死，再跑**）

| 控制 | 构造 | 期望 | 通过线 |
|---|---|---|---|
| 正控制 1 | `truth_img` vs `input` | `seg_cnr_median(truth) > seg_cnr_median(input)` | **6/6 卷** |
| 正控制 2 | 同上 | `seg_cnr_p10(truth) > seg_cnr_p10(input)` | **6/6 卷** |
| 正控制 3 | `gauss(σ*)` vs `truth_img` | 真值的 `seg_cnr_median` 不低于高斯（高斯是"能赢噪声但丢细节"的参照） | **6/6 卷** |
| 负控制 A（断枝） | `cut30` = 真值里**体积最小的 30% 段**整段置 0（确定性，不用随机数） | `seg_recall_z2(cut30) ≤ 0.75` 且 `< seg_recall_z2(truth)` | **6/6 卷** |
| 负控制 B（造假） | `fake` = 真值背景里插一条直管，强度**按构造**取 `bg_mu + 3·bg_sd`（保证可检出），半径 1.5 体素、长度 ≥30 体素、完全落在 `bg` 内 | `fp_segments(fake) > fp_segments(truth)` | **6/6 卷** |
| 正控制 4（空答案） | `zero` = 全零 | `seg_cnr_median(zero) = 0`（常数图不得有任何段"显著"） | 6/6 卷 |

全部满足 → `METRIC_RESOLVED_V2`；否则仍 `METRIC_UNRESOLVED`，此时**P0 只报三个标量**，
不得宣称结构安全/受损（与 v1 规则相同，不豁免）。

**已知风险并预先声明**：`seg_cnr_median` 会奖励"背景压得干净"，一个把血管也压掉的输出
可能仍拿到不错的中位 CNR；因此它必须与 `seg_cnr_p10`、`seg_recall_z2`（对弱段敏感）**一起**看，
任何单一端点都不作为"结构安全"的充分证据。这条限制写进 P0 报告的结论段。

## 3. 对 P0 Gate 与 B Gate 的影响（只改"结构"那部分，其余不动）

- 0 Gate 的支配判定里，"结构"分量由 v1 的 `recall_at_fpr5` 换成 **`seg_cnr_median` 与 `seg_cnr_p10`**
  （两者都不劣于网络臂才算支配），`rmse_img` 与 `bg_std_ratio` 不变。
- B Gate 的"结构安全"分量同上替换；若 `METRIC_UNRESOLVED_V2`，B 的结构判据作废，
  只剩 `rmse_img` 与 `bg_std_ratio` 两条，结论标签固定为探索。
- 预算、split、选参纪律、停止条件、旧的 Gate（`G-B1` 等）**全部不变**。

## 4. 不改的东西

v1 的 JSON、判据、以及"v1 失败"这个事实全部保留并可复算（`scripts/v3_round4_metric_controls.py` 未删未改）。
v2 另写 `scripts/v3_round4_metric_controls_v2.py`，输出 `reports/v3_round4/metric_controls_v2.json`。
