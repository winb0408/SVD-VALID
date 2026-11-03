# 几何与损失梯度审计（R3-P04 / R3-P06）

**日期**：2026-10-28。**性质**：只读诊断 + 单测，不训练。产物：

- 单测：`tests/test_v3_round3_geometry.py`（15 项）、`tests/test_v3_round3_dense_identity.py`（7 项）
- 实测：`runs/v3_round3/audit_{dense_clean,random_clean,random_n2n,fixed_clean}/loss_grad_audit_*.json`
- 代码：`train.py` 里新增 `dense_views()` / `loss_terms()` / `grad_norm()` / `sample_views()` 与 `--audit-loss-grads`

复算命令：

```bash
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
D=data/processed/v3_round2_s0/low
VOLS=$(ls $D/train/*_log1p.npy); CLEANS=$(ls $D/train/*_clean.npy)
# 相位臂（有监督）
$PY -u train.py --train-volumes $VOLS --train-clean-volumes $CLEANS \
  --patch 32 64 64 --epochs 1 --steps-per-epoch 1 --features 16 --seed 3407 \
  --out runs/v3_round3/audit_random_clean --lr-schedule poly --pairing within \
  --mode random --audit-loss-grads --audit-seed 1234 --arm-label random_clean
# dense 臂：把 --mode random 换成 --mode dense，其余逐字相同
# 判定只看 stdout 的 [audit] 行与 loss_grad_audit_*.json；epochs/steps 用 1 是因为
# --audit-loss-grads 在进入训练循环之前就 return 0，不会训练。
$PY -m pytest tests/test_v3_round3_geometry.py tests/test_v3_round3_dense_identity.py -q
```

---

## 1. 监督几何：相位臂 ≠ 同位置（R3-P04）

### 1.1 结论

`--train-clean-volumes` 只是把**目标的来源**从"带噪 patch"换成"无噪 patch"，
采样几何完全不变（同一个 `torch.Generator` 状态跑两遍）。因此：

- **相位臂**（`random` / `fixed` / `guided` …）：`xa,xb` 与 `ya,yb` 是**四个互不重叠**的体素集合，各占每个 2×2×2 块的一个角点，覆盖率 4/8。目标落在**另外两个角点**上 —— 这是**跨位置**监督，不是"同位置去噪"。
- **dense 臂**：四个视图都是同一张满分辨率 patch ⇒ 真正的同位置监督。

坐标编码体（每个体素的值 = 它自己的编号）把这件事直接读出来：

| 检查 | 结果 |
|---|---|
| 相位臂四个视图两两不交 | 通过（15 项测试中的 `test_phase_views_are_disjoint_position_sets`） |
| 并集 = 每块 4 个角、每块恰好 4 个 | 通过 |
| 固定 tet 臂：每个视图内部 `(z%2,y%2,x%2)` 恒定（确为一个全局相位） | 8/8 pattern 通过 |
| 忠实 VALID 臂：**逐块**随机抽模式与角色，所以单个视图**不是**全局相位（8 种奇偶组合都出现） | 通过 |
| 单点冲激在相位臂最多命中 1 个视图；在 dense 臂命中 4 个 | 通过 |
| dense 四视图逐体素等于输入 patch | 通过 |
| dense 不给`--train-clean-volumes` ⇒ 目标就是输入（平凡恒等） | 通过（必须禁止） |

第 4 条是我一开始搞错的边界：我原以为"视图 = 一个相位"对所有臂都成立，测试当场抓出
忠实 VALID 臂是**逐块**相位随机的。两种臂"输入与目标不重叠"都成立，但只有固定/步距臂能叫"一个相位"。

### 1.2 对结论表述的约束

- 相位臂上的 `n2n` 读数应表述为"**跨位置**监督训练出的网络，在部署几何（满分辨率滑窗）下的表现"，不能表述为"同位置去噪能力"。
- 只有当 dense 臂也给出正结果时，才能说"同位置监督下网络能学到去噪"。
- 两者不可混谈，报告里必须标出是哪一种。

---

## 2. 损失项的实际梯度（R3-P06）

### 2.1 实测（patch 32×64×64，features 16，seed 3407，未训练的网络，`--audit-seed 1234`）

| 臂 | `inputs_identical` | loss: neighbor | loss: reg | loss: identity | ‖∇‖: neighbor | ‖∇‖: reg | ‖∇‖: identity | identity 占 neighbor 梯度 |
|---|---|---|---|---|---|---|---|---|
| dense + 干净目标 | **True** | 2.331e1 | 3.619e1 | **0.000e0** | 1.239e3 | 1.650e3 | **0.000e0** | **0.000%** |
| random + 干净目标 | False | 3.027e1 | 1.720e2 | 5.825e-2 | 1.634e3 | 7.053e3 | 1.886e0 | 0.115% |
| random + 第二观测（N2N） | False | 3.032e1 | 1.720e2 | 5.829e-2 | 1.639e3 | 7.049e3 | 1.869e0 | 0.114% |
| fixed + 干净目标 | False | 3.022e1 | 1.719e2 | 6.361e-1 | 1.618e3 | 6.962e3 | 2.414e1 | 1.49% |

### 2.2 判读

1. **dense 臂上 identity 项是精确的空操作**：`inputs_identical=True` ⇒ `ha == hb` 逐位相同 ⇒ loss 与梯度都是 0.0（不是"很小"，是 0）。`--weight-identity` 取 0 还是取 1，训练轨迹逐位相同。所以 **"dense + 关掉 identity"这个消融作废**，从 B3 里删除。
2. **相位臂上它非零但可忽略**：0.115%（random）/ 1.49%（fixed）。低于种子间波动一个量级以上，任何"开/关它"的差异都会被噪声吞掉。若将来要研究 identity 项，必须造一个**两路输入确实不同**的受控双视图（同一 patch 加两次独立噪声），否则测不到。
3. **Hessian/DWT 正则的原始梯度比 neighbor 还大**（1.65e3–7.05e3 vs 1.24e3–1.63e3），是 `weight_reg = 1e-4` 把它压到 0.013% 的。这条对 B3 的含义：系数的小幅调整仍然在噪声里；要让正则真正起作用，需要改动数量级，而那种改动会直接威胁弱血管结构 —— 所以**先不要动**，等结构指标（弱血管召回@固定 FPR、连通性、伪血管数）能测出来再说。
4. **未训练网络的输出均值偏移很大**：`mean(out − in)` = −2.35（dense）/ −3.20（random）/ −3.17（fixed），而输入本身是 `log1p` 域、背景在 0 附近。这与 A1 要查的"直流/电平丢失"是同一现象的初值版本，但**不能**作为证据：这是随机初始化、一个种子、一个 patch 的读数，且减掉了 GroupNorm 之后特征的整体平移。A1 要在 in-distribution 与 OOD 两档上分别量，并报电平曲线。

### 2.3 与旧结论的关系

旧报告里"`--weight-identity` 的消融结果"一律撤回；`reports/v3_round2/` 下的历史数值不删，但引用时必须注明"该项在 dense 臂上是空操作，读数无意义"。

---

## 3. 限制

1. 审计用的是**未训练**网络的梯度。训练后的梯度量级会变，但"dense 臂 identity = 0"是**结构性的**（同一输入 ⇒ 同一输出），与训练无关，这条结论不会变。
2. `--audit-seed 1234` 只给一个 patch／一个起点。相位臂的 identity 占比依赖抽到的模式（fixed 臂明显更大），所以给的是量级，不是精确比例。
3. `reg` 的梯度里包含 DWT 的高低频带组合，本审计只报总范数，没有分频带拆开（A4 的频谱分解会做）。
4. 几何测试用的是 8³ 的合成体，与真实的 32×64×64 只在"是否重叠"这一点上等价；重叠与否不随尺寸改变。

---

## 4. 通俗总结

这一节回答两个很具体的问题：训练时"网络看到的输入"和"它被要求模仿的目标"是不是同一批像素？以及那个"让两路输出互相靠拢"的附加损失项，到底有没有在起作用？

第一个问题：在做相位抽取的那几种训练方式里，四个视图是同一个 2×2×2 小块里**四个不同的角**。所以输入用的是其中两个角，目标来自另外两个角 —— 网络是在"看着这几根像素、猜那几根像素"，**不是**在同一个位置上做去噪。只有把训练改成满分辨率的那一种方式，输入和目标才真的落在同一批像素上。这两件事说明的问题不一样，以后不能再混着说。

第二个问题：我们量了每一项损失对网络参数的实际推力。结论很干脆 —— 在满分辨率那种训练方式下，那个"两路输出要互相靠拢"的项恒等于零，一点力都没有，因为两路输入本来就是同一张图。换句话说，我们之前准备拿它做的对照，实际上什么都没对照。在相位抽取的方式下它确实非零，但只有主损失的千分之一到百分之一，开关它看到的差别全是随机波动。

顺带量到：那个"平滑正则"项的原始推力其实比主损失还大，只是被一个很小的系数压住了；未训练的网络会把输出的整体亮度往下拉两三个单位。前者说明想靠微调正则系数解决问题是走不通的，后者是下一步要专门查的"亮度/电平"问题的初值版本。

现在可以说的：我们知道哪几个开关是真的、哪几个是假的，以后的对照不会再建在假开关上。
现在还不能说的：这些损失项应该设成多少 —— 那要等结构的量化指标能测出来。
