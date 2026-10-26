"""VALID-compatible Tetris sampling plus an SVD-guided selector.

The sampling geometry is adapted from FDU-donglab/VALID v1.0.
Only the pattern selection policy is new in this repository.

Two samplers are provided:

* :func:`valid_random_tetris` -- a faithful port of the original VALID
  sampler: every 2x2x2 block independently draws one of the eight Tetris
  patterns and randomly assigns its four sampled positions to the four
  subvolume roles. Use this for the baseline arm.
* :func:`guided_tetris_subvolumes` -- our method: one pattern per patch,
  selected from SVD-derived signal/noise statistics.
"""

import torch


TETRIS_PATTERNS = torch.tensor(
    [
        [0, 1, 2, 4],
        [0, 1, 3, 5],
        [1, 2, 3, 7],
        [0, 2, 3, 6],
        [0, 4, 5, 6],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
        [3, 5, 6, 7],
    ],
    dtype=torch.long,
)


def index_offset(index):
    return index // 4, (index // 2) % 2, index % 2


def one_phase(volume, index):
    dz, dy, dx = index_offset(index)
    D, H, W = volume.shape
    return volume[
        dz:D - (1 - dz):2,
        dy:H - (1 - dy):2,
        dx:W - (1 - dx):2,
    ]


def all_phases(volume):
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D,H,W].")
    D, H, W = volume.shape
    if D % 2 or H % 2 or W % 2:
        raise ValueError("All spatial dimensions must be even.")
    return [one_phase(volume, i) for i in range(8)]


# --------------------------------------------------------------- V3 采样 ----
# 为什么需要下面的东西（2026-10-20，接训练时发现）：
#   8 个 tet 每个都恰好从 4 组"对顶角对"（{0,7},{1,6},{2,5},{3,4}）里各取一个，
#   所以**任何一个 tet 内部都不存在体对角对**（逐模式枚举为空，见
#   scripts/v3_phase3_probe.py 的自检）。而 V3 预注册的首选角色对正是体对角
#   （reports/v3_phase2_role_order.md §4.1）。结论：要落地预注册的角色序，
#   输入必须取 tet A、目标必须取它的互补 tet B —— 这也正是
#   reports/phase2_pair_statistics.md 记录的 VALID 自己的配对
#   （"两组互补体素"）。原来 `valid_random_tetris` 只返回同一个 tet 的 4 个视图，
#   输入与目标都在 tet A 内，**实现不了预注册的首选**，所以这里补 cross 配对。

COMPLEMENT_PATTERN = torch.tensor(
    [7, 6, 4, 5, 2, 3, 1, 0], dtype=torch.long
)  # COMPLEMENT_PATTERN[p] == the tet complementary to TETRIS_PATTERNS[p]


def strided_phase(volume, index, stride=1):
    """一个相位（tet 内的一个角点）在采样步距 `stride` 下的稠密视图。

    stride=1 时与 :func:`one_phase` 逐体素相同（要求各轴为偶数）。
    stride=m 时，一个 (2m)^3 的块里只有 8 个角点被采到，该相位每轴取
    `m*o, m*o+2m, m*o+4m, ...`。要求各轴能被 2m 整除，否则 8 个相位不等长。
    """
    x = torch.as_tensor(volume).float()
    if stride < 1:
        raise ValueError("stride must be >= 1")
    step = 2 * stride
    if any(n % step for n in x.shape):
        raise ValueError(
            f"每个轴都必须能被 2*stride={step} 整除，实际形状 {tuple(x.shape)}"
        )
    dz, dy, dx = index_offset(index)
    return x[stride * dz::step, stride * dy::step, stride * dx::step]


def strided_views(volume, slots, stride=1):
    """按显式相位槽位抽 4 个视图（`slots` 是相位编号 0..7，依次给 xa,xb,ya,yb）。"""
    phases = [strided_phase(volume, i, stride) for i in range(8)]
    return [phases[i] for i in slots]


def check_cross_slots(slots):
    """cross 配对的合法性：前两个槽位同属一个 tet，后两个同属它的互补 tet。"""
    if len(slots) != 4 or sorted(slots) != sorted(set(slots)):
        raise ValueError(f"--slots 必须是 4 个互不相同的相位编号，实际 {slots}")
    if not set(slots) < set(range(8)):
        raise ValueError(f"--slots 取值必须是 0..7，实际 {slots}")
    src, tgt = set(slots[:2]), set(slots[2:])
    p = next((i for i, pat in enumerate(TETRIS_PATTERNS.tolist()) if src <= set(pat)), None)
    if p is None:
        raise ValueError(f"输入槽位 {sorted(src)} 不属于同一个 tet")
    comp = set(range(8)) - set(TETRIS_PATTERNS[p].tolist())
    if not tgt <= comp:
        raise ValueError(
            f"目标槽位 {sorted(tgt)} 必须落在输入 tet {TETRIS_PATTERNS[p].tolist()} "
            f"的互补 tet {sorted(comp)} 内"
        )
    return p


def valid_random_tetris_strided(volume, stride=1, generator=None):
    """与原版随机 Tetris 同样的抽样规则，但支持采样步距 m，且四种角色取自同一 tet。

    stride=1 时与 :func:`valid_random_tetris` **逐体素相同**（同样的 RNG 消耗顺序），
    这一点由 tests/test_smoke.py 的回归检查保证。
    注意这是"同 tet 配对"，**用不了体对角角色对**；cross 配对见
    :func:`valid_random_tetris_cross`。
    """
    x = torch.as_tensor(volume).float()
    if x.ndim != 3:
        raise ValueError("volume must have shape [D,H,W].")
    step = 2 * stride
    if any(n % step for n in x.shape):
        raise ValueError(f"每个轴都必须能被 2*stride={step} 整除")
    phases = [strided_phase(x, i, stride) for i in range(8)]
    view_shape = phases[0].shape
    nb = 1
    for n in view_shape:
        nb *= int(n)

    k = torch.randint(0, len(TETRIS_PATTERNS), (nb,), generator=generator)
    chosen = TETRIS_PATTERNS[k]                  # (nb, 4)
    perm = torch.rand(nb, 4, generator=generator).argsort(dim=1)

    zz, yy, xx = torch.meshgrid(
        *[torch.arange(n) for n in view_shape], indexing="ij"
    )
    stack = torch.stack(phases)                  # (8, *view_shape)
    subs = []
    for i in range(4):
        corner = chosen.gather(1, perm[:, i:i + 1]).squeeze(1).reshape(view_shape)
        subs.append(stack[corner, zz, yy, xx])
    return subs


def valid_random_tetris_cross(volume, stride=1, generator=None):
    """VALID 的互补配对 + 随机角色：输入取随机 tet 的两个角，目标取其互补 tet 的两个角。

    与 :func:`valid_random_tetris_strided` 的区别**只在目标从哪来**：这里目标来自与
    输入互补的那半个块，所以体对角角色对是可达的（它是 V3 预注册的首选）。
    覆盖率和 within 配对一样是 4/8（每块 2 个输入角 + 2 个目标角），
    两者采到的体素总数相同，区别是"目标的 2 个体素落在哪半个块里"。
    """
    x = torch.as_tensor(volume).float()
    if x.ndim != 3:
        raise ValueError("volume must have shape [D,H,W].")
    step = 2 * stride
    if any(n % step for n in x.shape):
        raise ValueError(f"每个轴都必须能被 2*stride={step} 整除")
    phases = [strided_phase(x, i, stride) for i in range(8)]
    view_shape = phases[0].shape
    nb = 1
    for n in view_shape:
        nb *= int(n)

    k = torch.randint(0, len(TETRIS_PATTERNS), (nb,), generator=generator)
    q = COMPLEMENT_PATTERN[k]
    # 输入 tet 的 4 个角点里随机取 2 个（有序），目标在互补 tet 里同样取 2 个
    rin = torch.rand(nb, 4, generator=generator).argsort(dim=1)
    rtg = torch.rand(nb, 4, generator=generator).argsort(dim=1)
    src_in = TETRIS_PATTERNS[k].gather(1, rin[:, :2])
    src_tg = TETRIS_PATTERNS[q].gather(1, rtg[:, :2])

    zz, yy, xx = torch.meshgrid(
        *[torch.arange(n) for n in view_shape], indexing="ij"
    )
    stack = torch.stack(phases)
    subs = []
    for c in (src_in[:, 0], src_in[:, 1], src_tg[:, 0], src_tg[:, 1]):
        subs.append(stack[c.reshape(view_shape), zz, yy, xx])
    return subs


def _space_to_depth(x, block_size=2):
    """(1,D,H,W) -> (D/2,H/2,W/2,8); last dim orders positions as dz*4+dy*2+dx."""
    n, d, h, w = x.shape
    x = x.view(n, d // block_size, block_size, h // block_size, block_size, w // block_size, block_size)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return x.view(d // block_size, h // block_size, w // block_size, block_size ** 3)


def valid_random_tetris(volume, generator=None):
    """Faithful port of VALID v1.0 random Tetris sampling.

    Each 2x2x2 block draws one of the eight patterns uniformly at random,
    and the four sampled positions are assigned to the four subvolume
    roles through a random permutation, exactly as the upstream
    generate_mask_pair / RandomSampler_0123 combination does.

    Returns four tensors of shape (D/2, H/2, W/2).
    """
    x = torch.as_tensor(volume).float()
    if x.ndim != 3:
        raise ValueError("volume must have shape [D,H,W].")
    d, h, w = x.shape
    if d % 2 or h % 2 or w % 2:
        raise ValueError("All spatial dimensions must be even.")
    blocks = _space_to_depth(x[None])            # (nb_d, nb_h, nb_w, 8)
    nb = blocks.shape[0] * blocks.shape[1] * blocks.shape[2]
    blocks = blocks.reshape(nb, 8)

    k = torch.randint(0, len(TETRIS_PATTERNS), (nb,), generator=generator)
    chosen = TETRIS_PATTERNS[k]                  # (nb, 4) positions per block
    offsets = torch.arange(nb).unsqueeze(1) * 8
    flat = blocks.reshape(-1)                    # (nb*8,)

    # Random permutation per block: sub i receives pattern slot perm[i].
    perm = torch.rand(nb, 4, generator=generator).argsort(dim=1)

    subs = []
    for i in range(4):
        take = offsets + chosen.gather(1, perm[:, i:i + 1])
        subs.append(flat.take(take.squeeze(1)).reshape(d // 2, h // 2, w // 2))
    return subs


def corr(a, b, eps=1e-6):
    a = torch.as_tensor(a).flatten().float()
    b = torch.as_tensor(b).flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.square().mean().sqrt() * b.square().mean().sqrt() + eps)


def pair_quality_score(signal_views, noise_views):
    s, n = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            s.append(corr(signal_views[i], signal_views[j]))
            n.append(abs(corr(noise_views[i], noise_views[j])))
    return torch.stack(s).mean(), torch.stack(n).mean()


def zscore(x, eps=1e-6):
    return (x - x.mean()) / (x.std() + eps)


def select_guided_pattern(signal_volume, noise_volume):
    """Select the best of the eight VALID pattern sets.

    Quality = standardized structural similarity - standardized noise similarity.
    No trainable parameter is introduced.
    """
    signal_phases = all_phases(signal_volume)
    noise_phases = all_phases(noise_volume)

    stats = []
    for pattern in TETRIS_PATTERNS.tolist():
        sv = [signal_phases[i] for i in pattern]
        nv = [noise_phases[i] for i in pattern]
        s, n = pair_quality_score(sv, nv)
        stats.append(torch.stack([s, n]))

    stats = torch.stack(stats)
    q = zscore(stats[:, 0]) - zscore(stats[:, 1])
    best = int(torch.argmax(q).item())
    return best, q.detach().cpu()


def tetris_from_pattern(volume, pattern_id):
    phases = all_phases(volume)
    return [phases[i] for i in TETRIS_PATTERNS[pattern_id].tolist()]


def random_tetris_subvolumes(volume):
    """Single-pattern-per-patch random sampling (kept for ablation only).

    This is NOT the original VALID sampler; use :func:`valid_random_tetris`
    for the faithful baseline.
    """
    pattern_id = int(torch.randint(0, len(TETRIS_PATTERNS), ()).item())
    return pattern_id, tetris_from_pattern(volume, pattern_id)


def guided_tetris_subvolumes(volume, signal_volume, noise_volume):
    pattern_id, diagnostics = select_guided_pattern(signal_volume, noise_volume)
    return pattern_id, tetris_from_pattern(volume, pattern_id), diagnostics
