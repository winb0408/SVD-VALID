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
