"""VALID-compatible Tetris sampling plus an SVD-guided selector.

The sampling geometry is adapted from FDU-donglab/VALID v1.0.
Only the pattern selection policy is new in this repository.
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


def corr(a, b, eps=1e-6):
    a = a.flatten().float()
    b = b.flatten().float()
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
    pattern_id = int(torch.randint(0, len(TETRIS_PATTERNS), ()).item())
    return pattern_id, tetris_from_pattern(volume, pattern_id)


def guided_tetris_subvolumes(volume, signal_volume, noise_volume):
    pattern_id, diagnostics = select_guided_pattern(signal_volume, noise_volume)
    return pattern_id, tetris_from_pattern(volume, pattern_id), diagnostics
