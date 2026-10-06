# SVD-VALID

**SVD-guided Volumetric Orthogonal Self-Supervised Denoising for Ultrasound**

This repository is an initial research implementation that adapts the public VALID v1.0 framework to volumetric ultrasound. The initial version deliberately keeps the VALID CRN backbone and original three-term loss unchanged. The first methodological modification is the replacement of random Tetris sampling with an SVD-guided Tetris selector.

## Research hypothesis

For an ultrafast ultrasound volume, SVD provides a data-dependent decomposition of the slow-time signal into subspaces associated with clutter/blood/noise. We use these subspace-derived signal and noise maps only to select better self-supervised orthogonal views:

Q(k) = Z(S(k)) - Z(N(k))

where S(k) is cross-view structural similarity and N(k) is residual-noise similarity for Tetris pattern k. The denoising network and loss remain unchanged.

This is intentionally a sampling-first, loss-minimal baseline. Later extensions can add anisotropic/PSF-aware constraints only after the sampling hypothesis is experimentally validated.

## Relationship to prior work

The implementation is derived from:

- Gu et al., Science Advances (2026), "Enhancing biomedical optical volumetric imaging via self-supervised orthogonal learning" (VALID).
- Zhang et al., 2026 preprint, "SVD-SN2N: Self-Inspired Noise2Noise Learning for Denoising Log-Compressed SVD-Filtered Ultrasound Imaging."

The VALID v1.0 code is reused for the CRN architecture, 3-D Tetris pattern definitions and the Hessian regularizer. See THIRD_PARTY_LICENSES.md.

## Current v0.1 scope

1. Convert a beamformed ultrafast ultrasound time series (.mat / .npy) into an SVD-filtered power-Doppler volume.
2. Estimate SVD signal/noise energy maps.
3. Apply log compression.
4. Generate four VALID-style Tetris subvolumes.
5. Select Tetris patterns using SVD-derived structural/noise scores.
6. Train the original VALID CRN with the original loss.
7. Provide random-Tetris and SVD-guided-Tetris modes under the same network and loss.

## Recommended first dataset

Dryad 10.5061/dryad.47d7wm3qg is the preferred first validation target. It contains open 3-D RCA/CMUT ultrasound acquisitions as beamformed volumes with 37 x 64 x 64 voxels and 400/800 frames, and the supplied MATLAB script itself applies SVD clutter filtering to form power-Doppler volumes. See docs/datasets.md.

## Quick start

### 1. Environment

Python 3.9+ and a CUDA-enabled PyTorch installation are recommended.

    conda create -n svd-valid python=3.10
    conda activate svd-valid
    pip install -r requirements.txt

Install a CUDA-compatible PyTorch build separately if required.

### 2. Convert and SVD-filter a public ultrasound acquisition

    python scripts/prepare_svd.py --input /path/to/Data_PW3232.mat --variable Data_PW3232 --emitting-array 0 --frames 400 --rank-low 10 --rank-high 80 --output data/processed/pw3232

The script writes:

- power_doppler.npy — log-domain training volume source.
- signal_map.npy — SVD retained-signal energy map.
- noise_map.npy — high-rank residual energy map.
- metadata.json — shape and SVD parameters.

### 3. Train baseline VALID-style random Tetris

    python train.py --train-volume data/processed/pw3232/power_doppler.npy --patch 32 64 64 --epochs 50 --steps-per-epoch 200 --mode random

### 4. Train SVD-guided Tetris

    python train.py --train-volume data/processed/pw3232/power_doppler.npy --signal-map data/processed/pw3232/signal_map.npy --noise-map data/processed/pw3232/noise_map.npy --patch 32 64 64 --epochs 50 --steps-per-epoch 200 --mode guided

The two runs intentionally share the same CRN, optimizer, learning-rate schedule and loss. This makes the sampling contribution easier to attribute.

## Important experimental rule

Do not add new loss terms in v0.1. First test whether random Tetris -> SVD-guided Tetris reduces pairwise noise correlation and improves volumetric restoration while all optimization terms remain fixed.

## Repository layout

    SVD-VALID/
    ├── datasets/
    │   ├── __init__.py
    │   └── sampling.py
    ├── models/
    │   ├── __init__.py
    │   ├── model_CRN.py
    │   └── network.py
    ├── scripts/
    │   └── prepare_svd.py
    ├── docs/
    │   └── datasets.md
    ├── THIRD_PARTY_LICENSES.md
    ├── requirements.txt
    └── train.py
