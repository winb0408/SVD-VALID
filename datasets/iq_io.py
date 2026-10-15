"""Read the Dryad beamformed complex IQ volumes.

`Data_{SA64,HSA64,PW3232,PW1616}.mat` are MATLAB v7.3 (HDF5) files.  h5py
returns them with the axis order reversed relative to the documented MATLAB
layout, and complex values as compound `real`/`imag` fields; both are handled
here.  The returned array is `[Z, X, Y, T]` (and `[..., EmittingArray]` when the
source has more than one emitting array), complex64 when the source is complex.

Flatten-order note
------------------
The volume is flattened with ``x.reshape(-1, T)``, i.e. C order (Y fastest).
MATLAB's ``reshape(...,[V,T])`` is column-major (Z fastest), so the *row
permutation* of the Casorati matrix differs from the official script.  This is
harmless for the SVD blood-band filter: a row permutation is a similarity
transform, so the retained modes and the resulting per-voxel power volume are
identical as long as the forward and inverse reshapes use the same convention,
which they do throughout this repository.  The X/Y axis assignment cannot be
resolved from the header alone (both have length 64); it only affects display
orientation, not any measurement.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _compound_to_complex(raw):
    """Turn an h5py compound `real`/`imag` array into a numpy complex array."""
    if raw.dtype.names is None:
        return raw
    fields = set(raw.dtype.names)
    if not {"real", "imag"} <= fields:
        raise ValueError(f"Unsupported compound dtype fields: {raw.dtype.names}")
    return (raw["real"] + 1j * raw["imag"]).astype(np.complex64)


def load_iq(path, variable=None):
    """Load a beamformed IQ volume as `[Z, X, Y, T]` (or `[Z, X, Y, T, Array]`)."""
    path = Path(path)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat source, got {path}")
    try:
        mat = loadmat(str(path))
        if variable is None:
            keys = [k for k in mat if not k.startswith("__")]
            if len(keys) != 1:
                raise ValueError(f"Pass variable=; found variables: {keys}")
            variable = keys[0]
        return np.asarray(mat[variable])
    except NotImplementedError:
        import h5py

        with h5py.File(str(path), "r") as f:
            if variable is None:
                keys = [k for k in f.keys() if not k.startswith("__")]
                if len(keys) != 1:
                    raise ValueError(f"Pass variable=; found variables: {keys}")
                variable = keys[0]
            raw = np.asarray(f[variable][:])
    raw = _compound_to_complex(raw)
    # h5py reverses the MATLAB axis order.
    return np.transpose(raw, axes=tuple(range(raw.ndim))[::-1])
