import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_CRN import CRN


class HessianConstraintLoss3D(nn.Module):
    """VALID v1.0-style 3-D Hessian regularization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected [B,C,D,H,W].")
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")

        dz = x[:, :, 1:-1, 1:-1, 2:] - x[:, :, 1:-1, 1:-1, :-2]
        dx = x[:, :, 1:-1, 2:, 1:-1] - x[:, :, 1:-1, :-2, 1:-1]
        dy = x[:, :, 2:, 1:-1, 1:-1] - x[:, :, :-2, 1:-1, 1:-1]

        dzz = dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :]
        dxx = dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]
        dyy = dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1]

        dzx = dz[:, :, :, 1:, :-1] - dz[:, :, :, :-1, 1:]
        dzy = dz[:, :, 1:, :, :-1] - dz[:, :, :-1, :, 1:]
        dxy = dx[:, :, 1:, :, :-1] - dx[:, :, :-1, :, 1:]

        nz = min(t.shape[2] for t in (dzz, dxx, dyy, dzx, dzy, dxy))
        nh = min(t.shape[3] for t in (dzz, dxx, dyy, dzx, dzy, dxy))
        nw = min(t.shape[4] for t in (dzz, dxx, dyy, dzx, dzy, dxy))
        dzz, dxx, dyy, dzx, dzy, dxy = [
            t[:, :, :nz, :nh, :nw] for t in (dzz, dxx, dyy, dzx, dzy, dxy)
        ]

        return (
            dzz.square()
            + dxx.square()
            + dyy.square()
            + 2.0 * dzx.square()
            + 2.0 * dzy.square()
            + 2.0 * dxy.square()
        ).mean()


class NetworkCNR(nn.Module):
    """VALID CRN wrapper."""

    def __init__(self, in_channels=1, out_channels=1, features=16, groups=4):
        super().__init__()
        self.net = CRN(
            in_channels=in_channels,
            out_channels=out_channels,
            n_features=features,
            n_groups=groups,
        )

    def forward(self, x):
        return self.net(x)
