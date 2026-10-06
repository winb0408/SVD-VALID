# Adapted from FDU-donglab/VALID v1.0.
# Upstream: https://github.com/FDU-donglab/VALID/tree/v1.0
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE3DBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.shape
        z = self.pool(x).view(b, c)
        z = self.gate(self.fc2(self.act(self.fc1(z))))
        return x * z.view(b, c, 1, 1, 1)


def block(cin, cout, groups):
    return nn.Sequential(
        nn.ReflectionPad3d(1),
        nn.Conv3d(cin, cout, 3),
        SE3DBlock(cout),
        nn.GroupNorm(groups, cout, eps=1e-5, affine=False),
        nn.ELU(inplace=True),
        nn.ReflectionPad3d(1),
        nn.Conv3d(cout, cout, 3),
        SE3DBlock(cout),
        nn.GroupNorm(groups, cout, eps=1e-5, affine=False),
        nn.ELU(inplace=True),
    )


class CRN(nn.Module):
    """Compact Cross-scale Recursive Network used by VALID."""

    def __init__(self, in_channels=1, out_channels=1, n_features=16, n_groups=4):
        super().__init__()
        f = n_features

        self.conv_in = nn.Conv3d(in_channels, f, 1)

        self.d1 = nn.Sequential(block(f, 8 * f, n_groups), nn.AvgPool3d(8))
        self.d2 = nn.Sequential(block(f, 4 * f, n_groups), nn.AvgPool3d(4))
        self.d3 = nn.Sequential(block(f, 2 * f, n_groups), nn.AvgPool3d(2))

        self.u1 = block(12 * f, 4 * f, n_groups)
        self.u2 = block(6 * f, 2 * f, n_groups)
        self.uu1 = block(8 * f, 2 * f, n_groups)

        self.b1 = block(3 * f, f, n_groups)
        self.b2 = block(3 * f, f, n_groups)
        self.b3 = block(3 * f, f, n_groups)

        self.out = nn.Conv3d(f, out_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def up(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x

    def forward(self, x):
        x0 = self.conv_in(x)
        d1 = self.d1(x0)
        d2 = self.d2(x0)
        d3 = self.d3(x0)

        d1u = self.up(d1, d2)
        d2u = self.up(d2, d3)
        d3u = self.up(d3, x0)

        u1 = self.u1(torch.cat([d2, d1u], 1))
        u2 = self.u2(torch.cat([d3, d2u], 1))

        u1u = self.up(u1, u2)
        u2f = self.up(u2, x0)
        uff = self.uu1(torch.cat([d3, u1u, u2], 1))
        uff = self.up(uff, x0)

        y = self.b1(torch.cat([x0, d3u], 1))
        y = self.b2(torch.cat([y, u2f], 1))
        y = self.b3(torch.cat([y, uff], 1))
        return self.out(y)
