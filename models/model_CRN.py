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


def conv_pair(cin, hidden, cout, groups):
    return nn.Sequential(
        nn.ReflectionPad3d(1),
        nn.Conv3d(cin, hidden, 3),
        SE3DBlock(hidden),
        nn.GroupNorm(groups, hidden, eps=1e-5, affine=False),
        nn.ELU(inplace=True),
        nn.ReflectionPad3d(1),
        nn.Conv3d(hidden, cout, 3),
        SE3DBlock(cout),
        nn.GroupNorm(groups, cout, eps=1e-5, affine=False),
        nn.ELU(inplace=True),
    )


class CRN(nn.Module):
    """Channel-faithful port of the VALID v1.0 CRN."""

    def __init__(self, in_channels=1, out_channels=1, n_features=16, n_groups=4):
        super().__init__()
        f = n_features
        self.conv_in = nn.Conv3d(in_channels, f, 1)
        self.conv_d1 = nn.Sequential(conv_pair(f, 8*f, 8*f, n_groups), nn.AvgPool3d(8))
        self.conv_d2 = nn.Sequential(conv_pair(f, 4*f, 4*f, n_groups), nn.AvgPool3d(4))
        self.conv_d3 = nn.Sequential(conv_pair(f, 2*f, 2*f, n_groups), nn.AvgPool3d(2))
        self.conv_u1 = conv_pair(12*f, 4*f, 4*f, n_groups)
        self.conv_u2 = conv_pair(6*f, 2*f, 2*f, n_groups)
        self.conv_uu1 = conv_pair(8*f, 2*f, 2*f, n_groups)
        self.conv_b1 = conv_pair(3*f, 2*f, f, n_groups)
        self.conv_b2 = conv_pair(3*f, 2*f, f, n_groups)
        self.conv_b3 = conv_pair(3*f, 2*f, f, n_groups)
        self.out = nn.Conv3d(f, out_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def up(source, target):
        if source.shape[2:] != target.shape[2:]:
            return F.interpolate(source, size=target.shape[2:], mode="trilinear", align_corners=False)
        return source

    def forward(self, x):
        x = self.conv_in(x)
        d1, d2, d3 = self.conv_d1(x), self.conv_d2(x), self.conv_d3(x)
        d1f, d2f, d3f = self.up(d1, d2), self.up(d2, d3), self.up(d3, x)
        u1 = self.conv_u1(torch.cat([d2, d1f], dim=1))
        u2 = self.conv_u2(torch.cat([d3, d2f], dim=1))
        u1f, u2f = self.up(u1, u2), self.up(u2, x)
        uf = self.conv_uu1(torch.cat([d3, u1f, u2], dim=1))
        uf = self.up(uf, x)
        y = self.conv_b1(torch.cat([x, d3f], dim=1))
        y = self.conv_b2(torch.cat([y, u2f], dim=1))
        y = self.conv_b3(torch.cat([y, uf], dim=1))
        return self.out(y)
