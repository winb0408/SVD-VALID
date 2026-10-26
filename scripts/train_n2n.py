"""中心可见的 N2N 训练循环（方案②）。由 `scripts/v3_phase4_train_n2n.py` 调用。

输入与目标是**同一场景两次独立噪声实现**的 log 体积；网络可以看到输入的中心体素，
因此不会退化成"输出=输入"（目标是另一次实现，不是输入自己）。

    residual=0:  out = f(x)
    residual=1:  out = x + f(x)

checkpoint 里写 `"infer": "full"`，让 `v3_phase4_scoreboard.py` 走整卷前向，
而不是按相位散回体素。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.network import NetworkCNR  # noqa: E402


def rand_patch(vol, patch, rng):
    sl = []
    for n, p in zip(vol.shape, patch):
        if p >= n:
            sl.append(slice(0, n))
        else:
            o = int(rng.integers(0, n - p + 1))
            sl.append(slice(o, o + p))
    return tuple(sl)


def masked_mse(pred, tgt, m):
    return (torch.sum((pred - tgt) ** 2 * m) / torch.sum(m).clamp_min(1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, nargs="+",
                    help="观测 A 的 log 体积 .npy（可给多个场景，逐步随机取一个）")
    ap.add_argument("--target", required=True, nargs="+", help="观测 B 的 log 体积 .npy")
    ap.add_argument("--valid", required=True, nargs="+",
                    help="有效体素掩膜 .npy；损失只在有效区里算（零填充区会把损失带偏 10 倍）")
    ap.add_argument("--val-input", default=None, help="留出场景的 A（用于早停，不进训练）")
    ap.add_argument("--val-target", default=None, help="留出场景的 B")
    ap.add_argument("--val-valid", default=None, help="留出场景的掩膜")
    ap.add_argument("--val-every", type=int, default=10, help="每隔多少 epoch 在留出场景上算一次")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--residual", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xs = [np.load(p).astype(np.float32) for p in args.input]
    ys = [np.load(p).astype(np.float32) for p in args.target]
    vs = [np.load(p).astype(np.float32) for p in args.valid]
    assert len(xs) == len(ys) == len(vs), (len(xs), len(ys), len(vs))
    for x, y, m in zip(xs, ys, vs):
        assert x.shape == y.shape == m.shape, (x.shape, y.shape, m.shape)
    assert all(a.shape == xs[0].shape for a in xs), "多场景必须同形状"

    val = None
    if args.val_input:
        val = (np.load(args.val_input).astype(np.float32),
               np.load(args.val_target).astype(np.float32),
               np.load(args.val_valid).astype(np.float32))

    # 恒等基准：任何"去噪器"至少必须优于它，否则等于把图搞坏
    with torch.no_grad():
        xt = torch.as_tensor(xs[0])[None, None]
        yt = torch.as_tensor(ys[0])[None, None]
        mt = torch.as_tensor(vs[0])[None, None]
        ident = float(masked_mse(xt, yt, mt))
    print(f"场景数={len(xs)}  恒等基准（预测=输入A）valid 区 MSE = {ident:.6f}", flush=True)

    model = NetworkCNR(features=args.features, groups=args.groups).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps_per_epoch)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    history = []
    t0 = time.time()
    best = {"ratio": float("inf"), "epoch": 0}
    cargs = dict(features=args.features, groups=args.groups, stride=1, mode="n2n",
                 pairing="n2n", epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                 lr=args.lr, patch=args.patch, seed=args.seed,
                 weight_identity=0.0, weight_reg=0.0, residual=args.residual,
                 infer="full", n_scenes=len(xs))

    @torch.no_grad()
    def validate(ep):
        """留出场景上的整卷比值。低于 0.615 只可能是记住了那次实现自己的噪声。"""
        if val is None:
            return None
        model.eval()
        xv, yv, mv = val
        xin = torch.as_tensor(xv, device=device)[None, None].float()
        h = model(xin)
        pred = xin + h if args.residual else h
        m = torch.as_tensor(mv, device=device)[None, None].float()
        y0 = torch.as_tensor(yv, device=device)[None, None].float()
        ratio = float(masked_mse(pred, y0, m) / masked_mse(xin, y0, m).clamp_min(1e-9))
        model.train()
        print(f"  [val ep{ep:03d}] ratio={ratio:.4f}"
              + ("  <= 低于下限 0.615，已开始记忆" if ratio < 0.615 else ""), flush=True)
        return ratio

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for step in range(args.steps_per_epoch):
            k = int(rng.integers(0, len(xs)))
            x, y, vmask = xs[k], ys[k], vs[k]
            sl = rand_patch(x, args.patch, rng)
            xa = torch.as_tensor(x[sl], device=device)[None, None].float()
            ya = torch.as_tensor(y[sl], device=device)[None, None].float()
            ma = torch.as_tensor(vmask[sl], device=device)[None, None].float()
            h = model(xa)
            pred = xa + h if args.residual else h
            loss = masked_mse(pred, ya, ma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            if step % 20 == 0:
                print(f"epoch={ep + 1:03d} step={step:03d} loss={float(loss):.6f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        rec = {"epoch": ep + 1, "loss": tot / args.steps_per_epoch,
               "identity_ref": ident, "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        print(f"epoch={ep + 1:03d} mean_loss={rec['loss']:.6f} "
              f"(恒等基准 {ident:.6f}, 比值 {rec['loss'] / max(ident, 1e-9):.2f}) "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val is not None and ((ep + 1) % args.val_every == 0 or ep + 1 == args.epochs):
            rr = validate(ep + 1)
            rec["val_ratio"] = rr
            if rr is not None and rr < best["ratio"]:
                best = {"ratio": rr, "epoch": ep + 1}
                torch.save({"model": model.state_dict(), "epoch": ep + 1,
                            "args": dict(cargs, val_ratio=rr)}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "args": cargs},
                   out_dir / "latest.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    print(f"done  best val ratio={best['ratio']:.4f} @ep{best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
