"""Global sequence model arm: one LSTM trained across the whole transformer fleet.

Sample = (transformer j, day d). Encoder reads the 168 scale-free hourly loads
through hour 23 of day d-1. Decoder inputs are the calendar of day d, the day-ahead
temperature path, static transformer attributes, and optionally a cluster embedding.
Output is the 24-hour scale-free profile of day d, so the model is a direct
multi-horizon forecaster with no recursion.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HIST = 168


def _samples(panel, rows, scale, min_row=HIST + 24):
    """Day indices usable as forecast targets within `rows`."""
    day_start = np.array(sorted({r - r % 24 for r in rows if r % 24 == 0 or True}))
    day_start = np.unique(rows - (rows % 24))
    return day_start[(day_start >= min_row) & (day_start + 24 <= panel.T)]


def build_tensors(panel, scale, days, cols, temp_noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    L = (panel.loads / scale[None, :]).astype(np.float32)
    tp = (panel.temp + rng.normal(0, temp_noise, panel.temp.shape)).astype(np.float32)
    cal = panel.cal
    st = panel.meta
    stat = np.stack([np.log(st.kva.to_numpy(np.float32)), st.n_cust.to_numpy(np.float32) / 10.0,
                     st.res_share.to_numpy(np.float32),
                     st.pv_kw.to_numpy(np.float32) / np.maximum(st.n_cust.to_numpy(np.float32), 1) / 5.0,
                     np.log(np.maximum(scale, 1e-3))], 1).astype(np.float32)
    H, Xe, Y, RR, CC = [], [], [], [], []
    for d in days:
        hist = L[d - HIST:d]                       # (168, N)
        fut = L[d:d + 24]                          # (24, N)
        ex = np.concatenate([
            tp[d:d + 24].T,                        # (N,24) temperature path
            np.repeat(cal["dow"][d:d + 1], len(cols))[:, None] / 6.0,
            np.repeat(cal["is_weekend"][d:d + 1], len(cols))[:, None],
            np.repeat(cal["holiday"][d:d + 1], len(cols))[:, None],
            np.repeat(cal["doy_sin"][d:d + 1], len(cols))[:, None],
            np.repeat(cal["doy_cos"][d:d + 1], len(cols))[:, None],
            stat[cols]], axis=1)
        ok = np.isfinite(hist[:, cols]).all(0) & np.isfinite(fut[:, cols]).all(0) & np.isfinite(ex).all(1)
        if not ok.any():
            continue
        sel = np.asarray(cols)[ok]
        H.append(hist[:, sel].T[:, :, None])
        Xe.append(ex[ok])
        Y.append(fut[:, sel].T)
        RR.append(np.repeat(d, len(sel)))
        CC.append(sel)
    return (np.concatenate(H).astype(np.float32), np.concatenate(Xe).astype(np.float32),
            np.concatenate(Y).astype(np.float32), np.concatenate(RR), np.concatenate(CC))


def lstm_global_predict(panel, scale, tr_rows, va_rows, te_rows, cols=None, target_cols=None,
                        hidden=48, epochs=12, batch=256, lr=1.5e-3, max_train=80000,
                        groups=None, n_group=0, seed=0, verbose=True):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    cols = np.arange(panel.N) if cols is None else np.asarray(cols)
    tgt = cols if target_cols is None else np.asarray(target_cols)

    tr_d, va_d, te_d = (_samples(panel, r, scale) for r in (tr_rows, va_rows, te_rows))
    Htr, Etr, Ytr, _, Ctr = build_tensors(panel, scale, tr_d, cols, seed=seed)
    Hva, Eva, Yva, _, Cva = build_tensors(panel, scale, va_d, cols, seed=seed + 1)
    Hte, Ete, Yte, Rte, Cte = build_tensors(panel, scale, te_d, tgt, seed=seed + 2)
    if len(Ytr) > max_train:
        k = np.random.default_rng(seed).choice(len(Ytr), max_train, replace=False)
        Htr, Etr, Ytr, Ctr = Htr[k], Etr[k], Ytr[k], Ctr[k]
    if verbose:
        print(f"  lstm: {len(Ytr)} train / {len(Yva)} val / {len(Yte)} test samples", flush=True)

    use_g = groups is not None and n_group > 0

    class Net(nn.Module):
        def __init__(self, n_ex):
            super().__init__()
            self.enc = nn.LSTM(1, hidden, batch_first=True)
            self.emb = nn.Embedding(max(n_group, 1), 8) if use_g else None
            d = hidden + n_ex + (8 if use_g else 0)
            self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 24))

        def forward(self, h, e, g=None):
            _, (hn, _) = self.enc(h)
            z = [hn[-1], e]
            if use_g:
                z.append(self.emb(g))
            return self.head(torch.cat(z, 1))

    net = Net(Etr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss(beta=0.15)

    def tens(*a):
        return [torch.from_numpy(x) for x in a]

    Ht, Et, Yt = tens(Htr, Etr, Ytr)
    Gt = torch.from_numpy(groups[Ctr].astype(np.int64)) if use_g else None
    Hv, Ev, Yv = tens(Hva, Eva, Yva)
    Gv = torch.from_numpy(groups[Cva].astype(np.int64)) if use_g else None
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            out = net(Ht[b], Et[b], Gt[b] if use_g else None)
            loss = lossf(out, Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Hv, Ev, Gv if use_g else None), Yv))
        if verbose:
            print(f"   epoch {ep+1}: train {tot/len(Yt):.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-5:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    Hs, Es = tens(Hte, Ete)
    Gs = torch.from_numpy(groups[Cte].astype(np.int64)) if use_g else None
    with torch.no_grad():
        P = np.concatenate([net(Hs[i:i + 1024], Es[i:i + 1024],
                                Gs[i:i + 1024] if use_g else None).numpy()
                            for i in range(0, len(Hte), 1024)])
    rows = (Rte[:, None] + np.arange(24)[None, :]).ravel()
    cols_out = np.repeat(Cte, 24)
    return P.ravel(), Yte.ravel(), rows, cols_out
