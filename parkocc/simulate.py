"""Synthetic lot-occupancy generator (the 'Dataset A' idea from the report, scaled for a prototype).

A clean latent occupancy ratio is built from a bimodal daily demand curve, a weekday/weekend factor and a
slow drift. A camera-like noisy observation is derived from it with four noise channels. Labels come from the
CLEAN ratio only; models only ever see the noisy observation.
"""
import numpy as np
from . import WINDOW

STEPS_PER_DAY = 96                      # 15-minute resolution
COND_SIGMA = {0: 0.03, 1: 0.05, 2: 0.09}   # jitter per filming condition: clear, overcast, rain
P_SPIKE, P_CORRUPT = 0.02, 0.01         # occlusion/glare spikes, unusable frames
CAP_MIN, CAP_MAX = 40, 400


def ratio_to_class(r):
    """Five ordinal classes at fixed 20% boundaries."""
    return np.clip((np.asarray(r) / 0.2).astype(int), 0, 4)


def time_features(tod_hours, dow):
    """sin/cos time of day, sin/cos day of week, weekend flag -> (..., 5)."""
    tod_hours = np.asarray(tod_hours, dtype=np.float32); dow = np.asarray(dow, dtype=np.float32)
    a = 2 * np.pi * tod_hours / 24.0; b = 2 * np.pi * dow / 7.0
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b), (dow >= 5).astype(np.float32)], axis=-1)


def _demand(tod):
    morning = np.exp(-0.5 * ((tod - 9.0) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((tod - 17.5) / 2.5) ** 2)
    return 0.16 + 0.60 * morning + 0.55 * evening


def make_lot(rng, days=70):
    """Return per-step arrays for one lot."""
    n = days * STEPS_PER_DAY
    tod = np.tile(np.arange(STEPS_PER_DAY) * 24.0 / STEPS_PER_DAY, days)
    day_idx = np.repeat(np.arange(days), STEPS_PER_DAY)
    dow = day_idx % 7
    cap = rng.integers(CAP_MIN, CAP_MAX + 1)
    amp = rng.uniform(0.75, 1.35); bias = rng.normal(0, 0.03)
    wk = np.where(dow >= 5, rng.uniform(0.5, 0.9), 1.0)
    drift_d = np.clip(np.cumsum(rng.normal(0, 0.035, days)), -0.12, 0.12)
    cond_day = rng.choice(3, size=days, p=[0.6, 0.25, 0.15])
    ar = np.zeros(n); e = rng.normal(0, 0.015, n)
    for i in range(1, n): ar[i] = 0.9 * ar[i - 1] + e[i]
    r = np.clip(_demand(tod) * amp * wk + drift_d[day_idx] + bias + ar, 0, 1)
    cond = cond_day[day_idx]
    # --- noisy observation ---
    x = r + rng.normal(0, 1, n) * np.vectorize(COND_SIGMA.get)(cond)
    spike = rng.random(n) < P_SPIKE
    x[spike] += rng.choice([-1, 1], spike.sum()) * rng.uniform(0.2, 0.4, spike.sum())
    corrupt = rng.random(n) < P_CORRUPT
    x[corrupt] = np.nan
    idx = np.arange(n); ok = ~np.isnan(x)
    x = np.interp(idx, idx[ok], x[ok])            # interpolate unusable frames; flag stays 1
    x = np.clip(x, 0, 1)
    feats = np.concatenate([x[:, None], time_features(tod, dow),
                            np.full((n, 1), (cap - CAP_MIN) / (CAP_MAX - CAP_MIN)),
                            (cond / 2.0)[:, None], corrupt[:, None].astype(float)], axis=1).astype(np.float32)
    return feats, ratio_to_class(r)


def windows(feats, labels, stride=1):
    """Sliding windows (N, WINDOW, 9); label = class at the window's last step."""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(feats, (WINDOW, feats.shape[1]))[:, 0][::stride]
    y = labels[WINDOW - 1:][::stride]
    return np.ascontiguousarray(w), y[: len(w)]


def build_dataset(n_lots=40, days=70, seed=42, stride=2, split=(28, 6, 6)):
    """Lot-disjoint train/val/test split (no test lot is ever seen in training)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    bounds = np.cumsum(split)
    for lot in range(n_lots):
        f, y = make_lot(rng, days)
        name = "train" if lot < bounds[0] else "val" if lot < bounds[1] else "test"
        parts[name].append(windows(f, y, stride))
    out = {}
    for k, v in parts.items():
        out[k] = (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]))
    return out
