"""Train everything and write ./artifacts:  python -m parkocc.train   (about 10 minutes on one CPU core)

1. frame feature extractor  (rendered lots -> occupancy ratio)
2. baseline / RNN / LSTM / GRU on the synthetic noisy dataset (lot-disjoint 28/6/6 split)
3. evaluation: held-out test lots, plus end-to-end on freshly rendered clips (frames -> extractor -> models)
"""
import os, json, time, sys
import numpy as np
from . import CLASSES, WINDOW, simulate, render, frames, nn

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
HIDDEN = {"rnn": 32, "lstm": 64, "gru": 64}


def log(*a): print(*a, flush=True)


def train_extractor(n_train=4200, n_test=600, seed=0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed); X, y, c = [], [], []
    t0 = time.time()
    for i in range(n_train + n_test):
        k = int(rng.integers(0, render.N_SLOTS + 1)); cond = int(rng.choice(3, p=[.55, .25, .2]))
        m = np.zeros(render.N_SLOTS, bool); m[rng.choice(render.N_SLOTS, k, replace=False)] = True
        X.append(frames.frame_stats(render.recompress(render.render_lot(m, rng, cond, True), rng))); y.append(k / render.N_SLOTS); c.append(cond)
        if i % 500 == 0: log(f"  rendering {i}/{n_train + n_test}")
    X, y, c = np.array(X), np.array(y), np.array(c)
    sc = StandardScaler().fit(X[:n_train])
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=seed)
    mlp.fit(sc.transform(X[:n_train]), y[:n_train])
    pred = np.clip(mlp.predict(sc.transform(X[n_train:])), 0, 1); err = np.abs(pred - y[n_train:])
    params = {"mu": sc.mean_.astype(np.float32), "sd": sc.scale_.astype(np.float32)}
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)): params[f"W{i}"] = W.astype(np.float32); params[f"b{i}"] = b.astype(np.float32)
    np.savez(os.path.join(ART, "extractor.npz"), **params)
    info = {"mae_ratio": float(err.mean()), "mae_slots": float(err.mean() * render.N_SLOTS),
            "mae_by_condition": {n: float(err[c[n_train:] == k].mean()) for k, n in enumerate(["clear", "overcast", "rain"])},
            "train_images": n_train, "test_images": n_test, "seconds": round(time.time() - t0, 1)}
    log("  extractor", info); return frames.FrameExtractor(params), info


def window_from_ratios(ratios, cond, hour, dow, cap_norm=0.5, flags=None):
    """(6,9) model input from six per-frame ratios + context. Frames are treated as consecutive 15-min readings."""
    T = len(ratios); tod = (hour + np.arange(T) * 0.25) % 24
    f = np.zeros((T, 9), np.float32); f[:, 0] = ratios
    f[:, 1:6] = simulate.time_features(tod, np.full(T, dow)); f[:, 6] = cap_norm; f[:, 7] = cond / 2.0
    f[:, 8] = 0 if flags is None else np.asarray(flags, np.float32); return f


def main():
    os.makedirs(ART, exist_ok=True); t_all = time.time()
    log("1/3 frame feature extractor"); ext, ext_info = train_extractor()
    log("2/3 synthetic dataset + models"); data = simulate.build_dataset(stride=2, seed=42)
    for k, (X, y) in data.items(): log(f"  {k}: {X.shape}, class share {np.round(np.bincount(y, minlength=5) / len(y), 3)}")
    freq = np.bincount(data["train"][1], minlength=5) / len(data["train"][1])
    cw = (1 / np.sqrt(freq)); cw = (cw / cw.mean()).astype(np.float32)
    models, report = {}, {}
    for kind in ["baseline", "rnn", "lstm", "gru"]:
        log(f"training {kind}"); t0 = time.time()
        m = nn.Classifier(kind, H=HIDDEN.get(kind, 64), seed=42)
        info = nn.fit(m, data["train"], data["val"], class_weights=cw, log=log)
        Xt, yt = data["test"]; pred = m.predict_proba(Xt).argmax(1); res = nn.metrics(yt, pred)
        one = Xt[:1]; t1 = time.time()
        for _ in range(300): m.predict_proba(one)
        res.update({"params": m.n_params(), "train_seconds": round(time.time() - t0, 1), "epochs": info["epochs"],
                    "best_val_acc": info["best_val_acc"], "ms_per_window": round((time.time() - t1) / 300 * 1000, 3),
                    "hidden": HIDDEN.get(kind)})
        models[kind] = m; report[kind] = res
        np.savez(os.path.join(ART, f"{kind}.npz"), **m.state())
        log(f"  TEST {kind}: acc {res['accuracy']:.4f} macroF1 {res['macro_f1']:.4f} Full-recall {res['full_recall']:.3f} params {res['params']}")
    log("3/3 end-to-end check on freshly rendered clips")
    rng = np.random.default_rng(777); n_clips = 240; true, wins = [], []
    for i in range(n_clips):
        r = float(rng.uniform(0, 1)); cond = int(rng.choice(3, p=[.55, .25, .2]))
        fr, tr = render.render_clip(r, rng, cond, True)
        fr = [render.recompress(f, rng) for f in fr]; rat = np.array([ext.ratio(f) for f in fr]); flg = [frames.unusable(f) for f in fr]
        wins.append(window_from_ratios(rat, cond, float(rng.uniform(0, 24)), int(rng.integers(7)), flags=flg)); true.append(int(simulate.ratio_to_class(tr[-1])))
    W = np.array(wins); true = np.array(true); e2e = {}
    for kind, m in models.items():
        e2e[kind] = nn.metrics(true, m.predict_proba(W).argmax(1))
        e2e[kind] = {k: e2e[kind][k] for k in ("accuracy", "macro_f1", "full_recall", "empty_full_confusions", "confusion")}
        log(f"  E2E {kind}: acc {e2e[kind]['accuracy']:.3f} macroF1 {e2e[kind]['macro_f1']:.3f}")
    out = {"classes": CLASSES, "window": WINDOW, "extractor": ext_info, "test_lots": report, "end_to_end": {"n_clips": n_clips, **e2e},
           "dataset": {k: {"windows": int(len(v[0])), "share": np.round(np.bincount(v[1], minlength=5) / len(v[1]), 4).tolist()} for k, v in data.items()},
           "class_weights": cw.round(3).tolist(), "seed": 42, "total_seconds": round(time.time() - t_all, 1)}
    json.dump(out, open(os.path.join(ART, "metrics.json"), "w"), indent=1); log("done in", out["total_seconds"], "s")


if __name__ == "__main__":
    main()
