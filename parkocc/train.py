"""Train everything and write ./artifacts:  python -m parkocc.train   (about 15 minutes on one CPU core)

1. frame feature extractor  — MLP (fast) + CNN (numpy, slower)
2. baseline / RNN / LSTM / GRU  on the synthetic noisy dataset (lot-disjoint 28/6/6 split)
3. K-means+CNN classifier
4. GAN (conditional, 30 epochs) + GANClassifier head fine-tuning
5. evaluation: held-out test lots, plus end-to-end on freshly rendered clips
"""
import os, json, time, sys
import numpy as np
from . import CLASSES, WINDOW, simulate, render, frames, nn
from . import cnn as cnn_mod
from . import gan as gan_mod

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


def train_cnn_extractor(n_train=400, n_test=100, seed=1):
    """Train the numpy CNN extractor on rendered lot images."""
    rng = np.random.default_rng(seed)
    images, y = [], []
    t0 = time.time()
    log("  rendering images for CNN extractor...")
    for i in range(n_train + n_test):
        k = int(rng.integers(0, render.N_SLOTS + 1))
        cond = int(rng.choice(3, p=[.55, .25, .2]))
        m = np.zeros(render.N_SLOTS, bool); m[rng.choice(render.N_SLOTS, k, replace=False)] = True
        img = render.recompress(render.render_lot(m, rng, cond, True), rng)
        images.append(img); y.append(k / render.N_SLOTS)
        if i % 100 == 0: log(f"  CNN extractor rendering {i}/{n_train + n_test}")
    y = np.array(y, np.float32)
    cnn_ext = cnn_mod.CNNExtractor(rng=np.random.default_rng(seed))
    cnn_mod.fit_cnn_extractor(cnn_ext, images[:n_train], y[:n_train],
                               epochs=5, batch=16, lr=5e-4, log=log)
    # evaluate
    preds = np.array([cnn_ext.ratio(images[i]) for i in range(n_train, n_train + n_test)])
    err = np.abs(preds - y[n_train:])
    np.savez(os.path.join(ART, "cnn_extractor.npz"), **cnn_ext.state())
    info = {"mae_ratio": float(err.mean()), "mae_slots": float(err.mean() * render.N_SLOTS),
            "train_images": n_train, "test_images": n_test,
            "seconds": round(time.time() - t0, 1)}
    log("  CNN extractor", info)
    return cnn_ext, info


def window_from_ratios(ratios, cond, hour, dow, cap_norm=0.5, flags=None):
    """(6,9) model input from six per-frame ratios + context. Frames are treated as consecutive 15-minute readings."""
    T = len(ratios); tod = (hour + np.arange(T) * 0.25) % 24
    f = np.zeros((T, 9), np.float32); f[:, 0] = ratios
    f[:, 1:6] = simulate.time_features(tod, np.full(T, dow)); f[:, 6] = cap_norm; f[:, 7] = cond / 2.0
    f[:, 8] = 0 if flags is None else np.asarray(flags, np.float32); return f


def main():
    os.makedirs(ART, exist_ok=True); t_all = time.time()
    retrain_all = ("--retrain-all" in sys.argv) or ("--force" in sys.argv)

    # 1/5 MLP frame feature extractor
    ext_path = os.path.join(ART, "extractor.npz")
    met_path = os.path.join(ART, "metrics.json")
    old_met = json.load(open(met_path)) if os.path.exists(met_path) else {}
    if os.path.exists(ext_path) and not retrain_all:
        log("1/5 MLP frame feature extractor [found existing artifacts/extractor.npz]")
        ext = frames.FrameExtractor.load(ext_path)
        ext_info = old_met.get("extractor", {"train_images": 4200, "test_images": 600, "seconds": 0})
    else:
        log("1/5 MLP frame feature extractor"); ext, ext_info = train_extractor()

    # 2/5 CNN frame feature extractor (numpy)
    cnn_ext_path = os.path.join(ART, "cnn_extractor.npz")
    if os.path.exists(cnn_ext_path) and not retrain_all:
        log("2/5 CNN frame feature extractor [found existing artifacts/cnn_extractor.npz]")
        cnn_ext = cnn_mod.CNNExtractor.from_state(np.load(cnn_ext_path))
        cnn_ext_info = old_met.get("cnn_extractor", {})
    else:
        log("2/5 CNN frame feature extractor (numpy)"); cnn_ext, cnn_ext_info = train_cnn_extractor()

    log("3/5 synthetic dataset + RNN/LSTM/GRU/baseline models")
    data = simulate.build_dataset(stride=2, seed=42)
    for k, (X, y) in data.items(): log(f"  {k}: {X.shape}, class share {np.round(np.bincount(y, minlength=5) / len(y), 3)}")
    freq = np.bincount(data["train"][1], minlength=5) / len(data["train"][1])
    cw = (1 / np.sqrt(freq)); cw = (cw / cw.mean()).astype(np.float32)
    models, report = {}, {}
    for kind in ["baseline", "rnn", "lstm", "gru"]:
        kp = os.path.join(ART, f"{kind}.npz")
        if os.path.exists(kp) and not retrain_all and kind in old_met.get("test_lots", {}):
            log(f"  loading existing {kind} from {kp}")
            m = nn.Classifier.from_state(np.load(kp))
            models[kind] = m
            report[kind] = old_met["test_lots"][kind]
        else:
            log(f"  training {kind}"); t0 = time.time()
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

    log("4/5 K-means+CNN classifier")
    km_model = cnn_mod.KmeansCNNClassifier(K=10, seed=42)
    t0 = time.time()
    km_info = cnn_mod.fit_kmeans_cnn(km_model, data["train"], data["val"],
                                      class_weights=cw, log=log)
    Xt, yt = data["test"]
    km_pred = km_model.predict_proba(Xt).argmax(1)
    km_res = nn.metrics(yt, km_pred)
    one = Xt[:1]; t1 = time.time()
    for _ in range(300): km_model.predict_proba(one)
    km_res.update({"params": km_model.n_params(), "train_seconds": round(time.time() - t0, 1),
                   "epochs": km_info["epochs"], "best_val_acc": km_info["best_val_acc"],
                   "ms_per_window": round((time.time() - t1) / 300 * 1000, 3), "hidden": 32})
    models["kmeans_cnn"] = km_model; report["kmeans_cnn"] = km_res
    np.savez(os.path.join(ART, "kmeans_cnn.npz"), **km_model.state())
    log(f"  TEST kmeans_cnn: acc {km_res['accuracy']:.4f} macroF1 {km_res['macro_f1']:.4f}")

    log("5/5 GAN - loading trained discriminator, fitting new 2-layer head")
    Xtr, ytr = data["train"]
    t0 = time.time()
    gen_path = os.path.join(ART, "gan_generator.npz")
    gan_path = os.path.join(ART, "gan.npz")
    if os.path.exists(gen_path) and not retrain_all:
        log("  GAN already trained — reusing discriminator, fitting new 2-layer head")
        zgen = np.load(gen_path, allow_pickle=False)
        xmu = np.asarray(zgen["xmu"], np.float32)
        xsd = np.asarray(zgen["xsd"], np.float32)
        D = gan_mod.Discriminator(in_dim=9, n_classes=5, seed=1)
        if os.path.exists(gan_path):
            D.load_state(np.load(gan_path, allow_pickle=False))
    else:
        log("  training GAN from scratch (30 epochs)")
        G, D, xmu, xsd, gan_hist = gan_mod.train_gan(Xtr, ytr, epochs=30, log=log)
        gan_state = G.state(); gan_state.update({"xmu": xmu, "xsd": xsd})
        np.savez(os.path.join(ART, "gan_generator.npz"), **gan_state)
    # build new GANClassifier with two-layer head and fine-tune
    gan_cls = gan_mod.GANClassifier(D, xmu, xsd, seed=42)
    gan_head_info = gan_mod.fit_gan_head(gan_cls, data["train"], data["val"],
                                          class_weights=cw, log=log)
    Xt, yt = data["test"]
    gan_pred = gan_cls.predict_proba(Xt).argmax(1)
    gan_res = nn.metrics(yt, gan_pred)
    one = Xt[:1]; t1 = time.time()
    for _ in range(300): gan_cls.predict_proba(one)
    gan_res.update({"params": gan_cls.n_params(), "train_seconds": round(time.time() - t0, 1),
                    "epochs": gan_head_info["epochs"], "best_val_acc": gan_head_info["best_val_acc"],
                    "ms_per_window": round((time.time() - t1) / 300 * 1000, 3), "hidden": 128})
    models["gan"] = gan_cls; report["gan"] = gan_res
    np.savez(os.path.join(ART, "gan.npz"), **gan_cls.state())
    log(f"  TEST gan: acc {gan_res['accuracy']:.4f} macroF1 {gan_res['macro_f1']:.4f}")

    log("end-to-end check on freshly rendered clips")
    ALL_MODELS = {**models}
    rng = np.random.default_rng(777); n_clips = 240; true, wins = [], []
    for i in range(n_clips):
        r = float(rng.uniform(0, 1)); cond = int(rng.choice(3, p=[.55, .25, .2]))
        fr, tr = render.render_clip(r, rng, cond, True)
        fr = [render.recompress(f, rng) for f in fr]; rat = np.array([ext.ratio(f) for f in fr]); flg = [frames.unusable(f) for f in fr]
        wins.append(window_from_ratios(rat, cond, float(rng.uniform(0, 24)), int(rng.integers(7)), flags=flg)); true.append(int(simulate.ratio_to_class(tr[-1])))
    W = np.array(wins); true = np.array(true); e2e = {}
    for kind, m in ALL_MODELS.items():
        e2e[kind] = nn.metrics(true, m.predict_proba(W).argmax(1))
        e2e[kind] = {k: e2e[kind][k] for k in ("accuracy", "macro_f1", "full_recall", "empty_full_confusions", "confusion")}
        log(f"  E2E {kind}: acc {e2e[kind]['accuracy']:.3f} macroF1 {e2e[kind]['macro_f1']:.3f}")

    out = {"classes": CLASSES, "window": WINDOW,
           "extractor": ext_info, "cnn_extractor": cnn_ext_info,
           "test_lots": report, "end_to_end": {"n_clips": n_clips, **e2e},
           "dataset": {k: {"windows": int(len(v[0])), "share": np.round(np.bincount(v[1], minlength=5) / len(v[1]), 4).tolist()} for k, v in data.items()},
           "class_weights": cw.round(3).tolist(), "seed": 42, "total_seconds": round(time.time() - t_all, 1)}
    json.dump(out, open(os.path.join(ART, "metrics.json"), "w"), indent=1); log("done in", out["total_seconds"], "s")


if __name__ == "__main__":
    main()
