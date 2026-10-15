"""Pure-numpy GAN for parking-lot time-series augmentation.

Architecture
------------
Generator   : noise(z_dim) -> Dense(64) LReLU -> Dense(128) LReLU -> Dense(9) tanh
              Conditioned on class label (one-hot concatenated to z)
Discriminator: x(9)+label(5) -> Dense(128) LReLU -> Dense(64) LReLU -> Dense(1) sigmoid

Training    : standard minimax GAN; trained on the last-frame feature vector (9-d)
              extracted from the RNN/LSTM/GRU training set.

GANClassifier (selectable model)
---------------------------------
After GAN training, the Discriminator's internal representation (penultimate layer, 64-d)
is used as features fed into a Dense(5) softmax head — "GAN as feature extractor."
The head is fine-tuned on the real labelled data while the discriminator backbone is frozen.
"""
import numpy as np, time

lrelu = lambda x, a=0.2: np.where(x > 0, x, a * x)
dlrelu = lambda x, a=0.2: np.where(x > 0, 1.0, a)
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _glorot(rng, i, o):
    lim = np.sqrt(6.0 / (i + o))
    return rng.uniform(-lim, lim, (i, o)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────

class Generator:
    """Conditional generator: (z, label_onehot) -> fake sample (9-d)."""

    def __init__(self, z_dim=16, n_classes=5, out_dim=9, seed=0):
        rng = np.random.default_rng(seed)
        self.z_dim, self.n_classes, self.out_dim = z_dim, n_classes, out_dim
        inp = z_dim + n_classes
        self.p = [
            {"W": _glorot(rng, inp, 64),  "b": np.zeros(64,  np.float32)},
            {"W": _glorot(rng, 64,  128), "b": np.zeros(128, np.float32)},
            {"W": _glorot(rng, 128, out_dim), "b": np.zeros(out_dim, np.float32)},
        ]
        self._cache = []

    def forward(self, z, label_oh):
        inp = np.concatenate([z, label_oh], axis=1)
        self._cache = []
        a = inp
        for i, p in enumerate(self.p):
            pre = a @ p["W"] + p["b"]
            self._cache.append((a, pre))
            a = lrelu(pre) if i < len(self.p) - 1 else np.tanh(pre)
        return a   # (B, 9)

    def backward(self, dout):
        d = dout
        grads = []
        for i in range(len(self.p) - 1, -1, -1):
            a, pre = self._cache[i]
            if i < len(self.p) - 1:
                d = d * dlrelu(pre)
            else:
                d = d * (1 - np.tanh(pre) ** 2)
            grads.insert(0, {"W": a.T @ d, "b": d.sum(0)})
            d = d @ self.p[i]["W"].T
        return grads

    def state(self):
        d = {}
        for i, p in enumerate(self.p):
            for k, v in p.items():
                d[f"g{i}.{k}"] = v
        return d

    def load_state(self, z):
        for i, p in enumerate(self.p):
            for k in p:
                key = f"g{i}.{k}"
                if key in z:
                    self.p[i][k] = np.asarray(z[key], np.float32)


# ─────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────

class Discriminator:
    """Conditional discriminator: (x, label_oh) -> probability real."""

    def __init__(self, in_dim=9, n_classes=5, seed=1):
        rng = np.random.default_rng(seed)
        self.in_dim, self.n_classes = in_dim, n_classes
        inp = in_dim + n_classes
        self.p = [
            {"W": _glorot(rng, inp, 128), "b": np.zeros(128, np.float32)},
            {"W": _glorot(rng, 128, 64),  "b": np.zeros(64,  np.float32)},
            {"W": _glorot(rng, 64,  1),   "b": np.zeros(1,   np.float32)},
        ]
        self._cache = []

    def forward(self, x, label_oh):
        inp = np.concatenate([x, label_oh], axis=1)
        self._cache = []
        a = inp
        for i, p in enumerate(self.p):
            pre = a @ p["W"] + p["b"]
            self._cache.append((a, pre))
            if i < len(self.p) - 1:
                a = lrelu(pre)
            else:
                a = sigmoid(pre)
        self._out = a
        return a   # (B,1)

    def penultimate(self, x, label_oh):
        """Return the 64-d features just before the final layer (for GANClassifier)."""
        inp = np.concatenate([x, label_oh], axis=1)
        a = inp
        for i, p in enumerate(self.p):
            pre = a @ p["W"] + p["b"]
            if i < len(self.p) - 1:
                a = lrelu(pre)
            else:
                return a   # 64-d pre-sigmoid features

    def backward(self, dloss_dout):
        d = dloss_dout * self._out * (1 - self._out)   # sigmoid backward
        grads = []
        for i in range(len(self.p) - 1, -1, -1):
            a, pre = self._cache[i]
            if i < len(self.p) - 1:
                d = d * dlrelu(pre)
            grads.insert(0, {"W": a.T @ d, "b": d.sum(0)})
            d = d @ self.p[i]["W"].T
        return grads, d[:, :self.in_dim]   # grad w.r.t. original x

    def state(self):
        d = {}
        for i, p in enumerate(self.p):
            for k, v in p.items():
                d[f"d{i}.{k}"] = v
        return d

    def load_state(self, z):
        for i, p in enumerate(self.p):
            for k in p:
                key = f"d{i}.{k}"
                if key in z:
                    self.p[i][k] = np.asarray(z[key], np.float32)


# ─────────────────────────────────────────────────────────────
# Adam (generic, works on list of param dicts)
# ─────────────────────────────────────────────────────────────

class _Adam:
    def __init__(self, param_list, lr=2e-4, b1=0.5, b2=0.999, clip=5.0):
        self.pl = param_list
        self.lr, self.b1, self.b2, self.clip, self.t = lr, b1, b2, clip, 0
        self.s = [{k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}
                  for p in param_list]

    def step(self, grads):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for gl in grads for g in gl.values()) + 1e-12)
        sc = min(1.0, self.clip / norm)
        for p, gl, st in zip(self.pl, grads, self.s):
            for k in p:
                g = gl[k] * sc
                m, v = st[k]
                m[:] = self.b1 * m + (1 - self.b1) * g
                v[:] = self.b2 * v + (1 - self.b2) * g * g
                mh = m / (1 - self.b1 ** self.t)
                vh = v / (1 - self.b2 ** self.t)
                p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ─────────────────────────────────────────────────────────────
# GAN training loop
# ─────────────────────────────────────────────────────────────

def train_gan(X_real, y_real, z_dim=16, n_classes=5,
              epochs=30, batch=128, lr=2e-4, seed=42, log=print):
    """Train conditional GAN on last-frame feature vectors.
    X_real : (N, T, 9) windows  (only last frame used)
    y_real : (N,) integer labels 0-4
    Returns (generator, discriminator, history)
    """
    rng = np.random.default_rng(seed)
    x = X_real[:, -1, :].astype(np.float32)   # (N, 9)
    # normalise to ~[-1,1]
    xmu = x.mean(0); xsd = x.std(0) + 1e-6
    xn = (x - xmu) / xsd

    G = Generator(z_dim=z_dim, n_classes=n_classes, out_dim=x.shape[1], seed=seed)
    D = Discriminator(in_dim=x.shape[1], n_classes=n_classes, seed=seed + 1)
    opt_G = _Adam(G.p, lr=lr)
    opt_D = _Adam(D.p, lr=lr)

    REAL = np.ones((batch, 1), np.float32) * 0.9   # label smoothing
    FAKE = np.zeros((batch, 1), np.float32) + 0.1

    hist = []
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.permutation(len(xn))
        d_tot = g_tot = 0.0
        n_steps = 0
        for s in range(0, len(idx) - batch, batch):
            bi = idx[s:s + batch]
            xb = xn[bi]; yb = y_real[bi]
            loh = np.eye(n_classes, dtype=np.float32)[yb]

            # ── train D ────────────────────────────────────
            z = rng.standard_normal((batch, z_dim)).astype(np.float32)
            fake = G.forward(z, loh)
            p_real = D.forward(xb, loh)
            p_fake = D.forward(fake, loh)

            # BCE loss: -[real*log(p_real) + fake*log(1-p_fake)]
            d_loss_r = -np.log(p_real + 1e-8).mean()
            d_loss_f = -np.log(1 - p_fake + 1e-8).mean()
            d_loss = d_loss_r + d_loss_f

            # D backward (real)
            D.forward(xb, loh)
            dr_real = -1.0 / (p_real + 1e-8) / batch
            gD_real, _ = D.backward(dr_real)

            # D backward (fake)
            D.forward(fake, loh)
            dr_fake = 1.0 / (1 - p_fake + 1e-8) / batch
            gD_fake, _ = D.backward(dr_fake)

            # combine D grads
            gD = [{k: gD_real[i][k] + gD_fake[i][k] for k in D.p[i]}
                  for i in range(len(D.p))]
            opt_D.step(gD)

            # ── train G ────────────────────────────────────
            z2 = rng.standard_normal((batch, z_dim)).astype(np.float32)
            fake2 = G.forward(z2, loh)
            p_fake2 = D.forward(fake2, loh)

            g_loss = -np.log(p_fake2 + 1e-8).mean()
            # G backward through D
            dr_g = -1.0 / (p_fake2 + 1e-8) / batch
            _, dx_fake = D.backward(dr_g)
            # dx_fake is grad w.r.t. fake (= G output), shape (B, 9)
            dG = G.backward(dx_fake[:, :x.shape[1]])   # strip label part
            opt_G.step(dG)

            d_tot += float(d_loss); g_tot += float(g_loss); n_steps += 1

        hist.append({"ep": ep + 1, "D_loss": d_tot / n_steps, "G_loss": g_tot / n_steps})
        log(f"  [GAN] epoch {ep+1:2d}  D={d_tot/n_steps:.4f}  G={g_tot/n_steps:.4f}  ({time.time()-t0:.1f}s)")

    return G, D, xmu, xsd, hist


def generate_samples(G, xmu, xsd, n_per_class=200, z_dim=16, n_classes=5, seed=99):
    """Generate augmented samples for each class. Returns (X_aug, y_aug) as windows."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(n_classes):
        z = rng.standard_normal((n_per_class, z_dim)).astype(np.float32)
        loh = np.tile(np.eye(n_classes, dtype=np.float32)[c], (n_per_class, 1))
        fake = G.forward(z, loh)
        real = fake * xsd + xmu            # un-normalise
        real = np.clip(real, 0, 1)
        # repeat single frame as a T=6 window (same as image baseline)
        win = np.repeat(real[:, None, :], 6, axis=1).astype(np.float32)
        Xs.append(win); ys.append(np.full(n_per_class, c, dtype=np.int64))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


# ─────────────────────────────────────────────────────────────
# GAN Classifier  (selectable model, uses D's hidden repr.)
# ─────────────────────────────────────────────────────────────

class GANClassifier:
    """Uses the Discriminator's penultimate (64-d) layer as a fixed feature extractor.

    v2 improvements:
    - Features are averaged over ALL T frames (not just the last one).
    - Two-layer trainable head: 64 -> 128 ReLU -> n_classes.

    Input: window (B,T,9)
    """
    kind = "gan"

    def __init__(self, discriminator, xmu, xsd, n_classes=5, seed=42):
        self.D = discriminator
        self.xmu = xmu.astype(np.float32)
        self.xsd = xsd.astype(np.float32)
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        # two-layer head: 64 -> 128 (ReLU) -> n_classes
        self.h1 = {"W": _glorot(rng, 64, 128), "b": np.zeros(128, np.float32)}
        self.h2 = {"W": _glorot(rng, 128, n_classes), "b": np.zeros(n_classes, np.float32)}
        self._feat_cache = None
        self._h1_cache = None

    def _features(self, X):
        """Average penultimate D features over all T frames. Returns (B, 64)."""
        B, T, D = X.shape
        x = X.reshape(B * T, D).astype(np.float32)
        xn = (x - self.xmu) / self.xsd
        loh = np.ones((B * T, self.n_classes), np.float32) / self.n_classes
        feat = self.D.penultimate(xn, loh)          # (B*T, 64)
        return feat.reshape(B, T, 64).mean(1)        # (B, 64)

    def forward(self, X, train=False):
        feat = self._features(X)
        self._feat_cache = feat
        h1 = feat @ self.h1["W"] + self.h1["b"]
        h1_relu = np.maximum(h1, 0)
        self._h1_cache = (h1, h1_relu)
        return h1_relu @ self.h2["W"] + self.h2["b"]

    def predict_proba(self, X):
        z = self.forward(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def backward(self, dlogits):
        h1, h1_relu = self._h1_cache
        g_h2 = {"W": h1_relu.T @ dlogits, "b": dlogits.sum(0)}
        dh1 = (dlogits @ self.h2["W"].T) * (h1 > 0)
        g_h1 = {"W": self._feat_cache.T @ dh1, "b": dh1.sum(0)}
        return g_h1, g_h2

    def n_params(self):
        return (sum(v.size for v in self.h1.values()) +
                sum(v.size for v in self.h2.values()))

    # ── save / load ─────────────────────────────────────────
    def state(self):
        d = {"kind": np.array("gan"),
             "xmu": self.xmu, "xsd": self.xsd,
             "h1.W": self.h1["W"], "h1.b": self.h1["b"],
             "h2.W": self.h2["W"], "h2.b": self.h2["b"]}
        d.update(self.D.state())
        return d

    @classmethod
    def from_state(cls, z):
        D = Discriminator()
        D.load_state(z)
        m = cls(D, z["xmu"], z["xsd"])
        if "h1.W" in z:
            m.h1["W"] = np.asarray(z["h1.W"], np.float32)
            m.h1["b"] = np.asarray(z["h1.b"], np.float32)
            m.h2["W"] = np.asarray(z["h2.W"], np.float32)
            m.h2["b"] = np.asarray(z["h2.b"], np.float32)
        # old format (head.W / head.b) is silently ignored -> head re-initialised randomly
        return m


def fit_gan_head(model, train, val, epochs=12, patience=4,
                 batch=256, lr=1e-3, class_weights=None, log=print, seed=42):
    """Fine-tune only the GANClassifier head (backbone frozen). Supports two-layer head."""
    from . import nn as _nn
    (Xt, yt), (Xv, yv) = train, val
    rng = np.random.default_rng(seed)
    opt = _Adam([model.h1, model.h2], lr=lr)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        t0 = time.time(); idx = rng.permutation(len(Xt)); tot = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            logits = model.forward(Xt[b], True)
            loss, d = _nn.softmax_ce(logits, yt[b], class_weights)
            g_h1, g_h2 = model.backward(d)
            opt.step([g_h1, g_h2])
            tot += loss * len(b)
        va = float((model.predict_proba(Xv).argmax(1) == yv).mean())
        hist.append((tot / len(idx), va))
        log(f"  [gan]  epoch {ep+1:2d}  loss {tot/len(idx):.4f}  val_acc {va:.4f}  ({time.time()-t0:.1f}s)")
        if va > best:
            best, bad = va, 0
            best_state = {k: v.copy() for k, v in model.state().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    restored = GANClassifier.from_state(best_state)
    model.h1, model.h2, model.D = restored.h1, restored.h2, restored.D
    return {"best_val_acc": best, "epochs": len(hist), "history": hist}
