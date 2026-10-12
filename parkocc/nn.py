"""Tiny numpy deep-learning kit: RNN / LSTM / GRU cells with full BPTT, a dense head, softmax-CE, Adam.

Written in numpy because the build environment has no DL framework; the maths follows the report (Elman RNN,
LSTM with forget/input/output gates, GRU with update/reset gates). Parameter counts differ slightly from a
Keras implementation (e.g. one GRU bias vector instead of two).
"""
import numpy as np, time

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def glorot(rng, i, o): return rng.uniform(-1, 1, (i, o)).astype(np.float32) * np.sqrt(6.0 / (i + o))


class RNNCell:
    kind = "rnn"
    def __init__(self, D, H, rng):
        self.D, self.H = D, H
        self.p = {"Wx": glorot(rng, D, H), "Wh": glorot(rng, H, H), "b": np.zeros(H, np.float32)}
    def forward(self, X):
        B, T, _ = X.shape; h = np.zeros((B, self.H), np.float32); self.c = []
        for t in range(T):
            hn = np.tanh(X[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]); self.c.append((X[:, t], h, hn)); h = hn
        return h
    def backward(self, dh):
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        for x, hp, hn in reversed(self.c):
            dz = dh * (1 - hn ** 2); g["Wx"] += x.T @ dz; g["Wh"] += hp.T @ dz; g["b"] += dz.sum(0); dh = dz @ self.p["Wh"].T
        return g


class LSTMCell:
    kind = "lstm"
    def __init__(self, D, H, rng):
        self.D, self.H = D, H
        b = np.zeros(4 * H, np.float32); b[H:2 * H] = 1.0          # forget-gate bias 1
        self.p = {"W": glorot(rng, D + H, 4 * H), "b": b}
    def forward(self, X):
        B, T, _ = X.shape; H = self.H; h = np.zeros((B, H), np.float32); c = np.zeros((B, H), np.float32); self.c = []
        for t in range(T):
            xc = np.concatenate([X[:, t], h], 1); z = xc @ self.p["W"] + self.p["b"]
            i, f, g, o = sigmoid(z[:, :H]), sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:])
            cn = f * c + i * g; tc = np.tanh(cn); hn = o * tc
            self.c.append((xc, c, i, f, g, o, tc)); h, c = hn, cn
        return h
    def backward(self, dh):
        H = self.H; g_ = {k: np.zeros_like(v) for k, v in self.p.items()}; dc = np.zeros_like(dh)
        for xc, cp, i, f, g, o, tc in reversed(self.c):
            do = dh * tc; dc = dc + dh * o * (1 - tc ** 2)
            di, df, dg = dc * g, dc * cp, dc * i
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], 1)
            g_["W"] += xc.T @ dz; g_["b"] += dz.sum(0); dh = (dz @ self.p["W"].T)[:, self.D:]; dc = dc * f
        return g_


class GRUCell:
    kind = "gru"
    def __init__(self, D, H, rng):
        self.D, self.H = D, H
        self.p = {"Wx": glorot(rng, D, 3 * H), "Uzr": glorot(rng, H, 2 * H), "Un": glorot(rng, H, H), "b": np.zeros(3 * H, np.float32)}
    def forward(self, X):
        B, T, _ = X.shape; H = self.H; h = np.zeros((B, H), np.float32); self.c = []
        for t in range(T):
            x = X[:, t]; a = x @ self.p["Wx"] + self.p["b"]
            zr = sigmoid(a[:, :2 * H] + h @ self.p["Uzr"]); z, r = zr[:, :H], zr[:, H:]
            n = np.tanh(a[:, 2 * H:] + (r * h) @ self.p["Un"]); hn = (1 - z) * n + z * h
            self.c.append((x, h, z, r, n)); h = hn
        return h
    def backward(self, dh):
        H = self.H; g = {k: np.zeros_like(v) for k, v in self.p.items()}
        for x, hp, z, r, n in reversed(self.c):
            dn = dh * (1 - z); dz = dh * (hp - n); dhp = dh * z
            dan = dn * (1 - n ** 2); g["Un"] += (r * hp).T @ dan; drh = dan @ self.p["Un"].T
            dr = drh * hp; dhp += drh * r
            daz, dar = dz * z * (1 - z), dr * r * (1 - r)
            dazr = np.concatenate([daz, dar], 1); g["Uzr"] += hp.T @ dazr; dhp += dazr @ self.p["Uzr"].T
            da = np.concatenate([dazr, dan], 1); g["Wx"] += x.T @ da; g["b"] += da.sum(0); dh = dhp
        return g


CELLS = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}


class Dense:
    def __init__(self, i, o, rng): self.p = {"W": glorot(rng, i, o), "b": np.zeros(o, np.float32)}
    def forward(self, x): self.x = x; return x @ self.p["W"] + self.p["b"]
    def backward(self, d): return d @ self.p["W"].T, {"W": self.x.T @ d, "b": d.sum(0)}


class Classifier:
    """kind in {rnn,lstm,gru}: cell over the 6-step window, last state -> Dense(64) ReLU -> Dropout -> Dense(5).
    kind == 'baseline': no temporal memory, the LAST frame's vector goes straight into the same head."""
    def __init__(self, kind, D=9, H=64, n_classes=5, dropout=0.2, seed=42):
        rng = np.random.default_rng(seed); self.kind, self.H, self.dropout = kind, H, dropout
        self.cell = CELLS[kind](D, H, rng) if kind != "baseline" else None
        self.d1 = Dense(H if self.cell else D, 64, rng); self.d2 = Dense(64, n_classes, rng); self.rng = rng

    def layers(self):
        return ([self.cell] if self.cell else []) + [self.d1, self.d2]

    def n_params(self): return int(sum(v.size for L in self.layers() for v in L.p.values()))

    def forward(self, X, train=False):
        h = self.cell.forward(X) if self.cell else X[:, -1]
        a = np.maximum(self.d1.forward(h), 0); self.mask = None
        if train and self.dropout > 0:
            self.mask = (self.rng.random(a.shape) > self.dropout).astype(np.float32) / (1 - self.dropout); a = a * self.mask
        self.a = a; return self.d2.forward(a)

    def backward(self, dlogits):
        da, g2 = self.d2.backward(dlogits)
        if self.mask is not None: da = da * self.mask
        da = da * (self.a > 0)
        dh, g1 = self.d1.backward(da); grads = [g1, g2]
        if self.cell: grads = [self.cell.backward(dh)] + grads
        return grads

    def predict_proba(self, X):
        z = self.forward(X, False); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def state(self):
        out = {"kind": np.array(self.kind), "H": np.array(self.H)}
        for n, L in zip(["cell", "d1", "d2"][-len(self.layers()):], self.layers()):
            for k, v in L.p.items(): out[f"{n}.{k}"] = v
        return out

    @classmethod
    def from_state(cls, z):
        kind = str(z["kind"]); m = cls(kind, H=int(z["H"]))
        for n, L in zip(["cell", "d1", "d2"][-len(m.layers()):], m.layers()):
            for k in L.p: L.p[k] = z[f"{n}.{k}"].astype(np.float32)
        return m


def softmax_ce(logits, y, w=None):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
    sw = np.ones(len(y), np.float32) if w is None else w[y]
    loss = -(np.log(p[np.arange(len(y)), y] + 1e-9) * sw).sum() / sw.sum()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d *= (sw / sw.sum())[:, None]
    return loss, d


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, clip=5.0):
        self.m, self.lr, self.b1, self.b2, self.clip, self.t = model, lr, b1, b2, clip, 0
        self.s = [{k: (np.zeros_like(v), np.zeros_like(v)) for k, v in L.p.items()} for L in model.layers()]
    def step(self, grads):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for gl in grads for g in gl.values()))
        sc = min(1.0, self.clip / (norm + 1e-9))
        for L, gl, st in zip(self.m.layers(), grads, self.s):
            for k in L.p:
                g = gl[k] * sc; m, v = st[k]
                m[:] = self.b1 * m + (1 - self.b1) * g; v[:] = self.b2 * v + (1 - self.b2) * g * g
                mh = m / (1 - self.b1 ** self.t); vh = v / (1 - self.b2 ** self.t)
                L.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def metrics(y, pred, n=5):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, pred): cm[a, b] += 1
    tp = np.diag(cm).astype(float); P = tp / np.maximum(cm.sum(0), 1); R = tp / np.maximum(cm.sum(1), 1)
    F = 2 * P * R / np.maximum(P + R, 1e-9)
    under = int(np.tril(cm, -1).sum()); errs = int(cm.sum() - tp.sum())
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(F.mean()), "macro_precision": float(P.mean()),
            "macro_recall": float(R.mean()), "per_class_f1": F.round(4).tolist(), "per_class_recall": R.round(4).tolist(),
            "full_recall": float(R[-1]), "confusion": cm.tolist(), "errors": errs, "under_estimates": under,
            "under_share": float(under / errs) if errs else 0.0,
            "empty_full_confusions": int(cm[0, 4] + cm[4, 0])}


def fit(model, train, val, epochs=12, patience=3, batch=256, lr=1e-3, class_weights=None, log=print, seed=42):
    (Xt, yt), (Xv, yv) = train, val; rng = np.random.default_rng(seed); opt = Adam(model, lr)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        t0 = time.time(); idx = rng.permutation(len(Xt)); tot = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]; loss, d = softmax_ce(model.forward(Xt[b], True), yt[b], class_weights)
            opt.step(model.backward(d)); tot += loss * len(b)
        va = float((model.predict_proba(Xv).argmax(1) == yv).mean()); hist.append((tot / len(idx), va))
        log(f"  [{model.kind}] epoch {ep + 1:2d}  loss {tot / len(idx):.4f}  val_acc {va:.4f}  ({time.time() - t0:.1f}s)")
        if va > best: best, bad = va, 0; best_state = {k: v.copy() for k, v in model.state().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.__init__(model.kind, H=model.H, seed=seed); 
    restored = type(model).from_state(best_state); model.cell, model.d1, model.d2 = restored.cell, restored.d1, restored.d2
    return {"best_val_acc": best, "epochs": len(hist), "history": hist}
