"""Pure-numpy Convolutional Neural Network for parking-lot feature extraction.

Architecture (CNN feature extractor):
    Input : (B, H, W, 1)   greyscale patch (50x80 default)
    Conv1  : 8 filters, 3x3, stride 1, ReLU
    MaxPool: 2x2, stride 2
    Conv2  : 16 filters, 3x3, stride 1, ReLU
    MaxPool: 2x2, stride 2
    Flatten -> Dense(32) ReLU -> Dense(1) sigmoid  -> occupancy ratio

KmeansCNNClassifier:
    MLP encoder 9->64->32  +  K-means soft assignment (K centroids)
    -> Dense(5) softmax
    The K-means centroids are re-fitted from embeddings after each training epoch.
"""
import numpy as np, time

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

# ─────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────

def _glorot(rng, rows, cols):
    lim = np.sqrt(6.0 / (rows + cols))
    return rng.uniform(-lim, lim, (rows, cols)).astype(np.float32)


def _im2col(x, kh, kw, stride=1):
    """x: (B,H,W,C) -> (B, Oh, Ow, kh*kw*C)."""
    B, H, W, C = x.shape
    Oh = (H - kh) // stride + 1
    Ow = (W - kw) // stride + 1
    col = np.zeros((B, Oh, Ow, kh * kw * C), np.float32)
    for i in range(kh):
        for j in range(kw):
            col[:, :, :, (i * kw + j) * C:(i * kw + j + 1) * C] = \
                x[:, i:i + Oh * stride:stride, j:j + Ow * stride:stride, :]
    return col


def _col2im(dcol, x_shape, kh, kw, stride=1):
    B, H, W, C = x_shape
    Oh = (H - kh) // stride + 1
    Ow = (W - kw) // stride + 1
    dx = np.zeros(x_shape, np.float32)
    for i in range(kh):
        for j in range(kw):
            dx[:, i:i + Oh * stride:stride, j:j + Ow * stride:stride, :] += \
                dcol[:, :, :, (i * kw + j) * C:(i * kw + j + 1) * C]
    return dx


# ─────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────

class _Conv:
    def __init__(self, Cin, F, kh=3, kw=3, stride=1, rng=None):
        rng = rng or np.random.default_rng(0)
        self.kh, self.kw, self.stride = kh, kw, stride
        fan = kh * kw * Cin + kh * kw * F
        lim = np.sqrt(6.0 / fan)
        self.p = {"W": rng.uniform(-lim, lim, (kh * kw * Cin, F)).astype(np.float32),
                  "b": np.zeros(F, np.float32)}

    def forward(self, x):
        self._col = _im2col(x, self.kh, self.kw, self.stride)
        B, Oh, Ow, _ = self._col.shape
        out = self._col.reshape(B * Oh * Ow, -1) @ self.p["W"] + self.p["b"]
        self._xsh = x.shape
        return out.reshape(B, Oh, Ow, self.p["W"].shape[1])

    def backward(self, dout):
        B, Oh, Ow, F = dout.shape
        d = dout.reshape(B * Oh * Ow, F)
        cf = self._col.reshape(B * Oh * Ow, -1)
        gW = cf.T @ d
        gb = d.sum(0)
        dcol = (d @ self.p["W"].T).reshape(self._col.shape)
        dx = _col2im(dcol, self._xsh, self.kh, self.kw, self.stride)
        return dx, {"W": gW, "b": gb}


class _Pool2x2:
    def forward(self, x):
        self._in_shape = x.shape
        B, H, W, C = x.shape
        Oh, Ow = H // 2, W // 2
        xr = x[:, :Oh*2, :Ow*2, :].reshape(B, Oh, 2, Ow, 2, C)
        out = xr.max(axis=(2, 4))
        self._mask = (xr == out[:, :, None, :, None, :])
        return out

    def backward(self, dout):
        B, H, W, C = self._in_shape
        Oh, Ow = H // 2, W // 2
        dx = np.zeros(self._in_shape, np.float32)
        pooled_dx = (self._mask * dout[:, :, None, :, None, :]).reshape(B, Oh*2, Ow*2, C)
        dx[:, :Oh*2, :Ow*2, :] = pooled_dx
        return dx


class _Dense:
    def __init__(self, i, o, rng):
        self.p = {"W": _glorot(rng, i, o), "b": np.zeros(o, np.float32)}

    def forward(self, x):
        self._x = x
        return x @ self.p["W"] + self.p["b"]

    def backward(self, d):
        return d @ self.p["W"].T, {"W": self._x.T @ d, "b": d.sum(0)}


# ─────────────────────────────────────────────────────────────
# CNN Feature Extractor (numpy, replaces MLP extractor)
# ─────────────────────────────────────────────────────────────
PATCH_H, PATCH_W = 50, 80   # greyscale patch size


def _preprocess(bgr, ph=PATCH_H, pw=PATCH_W):
    import cv2
    gray = cv2.cvtColor(cv2.resize(bgr, (pw, ph)), cv2.COLOR_BGR2GRAY)
    return (gray.astype(np.float32) / 255.0)[None, :, :, None]   # (1,H,W,1)


class CNNExtractor:
    """Tiny 2-conv numpy CNN: image -> 32-d embedding + 0-1 occupancy ratio.

    Used as a drop-in replacement for FrameExtractor in frames.py.
    The same interface is preserved: .ratio(bgr) -> float
    """

    def __init__(self, rng=None):
        rng = np.random.default_rng(42) if rng is None else rng
        self.conv1 = _Conv(1, 8, 3, 3, rng=rng)
        self.conv2 = _Conv(8, 16, 3, 3, rng=rng)
        # Compute exact flattened feature dimension dynamically
        dummy = np.zeros((1, PATCH_H, PATCH_W, 1), np.float32)
        p1 = _Pool2x2().forward(self.conv1.forward(dummy))
        p2 = _Pool2x2().forward(self.conv2.forward(p1))
        flat = p2.size
        self.d1 = _Dense(flat, 32, rng)
        self.d2 = _Dense(32, 1, rng)
        self._flat = flat

    def _forward_full(self, x):
        """Returns (ratio (B,1), embed (B,32)) and caches intermediate tensors."""
        a1 = self.conv1.forward(x)
        self._pool1 = _Pool2x2(); p1 = self._pool1.forward(np.maximum(a1, 0))
        self._a1, self._relu1 = a1, (a1 > 0)
        a2 = self.conv2.forward(p1)
        self._pool2 = _Pool2x2(); p2 = self._pool2.forward(np.maximum(a2, 0))
        self._a2, self._relu2 = a2, (a2 > 0)
        flat = p2.reshape(p2.shape[0], -1)
        self._flat_shape = p2.shape
        h = np.maximum(self.d1.forward(flat), 0)
        self._h, self._h_relu = h, (h > 0)
        ratio = sigmoid(self.d2.forward(h))
        return ratio, h

    def forward_embed(self, bgr_batch):
        """bgr frames (list) -> (B, 32) embeddings."""
        xs = np.concatenate([_preprocess(f) for f in bgr_batch], axis=0)
        _, embeds = self._forward_full(xs)
        return embeds

    def ratio(self, bgr):
        x = _preprocess(bgr)
        r, _ = self._forward_full(x)
        return float(np.clip(r[0, 0], 0, 1))

    def ratio_from_batch(self, bgr_list):
        xs = np.concatenate([_preprocess(f) for f in bgr_list], axis=0)
        r, _ = self._forward_full(xs)
        return np.clip(r[:, 0], 0, 1)

    def backward(self, dr):
        """dr: (B,1) dLoss/dratio. Returns list of grad dicts."""
        dh2, g2 = self.d2.backward(dr)
        dh2 = dh2 * self._h_relu
        dflat, g1 = self.d1.backward(dh2)
        B, Oh, Ow, C = self._flat_shape
        dp2 = dflat.reshape(B, Oh, Ow, C)
        da2_pool = self._pool2.backward(dp2)
        da2 = da2_pool * self._relu2
        dp1, gc2 = self.conv2.backward(da2)
        da1_pool = self._pool1.backward(dp1)
        da1 = da1_pool * self._relu1
        _, gc1 = self.conv1.backward(da1)
        return [gc1, gc2, g1, g2]

    def all_layers(self):
        return [self.conv1, self.conv2, self.d1, self.d2]

    # ── serialisation ─────────────────────────────────────────
    def state(self):
        d = {}
        for prefix, layer in [("c1", self.conv1), ("c2", self.conv2),
                               ("d1", self.d1), ("d2", self.d2)]:
            for k, v in layer.p.items():
                d[f"{prefix}.{k}"] = v
        return d

    def load_state(self, z):
        for prefix, layer in [("c1", self.conv1), ("c2", self.conv2),
                               ("d1", self.d1), ("d2", self.d2)]:
            for k in layer.p:
                key = f"{prefix}.{k}"
                if key in z:
                    layer.p[k] = np.asarray(z[key], np.float32)

    @classmethod
    def from_state(cls, z):
        m = cls()
        m.load_state(z)
        return m


# ─────────────────────────────────────────────────────────────
# Adam for CNN layers
# ─────────────────────────────────────────────────────────────

class _Adam:
    def __init__(self, layers, lr=1e-3, b1=0.9, b2=0.999, clip=5.0):
        self.layers = layers
        self.lr, self.b1, self.b2, self.clip, self.t = lr, b1, b2, clip, 0
        self.s = [{k: (np.zeros_like(v), np.zeros_like(v))
                   for k, v in L.p.items()} for L in layers]

    def step(self, grads):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for gl in grads for g in gl.values()) + 1e-12)
        sc = min(1.0, self.clip / norm)
        for L, gl, st in zip(self.layers, grads, self.s):
            for k in L.p:
                g = gl[k] * sc
                m, v = st[k]
                m[:] = self.b1 * m + (1 - self.b1) * g
                v[:] = self.b2 * v + (1 - self.b2) * g * g
                mh = m / (1 - self.b1 ** self.t)
                vh = v / (1 - self.b2 ** self.t)
                L.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def fit_cnn_extractor(cnn, images, ratios, epochs=8, batch=32, lr=1e-3, log=print):
    """Train CNNExtractor as a regressor.
    images: list of BGR frames (same length as ratios)
    ratios: 1-D float array of ground-truth occupancy ratios
    """
    opt = _Adam(cnn.all_layers(), lr=lr)
    N = len(images)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        t0 = time.time(); tot = 0.0
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            bi = idx[s:s + batch]
            xs = np.concatenate([_preprocess(images[i]) for i in bi], axis=0)
            yt = ratios[bi].astype(np.float32)[:, None]
            r, _ = cnn._forward_full(xs)
            loss = float(((r - yt) ** 2).mean())
            dr = 2 * (r - yt) / len(bi)
            grads = cnn.backward(dr)
            opt.step(grads)
            tot += loss * len(bi)
        log(f"  [CNN extractor] epoch {ep+1}  MSE {tot/N:.5f}  ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────
# K-means + CNN Classifier  (new selectable model)
# ─────────────────────────────────────────────────────────────

class KmeansCNNClassifier:
    """K-means clustering over MLP-encoded features, then a dense classification head.

    Input: window (B, T, 9)  (same as RNN/LSTM/GRU)
    Pipeline:
        last_frame = X[:, -1, :]          (B, 9)
        embed      = MLP(last_frame)      (B, 32)  -- the 'CNN branch' stand-in
        soft_assign= softmax(-dist/T)     (B, K)
        logits     = head(soft_assign)    (B, 5)
    """
    kind = "kmeans_cnn"

    def __init__(self, K=10, D=9, embed_dim=32, n_classes=5, seed=42):
        rng = np.random.default_rng(seed)
        self.K, self.D, self.embed_dim, self.n_classes = K, D, embed_dim, n_classes
        # MLP encoder: 9->64->32
        self.enc = [
            {"W": _glorot(rng, D, 64),       "b": np.zeros(64, np.float32)},
            {"W": _glorot(rng, 64, embed_dim), "b": np.zeros(embed_dim, np.float32)},
        ]
        # K-means centroids (K, embed_dim)
        self.centroids = rng.standard_normal((K, embed_dim)).astype(np.float32)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9
        # Classification head: K -> n_classes
        self.head = {"W": _glorot(rng, K, n_classes), "b": np.zeros(n_classes, np.float32)}
        self.rng = rng
        self._enc_cache = []

    # ── encoder ────────────────────────────────────────────
    def _encode(self, x):
        self._enc_cache = []
        a = x
        for i, p in enumerate(self.enc):
            z = a @ p["W"] + p["b"]
            self._enc_cache.append((a, z))
            a = np.maximum(z, 0) if i < len(self.enc) - 1 else z
        return a

    def _encode_backward(self, dembed):
        grads = []
        d = dembed
        for i in range(len(self.enc) - 1, -1, -1):
            a, z = self._enc_cache[i]
            if i < len(self.enc) - 1:
                d = d * (z > 0)
            gW = a.T @ d
            gb = d.sum(0)
            d = d @ self.enc[i]["W"].T
            grads.insert(0, {"W": gW, "b": gb})
        return grads

    # ── soft cluster assignment ─────────────────────────────
    def _soft_assign(self, embed):
        diff = embed[:, None, :] - self.centroids[None, :, :]   # (B,K,E)
        d2 = (diff ** 2).sum(-1)                                # (B,K)
        T = float(self.embed_dim ** 0.5)
        e = np.exp(-d2 / T - d2.min(1, keepdims=True) / T)     # numerically stable
        s = e / (e.sum(1, keepdims=True) + 1e-9)
        self._sa_embed, self._sa_diff, self._sa_s, self._sa_e = embed, diff, s, e
        return s

    def _soft_assign_backward(self, ds):
        T = float(self.embed_dim ** 0.5)
        s = self._sa_s; e = self._sa_e
        # exact softmax backward: d_i = s_i * (ds_i - sum_j s_j * ds_j)
        de = s * (ds - (ds * s).sum(1, keepdims=True))
        # d/d_d2 : e_i = exp(-d2_i/T), so de_i/d_d2_i = -1/T * e_i
        dd2 = de * (-1.0 / T) * e / (e.sum(1, keepdims=True) + 1e-9)
        # d/d_embed: d2_ij = ||embed_i - centroid_j||^2, so grad = 2*(embed - centroid)
        dembed = (dd2[:, :, None] * 2 * self._sa_diff).sum(1)   # (B,E)
        return dembed

    # ── classification head ─────────────────────────────────
    def _head_fwd(self, s):
        self._h_s = s
        return s @ self.head["W"] + self.head["b"]

    def _head_bwd(self, d):
        ds = d @ self.head["W"].T
        return ds, {"W": self._h_s.T @ d, "b": d.sum(0)}

    # ── full model ──────────────────────────────────────────
    def forward(self, X, train=False):
        x = X[:, -1, :]
        embed = self._encode(x)
        s = self._soft_assign(embed)
        return self._head_fwd(s)

    def predict_proba(self, X):
        z = self.forward(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def backward(self, dlogits):
        ds, g_head = self._head_bwd(dlogits)
        dembed = self._soft_assign_backward(ds)
        g_enc = self._encode_backward(dembed)
        return g_enc + [g_head]

    # ── centroid update (hard K-means, run each epoch) ──────
    def update_centroids(self, X_all):
        x = X_all[:, -1, :]
        embed = self._encode(x)
        d2 = ((embed[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(1)
        for k in range(self.K):
            m = embed[assign == k]
            if len(m) > 0:
                self.centroids[k] = m.mean(0)

    # ── param layers (for Adam) ─────────────────────────────
    @property
    def _layers(self):
        class _L:
            def __init__(self, p): self.p = p
        return [_L(e) for e in self.enc] + [_L(self.head)]

    def n_params(self):
        return (sum(v.size for e in self.enc for v in e.values()) +
                sum(v.size for v in self.head.values()))

    # ── save / load ─────────────────────────────────────────
    def state(self):
        d = {"kind": np.array("kmeans_cnn"), "K": np.array(self.K),
             "centroids": self.centroids}
        for i, e in enumerate(self.enc):
            for k, v in e.items():
                d[f"enc{i}.{k}"] = v
        for k, v in self.head.items():
            d[f"head.{k}"] = v
        return d

    @classmethod
    def from_state(cls, z):
        K = int(z["K"])
        m = cls(K=K)
        m.centroids = np.asarray(z["centroids"], np.float32)
        for i in range(len(m.enc)):
            for k in m.enc[i]:
                m.enc[i][k] = np.asarray(z[f"enc{i}.{k}"], np.float32)
        for k in m.head:
            m.head[k] = np.asarray(z[f"head.{k}"], np.float32)
        return m


def fit_kmeans_cnn(model, train, val, epochs=15, patience=4,
                   batch=256, lr=5e-4, class_weights=None, log=print, seed=42):
    """Train KmeansCNNClassifier with Adam + K-means centroid refresh each epoch."""
    from . import nn as _nn
    (Xt, yt), (Xv, yv) = train, val
    rng = np.random.default_rng(seed)
    layers = model._layers
    opt = _Adam(layers, lr=lr)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        t0 = time.time(); idx = rng.permutation(len(Xt)); tot = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            logits = model.forward(Xt[b], True)
            loss, d = _nn.softmax_ce(logits, yt[b], class_weights)
            grads = model.backward(d)
            opt.step(grads)
            tot += loss * len(b)
        model.update_centroids(Xt)
        va = float((model.predict_proba(Xv).argmax(1) == yv).mean())
        hist.append((tot / len(idx), va))
        log(f"  [kmeans_cnn] epoch {ep+1:2d}  loss {tot/len(idx):.4f}  val_acc {va:.4f}  ({time.time()-t0:.1f}s)")
        if va > best:
            best, bad = va, 0
            best_state = {k: (v.copy() if hasattr(v, "copy") else v)
                          for k, v in model.state().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    restored = KmeansCNNClassifier.from_state(best_state)
    model.enc, model.head, model.centroids = restored.enc, restored.head, restored.centroids
    return {"best_val_acc": best, "epochs": len(hist), "history": hist}
