"""Synthetic top-down parking-lot renderer.

Used for (a) training the frame feature extractor and (b) the 'Generate sample clip' demo so the app works
without any external footage. Real CCTV frames will be harder: this is a prototype.
"""
import numpy as np, cv2

ROWS, COLS = 3, 12
N_SLOTS = ROWS * COLS
W, H = 480, 300
SLOT_W, SLOT_H = 34, 62
X0 = 36
ROW_Y = [20, 118, 216]
PALETTE = [(235, 235, 235), (30, 30, 30), (150, 150, 155), (60, 60, 190), (180, 90, 40), (40, 120, 60),
           (200, 200, 205), (70, 130, 200), (20, 20, 90), (110, 110, 115), (40, 180, 220)]


def slot_rect(i):
    r, c = divmod(i, COLS)
    return X0 + c * SLOT_W, ROW_Y[r]


def _base(rng):
    g = int(rng.integers(70, 125))
    img = np.full((H, W, 3), g, np.uint8)
    for i in range(N_SLOTS):
        x, y = slot_rect(i)
        cv2.rectangle(img, (x, y), (x + SLOT_W, y + SLOT_H), (200, 200, 200), 1)
    return img


def draw_cars(img, mask, rng):
    for i in np.flatnonzero(mask):
        x, y = slot_rect(i)
        jx, jy = int(rng.integers(-2, 3)), int(rng.integers(-3, 4))
        col = PALETTE[int(rng.integers(len(PALETTE)))]
        cv2.rectangle(img, (x + 7 + jx, y + 7 + jy), (x + SLOT_W - 3 + jx, y + SLOT_H - 3 + jy), (25, 25, 25), -1)  # shadow
        cv2.rectangle(img, (x + 4 + jx, y + 4 + jy), (x + SLOT_W - 6 + jx, y + SLOT_H - 6 + jy), col, -1)
        dark = tuple(int(c * 0.45) for c in col)
        cv2.rectangle(img, (x + 8 + jx, y + 16 + jy), (x + SLOT_W - 10 + jx, y + 26 + jy), dark, -1)   # windscreen
        cv2.rectangle(img, (x + 8 + jx, y + 40 + jy), (x + SLOT_W - 10 + jx, y + 46 + jy), dark, -1)   # rear window
    return img


def degrade(img, rng, condition=0, strength=1.0):
    """Camera effects: pose change, lighting, weather, glare, occlusion, sensor noise."""
    out = img.copy()
    s = float(rng.uniform(0.9, 1.1)); a = float(rng.uniform(-4, 4)) * strength
    M = cv2.getRotationMatrix2D((W / 2, H / 2), a, s); M[:, 2] += rng.uniform(-10, 10, 2) * strength
    out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    f = out.astype(np.float32)
    f = f * rng.uniform(0.8, 1.15) + rng.uniform(-12, 12)
    if condition == 1:      # overcast: desaturated, flatter
        gray = f.mean(axis=2, keepdims=True); f = 0.55 * f + 0.45 * gray; f = (f - 128) * 0.85 + 120
    if condition == 2:      # rain: darker, blurred, streaks
        f = f * 0.72
        out = cv2.GaussianBlur(np.clip(f, 0, 255).astype(np.uint8), (5, 5), 0); f = out.astype(np.float32)
        for _ in range(int(60 * strength)):
            x = int(rng.integers(0, W)); y = int(rng.integers(0, H)); L = int(rng.integers(8, 20))
            cv2.line(f, (x, y), (x - 2, y + L), (200, 200, 200), 1)
    if rng.random() < 0.3 * strength:     # glare blobs
        for _ in range(int(rng.integers(1, 3))):
            cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H)); r = int(rng.integers(25, 70))
            blob = np.zeros((H, W), np.float32); cv2.circle(blob, (cx, cy), r, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), r / 2.5)
            f += blob[..., None] * rng.uniform(70, 150)
    if rng.random() < 0.2 * strength:     # occlusion (tree / truck / shadow band)
        x, y = int(rng.integers(0, W - 80)), int(rng.integers(0, H - 40))
        w, h = int(rng.integers(40, 120)), int(rng.integers(20, 60))
        f[y:y + h, x:x + w] *= rng.uniform(0.25, 0.6)
    f += rng.normal(0, 5 * strength, f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.1 * strength: out = cv2.GaussianBlur(out, (7, 7), 0)    # motion blur / focus
    return out


def render_lot(mask, rng, condition=0, noise=True):
    img = draw_cars(_base(rng), mask, rng)
    return degrade(img, rng, condition, 1.0 if noise else 0.25)


def render_clip(target_ratio, rng, condition=0, noise=True, n_frames=6, k_range=None):
    """Six frames of the same lot: a stable car set with 0-2 arrivals/departures between frames.
    k_range=(lo, hi) keeps the occupied-slot count inside a band so the clip stays in one occupancy class."""
    k = int(round(float(np.clip(target_ratio, 0, 1)) * N_SLOTS))
    if k_range: k = int(np.clip(k, *k_range))
    mask = np.zeros(N_SLOTS, bool); mask[rng.choice(N_SLOTS, k, replace=False)] = True
    frames, ratios = [], []
    for t in range(n_frames):
        if t > 0:
            for _ in range(int(rng.integers(0, 3))):
                i = int(rng.integers(N_SLOTS)); new_k = mask.sum() + (-1 if mask[i] else 1)
                if k_range is None or k_range[0] <= new_k <= k_range[1]: mask[i] = not mask[i]
        frames.append(render_lot(mask, rng, condition, noise)); ratios.append(mask.mean())
    return frames, np.array(ratios, np.float32)


def recompress(img, rng, p=0.75):
    """Simulate how real uploads arrive: re-scaled and JPEG-compressed (widths 160-640 px, quality 40-95)."""
    if rng.random() > p: return img
    w = int(rng.choice([160, 240, 320, 480, 640])); h = max(1, int(img.shape[0] * w / img.shape[1]))
    y = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", y, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
