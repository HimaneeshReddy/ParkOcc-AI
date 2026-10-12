"""Frame -> per-frame occupancy ratio (the 'feature extractor' stage), plus image/video loading.

PROTOTYPE NOTE: the report's design uses a CNN here. This sandbox has no deep-learning framework, so the stage is
a small learned regressor (MLP, numpy inference) on hand-crafted image statistics. The interface
(`FrameExtractor.ratio(frame) -> float`) is the swap-in point for a real CNN.
"""
import numpy as np, cv2, os, tempfile
from . import WINDOW

SIZE = (320, 200)


def frame_stats(bgr):
    img = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150) > 0
    sat = hsv[..., 1].astype(np.float32); val = hsv[..., 2].astype(np.float32)
    med = np.median(gray)
    dev = np.abs(gray.astype(np.float32) - med)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    f = [edges.mean(), sat.mean() / 255, (sat > 70).mean(), (val < 60).mean(), (val > 215).mean(),
         np.log1p(lap) / 8, gray.std() / 64, (dev > 25).mean(), (dev > 45).mean(), med / 255]
    gh, gw = 4, 6; hh, ww = SIZE[1] // gh, SIZE[0] // gw
    cell = {"edge": [], "sat": [], "std": [], "dev": []}
    for i in range(gh):
        for j in range(gw):
            sl = (slice(i * hh, (i + 1) * hh), slice(j * ww, (j + 1) * ww))
            cell["edge"].append(edges[sl].mean()); cell["sat"].append(sat[sl].mean() / 255)
            cell["std"].append(gray[sl].std() / 64); cell["dev"].append((dev[sl] > 30).mean())
    for k in cell:
        a = np.array(cell[k]); f += [a.mean(), a.std(), *np.quantile(a, [0.1, 0.5, 0.9])]
    return np.array(f, np.float32)


def unusable(bgr):
    """Quality flag (the report's missing-frame indicator): too dark, too flat or too blurred to trust."""
    g = cv2.cvtColor(cv2.resize(bgr, SIZE), cv2.COLOR_BGR2GRAY)
    return bool(g.std() < 8 or g.mean() < 25 or cv2.Laplacian(g, cv2.CV_32F).var() < 4)


class FrameExtractor:
    def __init__(self, params):
        self.mu, self.sd = params["mu"], params["sd"]
        self.Ws = [params[f"W{i}"] for i in range(3)]; self.bs = [params[f"b{i}"] for i in range(3)]

    @classmethod
    def load(cls, path):
        z = np.load(path); return cls({k: z[k] for k in z.files})

    def ratio_from_stats(self, S):
        a = (S - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = a @ W + b
            if i < len(self.Ws) - 1: a = np.maximum(a, 0)
        return np.clip(a[..., 0], 0, 1)

    def ratio(self, bgr):
        return float(self.ratio_from_stats(frame_stats(bgr)[None])[0])


# ---------------- media loading ----------------
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


class MediaError(ValueError):
    pass


def load_frames(data: bytes, filename: str, n=WINDOW):
    """Return `n` BGR frames: a video is sampled evenly; a single image is repeated."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXT:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise MediaError("Could not decode the image file.")
        return [img] * n, "image"
    if ext in VIDEO_EXT:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t: t.write(data); path = t.name
        try:
            cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
            if total > 0:
                for idx in np.linspace(0, max(total - 1, 0), n).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read()
                    if ok: frames.append(fr)
            else:
                while True:
                    ok, fr = cap.read()
                    if not ok: break
                    frames.append(fr)
                frames = [frames[i] for i in np.linspace(0, len(frames) - 1, n).astype(int)] if frames else []
            cap.release()
        finally:
            os.unlink(path)
        if len(frames) < n: raise MediaError("Could not read enough frames from the video.")
        return frames, "video"
    raise MediaError(f"Unsupported file type '{ext or 'none'}'. Use an image ({', '.join(sorted(IMAGE_EXT))}) or a video ({', '.join(sorted(VIDEO_EXT))}).")
