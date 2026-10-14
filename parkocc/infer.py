"""Inference: media -> six frames -> per-frame ratios -> (6, 9) window -> all models."""
import os, json, base64, datetime as dt
import numpy as np, cv2
from . import CLASSES, WINDOW, simulate, frames, nn
from . import cnn as cnn_mod
from . import gan as gan_mod

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MODEL_NAMES = {
    "baseline":    "Single-frame baseline",
    "rnn":         "RNN",
    "lstm":        "LSTM",
    "gru":         "GRU",
    "kmeans_cnn":  "K-means + CNN",
    "gan":         "GAN (D-features)",
}
CONDITIONS = {"clear": 0, "overcast": 1, "rain": 2}
ADVICE = {0: "Plenty of space: park without hesitation.", 1: "Space is likely available.",
          2: "Moderately busy: spaces should still be findable.", 3: "Nearly full: expect to search, consider an alternative.",
          4: "Full or close to it: choose another lot."}


def thumb_b64(bgr, width=200, q=72):
    h = int(bgr.shape[0] * width / bgr.shape[1]); im = cv2.resize(bgr, (width, max(h, 1)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q]); return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# Classic models that use nn.Classifier.from_state
_CLASSIC_KINDS = {"baseline", "rnn", "lstm", "gru"}


class Predictor:
    def __init__(self, art_dir=ART):
        self.art_dir = art_dir
        need = ["extractor.npz", "metrics.json"] + [f"{k}.npz" for k in MODEL_NAMES]
        missing = [n for n in need if not os.path.exists(os.path.join(art_dir, n))]
        if missing: raise FileNotFoundError(f"Missing artifacts {missing}. Run:  python -m parkocc.train")
        self.extractor = frames.FrameExtractor.load(os.path.join(art_dir, "extractor.npz"))
        self.models = {}
        for k in MODEL_NAMES:
            path = os.path.join(art_dir, f"{k}.npz")
            z = np.load(path, allow_pickle=False)
            if k in _CLASSIC_KINDS:
                self.models[k] = nn.Classifier.from_state(z)
            elif k == "kmeans_cnn":
                self.models[k] = cnn_mod.KmeansCNNClassifier.from_state(z)
            elif k == "gan":
                self.models[k] = gan_mod.GANClassifier.from_state(z)
        with open(os.path.join(art_dir, "metrics.json")) as _f:
            self.metrics = json.load(_f)

    def window(self, ratios, condition, when, flags, capacity=0.5):
        times = [when - dt.timedelta(minutes=15 * (WINDOW - 1 - i)) for i in range(WINDOW)]
        tod = np.array([t.hour + t.minute / 60 for t in times]); dow = np.array([t.weekday() for t in times])
        f = np.zeros((WINDOW, 9), np.float32); f[:, 0] = ratios; f[:, 1:6] = simulate.time_features(tod, dow)
        f[:, 6] = capacity; f[:, 7] = CONDITIONS[condition] / 2.0; f[:, 8] = np.asarray(flags, np.float32); return f

    def analyse(self, bgr_frames, source="image", condition="clear", when=None, capacity=0.5):
        if condition not in CONDITIONS: raise ValueError("condition must be clear, overcast or rain")
        if len(bgr_frames) != WINDOW: raise ValueError(f"need exactly {WINDOW} frames")
        when = when or dt.datetime.now()
        ratios = np.array([self.extractor.ratio(f) for f in bgr_frames], np.float32)
        flags = [frames.unusable(f) for f in bgr_frames]
        X = self.window(ratios, condition, when, flags, capacity)[None]
        results = {}
        for k, m in self.models.items():
            p = m.predict_proba(X)[0]; c = int(p.argmax())
            results[k] = {"name": MODEL_NAMES[k], "class_index": c, "label": CLASSES[c], "confidence": float(p[c]),
                          "probs": {CLASSES[i]: float(p[i]) for i in range(5)}}
        warnings = []
        if source == "image": warnings.append("Single image: the frame is repeated six times, so the recurrent models cannot use any history. A short clip works better.")
        if any(flags): warnings.append(f"{sum(flags)} frame(s) looked unusable (too dark, flat or blurred) and were flagged for the models.")
        if float(np.ptp(ratios)) > 0.35: warnings.append("The six frames disagree strongly; check that they show the same lot.")
        return {"source": source, "condition": condition, "timestamp": when.isoformat(timespec="minutes"),
                "frames": [{"thumb": thumb_b64(f), "ratio": float(r), "unusable": bool(u)} for f, r, u in zip(bgr_frames, ratios, flags)],
                "estimated_ratio": float(ratios[-1]), "results": results, "advice": {k: ADVICE[v["class_index"]] for k, v in results.items()},
                "warnings": warnings}
