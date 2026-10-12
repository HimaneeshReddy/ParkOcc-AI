"""Flask REST API + dashboard.   python run.py   ->  http://127.0.0.1:5000"""
import os, uuid, datetime as dt
from collections import OrderedDict
import numpy as np, cv2
from flask import Flask, jsonify, request, send_from_directory
from . import CLASSES, render, simulate, frames
from .infer import Predictor, thumb_b64, MODEL_NAMES, CONDITIONS

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
LEVELS = {n.lower(): i for i, n in enumerate(CLASSES)}


def create_app(art_dir=None):
    app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    pred = Predictor(art_dir) if art_dir else Predictor()
    clips = OrderedDict(); rng = np.random.default_rng()

    def err(msg, code=400): return jsonify({"error": msg}), code

    @app.errorhandler(413)
    def too_big(_): return err("File too large (limit 50 MB).", 413)

    @app.get("/")
    def index(): return send_from_directory(STATIC, "index.html")

    @app.get("/api/health")
    def health(): return jsonify({"status": "ok", "models": list(MODEL_NAMES), "classes": CLASSES})

    @app.get("/api/models")
    def models(): return jsonify({"names": MODEL_NAMES, **pred.metrics})

    @app.post("/api/sample")
    def sample():
        body = request.get_json(silent=True) or {}
        level = str(body.get("level", "random")).lower(); cond = str(body.get("condition", "clear")).lower()
        if cond not in CONDITIONS: return err("condition must be clear, overcast or rain")
        if level != "random" and level not in LEVELS: return err("level must be random or one of " + ", ".join(CLASSES))
        k_range = None; n = render.N_SLOTS
        if level == "random": target = float(rng.uniform(0, 1))
        else:   # keep the clip inside the chosen class band (class boundaries are at 20% steps of the occupancy ratio)
            i = LEVELS[level]; lo = int(np.ceil(0.2 * i * n - 1e-9)); hi = n if i == 4 else int(np.ceil(0.2 * (i + 1) * n - 1e-9)) - 1
            k_range = (lo, hi); target = float(rng.integers(lo, hi + 1)) / n
        fr, tr = render.render_clip(target, rng, CONDITIONS[cond], bool(body.get("noise", True)), k_range=k_range)
        cid = uuid.uuid4().hex[:12]; clips[cid] = fr
        while len(clips) > 16: clips.popitem(last=False)
        return jsonify({"clip_id": cid, "frames": [thumb_b64(f, width=320, q=80) for f in fr],
                        "truth": {"ratio": float(tr[-1]), "class": CLASSES[int(simulate.ratio_to_class(tr[-1]))]}})

    @app.post("/api/predict")
    def predict():
        cond = request.form.get("condition", "clear").lower()
        if cond not in CONDITIONS: return err("condition must be clear, overcast or rain")
        sel = request.form.get("model", "gru").lower()
        if sel not in MODEL_NAMES: return err("model must be one of " + ", ".join(MODEL_NAMES))
        when = dt.datetime.now()
        if request.form.get("timestamp"):
            try: when = dt.datetime.fromisoformat(request.form["timestamp"])
            except ValueError: return err("timestamp must be ISO format, e.g. 2026-10-12T08:30")
        cid = request.form.get("clip_id"); f = request.files.get("file")
        if cid:
            if cid not in clips: return err("Unknown or expired clip_id; generate a new sample.", 404)
            fr, source = clips[cid], "video"
        elif f and f.filename:
            try: fr, source = frames.load_frames(f.read(), f.filename)
            except frames.MediaError as e: return err(str(e))
        else: return err("Send a file (image or video) or a clip_id.")
        out = pred.analyse(fr, source, cond, when); out["selected"] = sel
        return jsonify(out)

    return app
