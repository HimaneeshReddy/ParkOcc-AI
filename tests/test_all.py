"""Run:  python -m unittest discover -s tests -v"""
import io, os, sys, unittest, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from parkocc import simulate, render, frames, nn, CLASSES, WINDOW


class TestSimulation(unittest.TestCase):
    def test_window_shape_and_lot_disjoint_split(self):
        d = simulate.build_dataset(n_lots=10, days=8, split=(6, 2, 2), stride=1)
        for k in d: self.assertEqual(d[k][0].shape[1:], (WINDOW, 9)); self.assertEqual(len(d[k][0]), len(d[k][1]))
        self.assertEqual(len(d["train"][0]), 6 * (8 * 96 - WINDOW + 1))      # no window crosses a lot boundary
    def test_classes_from_clean_ratio_boundaries(self):
        self.assertEqual(simulate.ratio_to_class([0.0, 0.19, 0.2, 0.55, 0.99, 1.0]).tolist(), [0, 0, 1, 2, 4, 4])
    def test_time_encoding_is_unit_circle(self):
        f = simulate.time_features(np.arange(0, 24, 0.25), np.arange(96) % 7)
        np.testing.assert_allclose(f[:, 0] ** 2 + f[:, 1] ** 2, 1, atol=1e-5); np.testing.assert_allclose(f[:, 2] ** 2 + f[:, 3] ** 2, 1, atol=1e-5)
    def test_missing_frames_are_flagged_and_inputs_finite(self):
        f, y = simulate.make_lot(np.random.default_rng(0), days=20)
        self.assertTrue(np.isfinite(f).all()); self.assertGreater(f[:, 8].sum(), 0); self.assertTrue(((f[:, 0] >= 0) & (f[:, 0] <= 1)).all())


class TestNN(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(1); X = rng.normal(size=(4, 6, 9)); y = rng.integers(0, 5, 4)
        for kind in ["rnn", "lstm", "gru", "baseline"]:
            m = nn.Classifier(kind, H=5, dropout=0.0, seed=2)
            for L in m.layers():
                for k in L.p: L.p[k] = L.p[k].astype(np.float64)
            l, d = nn.softmax_ce(m.forward(X), y); grads = m.backward(d)
            for L, g in zip(m.layers(), grads):
                for k, v in L.p.items():
                    i = np.unravel_index(rng.integers(v.size), v.shape); o = v[i]; e = 1e-5
                    v[i] = o + e; a = nn.softmax_ce(m.forward(X), y)[0]; v[i] = o - e; b = nn.softmax_ce(m.forward(X), y)[0]; v[i] = o
                    self.assertAlmostEqual((a - b) / (2 * e), g[k][i], places=5, msg=f"{kind}.{k}")
    def test_param_counts_match_report_lstm(self):
        self.assertEqual(nn.Classifier("lstm", H=64).n_params(), 23429)
        self.assertEqual(nn.Classifier("rnn", H=32).n_params(), 3781)
    def test_probabilities_sum_to_one(self):
        p = nn.Classifier("gru", H=8).predict_proba(np.random.rand(7, 6, 9).astype(np.float32)); np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)


class TestFrames(unittest.TestCase):
    def test_unsupported_and_corrupt_media(self):
        with self.assertRaises(frames.MediaError): frames.load_frames(b"abc", "notes.txt")
        with self.assertRaises(frames.MediaError): frames.load_frames(b"not an image", "x.jpg")
    def test_image_repeated_to_window(self):
        ok, buf = cv2.imencode(".png", np.zeros((40, 60, 3), np.uint8)); fr, src = frames.load_frames(buf.tobytes(), "a.png")
        self.assertEqual((len(fr), src), (WINDOW, "image"))
    def test_quality_flag(self):
        self.assertTrue(frames.unusable(np.zeros((100, 100, 3), np.uint8)))
        self.assertFalse(frames.unusable(render.render_lot(np.arange(36) % 2 == 0, np.random.default_rng(0), 0, False)))


ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
@unittest.skipUnless(os.path.exists(os.path.join(ART, "metrics.json")), "run python -m parkocc.train first")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from parkocc.app import create_app; cls.c = create_app().test_client()
    def test_health_and_models(self):
        from parkocc.infer import MODEL_NAMES
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")
        m = self.c.get("/api/models").get_json(); self.assertEqual(set(m["names"]), set(MODEL_NAMES)); self.assertIn("test_lots", m)
        self.assertEqual(self.c.get("/").status_code, 200)
    def test_ordinal_safety_on_test_lots(self):
        m = self.c.get("/api/models").get_json()["test_lots"]
        for k, v in m.items(): self.assertEqual(v["empty_full_confusions"], 0, k)
    def test_sample_then_predict(self):
        from parkocc.infer import MODEL_NAMES
        s = self.c.post("/api/sample", json={"level": "Full", "condition": "clear", "noise": False}).get_json()
        self.assertEqual(len(s["frames"]), WINDOW); self.assertEqual(s["truth"]["class"], "Full")
        r = self.c.post("/api/predict", data={"clip_id": s["clip_id"], "model": "gru", "condition": "clear"}).get_json()
        self.assertEqual(set(r["results"]), set(MODEL_NAMES)); self.assertEqual(len(r["frames"]), WINDOW)
        for v in r["results"].values(): self.assertAlmostEqual(sum(v["probs"].values()), 1, places=4)
        self.assertGreater(r["estimated_ratio"], 0.6)
    def test_empty_vs_full_are_separated(self):
        out = {}
        for lvl in ["Empty", "Full"]:
            s = self.c.post("/api/sample", json={"level": lvl, "noise": False}).get_json()
            out[lvl] = self.c.post("/api/predict", data={"clip_id": s["clip_id"], "condition": "clear"}).get_json()
        self.assertLess(out["Empty"]["results"]["lstm"]["class_index"], out["Full"]["results"]["lstm"]["class_index"])
    def test_upload_image(self):
        img = render.render_lot(np.arange(36) % 2 == 0, np.random.default_rng(3), 0, False); ok, buf = cv2.imencode(".jpg", img)
        r = self.c.post("/api/predict", data={"file": (io.BytesIO(buf.tobytes()), "lot.jpg"), "model": "lstm"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200); j = r.get_json(); self.assertEqual(j["source"], "image"); self.assertTrue(j["warnings"])
    def test_upload_video(self):
        import tempfile
        rng = np.random.default_rng(4); fr, _ = render.render_clip(0.5, rng, 0, False, n_frames=30)
        p = os.path.join(tempfile.mkdtemp(), "c.avi"); vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 5, (render.W, render.H))
        for f in fr: vw.write(cv2.resize(f, (render.W, render.H)))
        vw.release(); r = self.c.post("/api/predict", data={"file": (open(p, "rb"), "c.avi"), "model": "gru"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["source"], "video")
    def test_bad_requests(self):
        post = lambda **kw: self.c.post("/api/predict", data=kw, content_type="multipart/form-data")
        self.assertEqual(post().status_code, 400)
        self.assertEqual(post(file=(io.BytesIO(b"hello"), "x.txt")).status_code, 400)
        self.assertEqual(post(file=(io.BytesIO(b"hello"), "x.jpg")).status_code, 400)
        self.assertEqual(post(clip_id="nope").status_code, 404)
        self.assertEqual(post(clip_id="nope", model="svm").status_code, 400)
        self.assertEqual(post(clip_id="nope", condition="fog").status_code, 400)
        self.assertEqual(post(clip_id="nope", timestamp="yesterday").status_code, 400)
        self.assertEqual(self.c.post("/api/sample", json={"level": "huge"}).status_code, 400)

if __name__ == "__main__":
    unittest.main()
