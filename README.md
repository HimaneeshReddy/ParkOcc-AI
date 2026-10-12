# ParkOcc-AI prototype

A small, working end-to-end version of the ParkOcc-AI plan: **image or short clip in, one of five occupancy levels
(Empty / Low / Medium / High / Full) plus a confidence score out**, with a model toggle and a web dashboard.

```
upload / sample clip -> 6 frames -> per-frame occupancy features -> RNN | LSTM | GRU (or single-frame baseline)
                     -> Dense + softmax (5 classes) -> REST API -> dashboard (badge, confidence, comparison)
```

## Run it

```bash
pip install -r requirements.txt
python run.py                      # http://127.0.0.1:5000   (trained models are included in artifacts/)
python -m unittest discover -s tests -v     # 17 tests
python -m parkocc.train            # optional: retrain everything (~4 min, one CPU core)
```

In the dashboard: **Upload** a photo or video, or open **Generate sample** to render a synthetic lot clip, pick a model,
press **Analyze occupancy**. The comparison table runs all four models on the same clip.

## What matches your plan, and what does not

| Plan | Prototype |
|---|---|
| Image / short video input | Yes: jpg, png, bmp, webp, mp4, avi, mov, webm, mkv. A video is sampled into 6 frames; a single image is repeated (and the UI warns about it). |
| Five classes, confidence score | Yes. Class boundaries at 20% occupancy, softmax confidence. |
| CNN, RNN, LSTM, GRU toggle | RNN / LSTM / GRU: yes, implemented in numpy with full backpropagation through time (gradients are checked against finite differences in the tests). **The "CNN" stage is not a CNN**: this build environment had no deep-learning framework, so per-frame features come from a small learned regressor on image statistics (`parkocc/frames.py`). The baseline is a single-frame network with no temporal memory. |
| PIL-park dataset | **Not used** (needs an IEEE Dataport login). Training uses the synthetic noisy-lot generator (28 / 6 / 6 lot-disjoint split) and a synthetic top-down lot renderer. |
| REST API + dashboard | Yes (`/api/health`, `/api/models`, `/api/sample`, `/api/predict`). |
| Noise handling | Four noise channels in the data (condition-scaled jitter, 2% spikes, 1% unusable frames, per-lot bias); frames that are too dark, flat or blurred are flagged to the models. |

## Measured results (this prototype only, not the report's numbers)

Held-out test lots, plus end-to-end on 240 freshly rendered, JPEG-compressed clips through the extractor:

| Model | Params | Test-lot accuracy | Macro-F1 | Full recall | End-to-end accuracy |
|---|---|---|---|---|---|
| Single-frame baseline | 965 | 83.9% | 0.842 | 0.87 | 64% |
| RNN | 3,781 | 89.4% | 0.898 | 0.93 | 75% |
| LSTM | 23,429 | 89.2% | 0.894 | 0.94 | 73% |
| GRU | 18,693 | 89.2% | 0.895 | 0.90 | 72% |

The frame extractor is off by about 1.9 of 36 slots on average (worst in rain). Errors stay within one class
level in nearly every case and no Empty/Full swap occurred in testing, but exact-class accuracy on images is only about
60-75%. Numbers are higher than in the report because this simulator is cleaner than the report's dataset.

## Limits you should know about

* It is trained on rendered top-down lots. A real CCTV photo will be much harder; expect poor accuracy until the extractor is a CNN trained on real frames.
* Frames of a clip are treated as consecutive 15-minute readings (as in training), not as 15-second frames.
* Capture time, filming condition and lot capacity are inputs to the models; capacity is fixed at a mid value.
* Single seed, no confidence intervals.

## Upgrading toward the full plan

1. Replace `FrameExtractor.ratio()` with a CNN trained on real or PIL-park-derived lot images (same interface, returns a 0-1 ratio, or swap the 9-d feature for a 128-d embedding and widen `N_FEATURES`).
2. Port `parkocc/nn.py` to PyTorch or Keras (the cell equations are the standard ones).
3. Add K-means, GAN and XAI as separate modules once you have results to report.

## Layout

```
parkocc/simulate.py  noisy lot generator + windows      parkocc/render.py  synthetic lot images
parkocc/frames.py    feature extractor, video loading   parkocc/nn.py      RNN/LSTM/GRU + Adam in numpy
parkocc/train.py     training + evaluation              parkocc/infer.py   predictor
parkocc/app.py       Flask API                          static/index.html  dashboard
artifacts/           trained weights + metrics.json     tests/test_all.py  unit + API tests
```
