"""ParkOcc-AI prototype: frames -> per-frame occupancy features -> RNN/LSTM/GRU -> 5 occupancy classes."""
CLASSES = ["Empty", "Low", "Medium", "High", "Full"]
WINDOW = 6          # frames per clip
N_FEATURES = 9      # per-timestep feature vector (see simulate.py)
