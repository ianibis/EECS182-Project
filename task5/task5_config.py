"""All settings for Task 5 live here, so an experiment can be read in one place."""

import os
import sys
from dataclasses import dataclass

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TASK_DIR)
for folder in (REPO, os.path.join(REPO, 'task3')):  # reuse the shared loader and Task 3 helpers
    if folder not in sys.path:
        sys.path.insert(0, folder)

CACHE_DIR = os.path.join(REPO, 'data', 'task5_cache')       # trial arrays (git-ignored)
RESULTS_DIR = os.path.join(REPO, 'task5_results')           # CSV results and figures
RUNS_DIR = os.path.join(REPO, 'data', 'task5_runs')         # checkpoints (git-ignored)

# ----------------------------------------------------------------- trial layout
BIN_SAMPLES = 50        # one 50 ms bin at 1 kHz
N_BINS = 40             # scored bins: 0.5 s before go to 1.5 s after
HISTORY_BINS = 9        # extra bins before the first scored bin, so every token sees 500 ms
N_SAMPLES = (N_BINS + HISTORY_BINS) * BIN_SAMPLES  # 2450 samples per trial
N_CHANNELS = 35         # the fixed channel mask from the data README
PRE_S, POST_S = 0.5, 1.5
KIN_COLS = ('Index_vx_cm', 'Index_vy_cm')
CLIP_Z = 10.0           # same cap as Task 3 (channel 64 artifact)

# ----------------------------------------------------------------- evaluation
K_VALUES = (0, 2, 5, 10, 20)
POOL_SIZE = 20          # blocks 1-2 of a held-out session

# Templates the data README calls poor candidates for semi-synthetic artifacts.
POOR_TEMPLATES = ('Butters-250606-104043', 'Butters-250609-111203', 'Butters-250529-102830')
TEMPLATE_BINS = 128     # every template is resampled to this many phase bins


@dataclass
class ShiftConfig:
    """The simulated stimulation shift. Each realism option can be turned off (Task 5.1.2)."""
    amplitude_min: float = 1.0          # artifact size relative to the real template
    amplitude_max: float = 30.0         # sampled log-uniformly; covers the real (1x) and 10x cases
    channel_gain_sigma: float = 0.30    # log-normal spread of per-channel artifact gain
    period_jitter: float = 0.01         # relative error of the AM frequency
    use_slow_modulation: bool = True    # slow drift of the artifact amplitude
    slow_modulation_depth: float = 0.20
    use_trial_gain: bool = True         # trial-to-trial amplitude change
    trial_gain_sigma: float = 0.15
    use_nonlinearity: bool = True       # mild amplifier nonlinearity
    nonlinearity_max: float = 0.10
    use_shared_component: bool = True   # a part of the artifact common to all channels
    shared_max: float = 0.50
    use_gain_drift: bool = True         # channel gain change across the session
    gain_sigma: float = 0.10
    drift_max: float = 0.05
    use_dropout: bool = True            # channels lost in this session
    dropout_max: float = 0.10
    use_residual: bool = True           # non-periodic part of the real artifact (added after check 5.1.3)
    residual_min: float = 0.10          # its size relative to the periodic artifact; real sessions have a
    residual_max: float = 0.55          # median of 0.16 (3 Hz) and 0.24 (20 Hz), and up to about 0.56
    residual_shared: float = 0.90       # share of the residual's power common to all channels


@dataclass
class ModelConfig:
    """EEGNet-style encoder plus a causal transformer (proposal: 4 layers, width 128, about 1M parameters)."""
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_size: int = 512
    dropout: float = 0.10
    encoder_dropout: float = 0.25
    temporal_filters: int = 16
    depth_multiplier: int = 2
    separable_filters: int = 64
    temporal_kernel: int = 125          # 125 ms
    separable_kernel: int = 64          # 320 ms at the 200 Hz rate after the first pooling
    blind: bool = False                 # True hides all brain signal: the labels-only transformer


@dataclass
class TrainConfig:
    seed: int = 1
    iterations: int = 6000
    batch_size: int = 8
    learning_rate: float = 3e-4
    warmup_iterations: int = 300
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 500
    max_context: int = 20
