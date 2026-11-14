"""Paths and constants for the section 5 in-context decoder."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ICL/
RESULTS = os.path.join(ROOT, 'results')
DATA = os.environ.get('EECS182_DATA', os.path.expanduser('~/data/EECS182'))
CACHE = os.environ.get('EECS182_ICL_CACHE', os.path.join(DATA, 'icl_cache'))
KIN_DIR = os.path.join(DATA, 'kinematics_v3', 'bins50ms')

BIN_S = 0.05
FS = 1000.0                  # nominal; the recorded rate is ~999.9998 Hz, i.e. 50 samples per bin
SAMPLES_PER_BIN = 50
PRE_S, POST_S = 2.5, 2.0     # cached window around the go cue
N_CACHE_BINS = 90            # round((PRE_S + POST_S) / BIN_S)
N_SAMPLES = N_CACHE_BINS * SAMPLES_PER_BIN
N_TARGET_BINS = 40           # decoded window
TARGET_START_BIN = 40        # standard decoded window: cache bins 40..79 = [t_go - 0.5, t_go + 1.5)
MAX_JITTER_BINS = 10         # training: decoded window shifted by up to +-0.5 s (cache covers it)
TOKEN_WIN = 500              # samples of LFP history behind each token (500 ms)
# Bins of LFP before the first decoded bin in the model input: token 0 covers its own bin + 9
# earlier bins (500 ms); 2 more feed the causal 64-sample temporal conv of EEGNet's first block.
MODEL_HISTORY_BINS = TOKEN_WIN // SAMPLES_PER_BIN - 1 + 2   # 11
MODEL_BINS = MODEL_HISTORY_BINS + N_TARGET_BINS              # 51 bins = model input window
MODEL_SAMPLES = MODEL_BINS * SAMPLES_PER_BIN                 # 2550 samples
LMP_WIN_S = 0.256            # local motor potential window (loader and features scripts)
N_CHANNELS = 35
AM_FREQS_HZ = {3: 2.9971, 20: 19.8911}   # PARRM-fitted AM frequencies (README)
KIN_COLS = ('Index_vx_cm', 'Index_vy_cm')
# channel_keep_fixed: the same 35 channels in every session (README)
FIXED_CHANNELS = (21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45,
                  47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64)
TEMPLATE_LIBRARY = os.path.join(DATA, 'parrm_outputs', 'template_library.h5')
SPLITS = os.path.join(DATA, 'splits.json')


def paths():
    import EECS182_NHP_data_loader as L
    return L.Paths(root=DATA, kin_dir=KIN_DIR)


def cache_dir(source='export'):
    return os.path.join(CACHE, source)


def load_stats(source='export'):
    with open(os.path.join(cache_dir(source), 'stats.json')) as file:
        return json.load(file)
