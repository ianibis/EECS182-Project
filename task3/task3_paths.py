"""Resolve checkout outputs separately from existing research inputs."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from EECS182_NHP_data_loader import Paths

DATA_ROOT = Path(os.environ.get('TASK3_DATA_ROOT', REPO)).expanduser().resolve()
GRU_RESULTS = REPO / 'task3_results' / 'gru'
MLP_RESULTS = REPO / 'task4_results' / 'mlp'


def feature_root(root):
    if (root / 'export_outputs').is_dir():
        return root
    if root == REPO or (root / 'data').is_dir():
        return root / 'data'
    return root


def data_paths(root):
    features = feature_root(root)
    if (root / 'export_outputs').is_dir():
        exports = root / 'export_outputs'
        parrm = root / 'parrm_outputs' / 'parrm_h5'
        kinematics = root / 'kinematics_v3' / 'bins50ms'
        export_dir, trial_dir = exports / 'h5_export_outputs', exports / 'trials'
    else:
        exports = features / 'exports'
        parrm = features / 'parrm' / 'parrm_h5_LFPs'
        kinematics = features / 'kinematics_bins50ms' / 'bins50ms'
        export_dir, trial_dir = exports / 'export_h5_LFPs', exports / 'export_trials'
    return Paths(root=str(features), export_dir=str(export_dir), trial_dir=str(trial_dir),
                 parrm_dir=str(parrm), kin_dir=str(kinematics))


def metadata_path(root, name):
    external = root / name
    return external if external.is_file() else REPO / name


def stored_path(root, relative):
    path = root / relative
    return path if path.is_file() else feature_root(root) / relative


def gru_artifact(group, name, root=DATA_ROOT):
    if group == 'checkpoints':
        local = GRU_RESULTS / 'training' / 'checkpoints' / name
    else:
        local = GRU_RESULTS / group / name
    if local.is_file():
        return local
    archived = root / 'results' / 'task3'
    if group.endswith('checkpoints'):
        archived /= 'checkpoints'
    return archived / name


def mlp_checkpoint(run):
    local = MLP_RESULTS / run / 'checkpoint.pt'
    if local.is_file():
        return local
    return DATA_ROOT / 'Task4' / 'results' / 'mlp' / run / 'checkpoint.pt'
