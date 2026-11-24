"""Shared helpers for the Task 4 notebooks.

Data, splits and loaders, metrics, the training loop, decoder models, the
calibration-draw sampler and the EEGNet baseline used by task 4.1-4.3.

Sections
  1  Imports and paths
  2  Data, splits and loaders
  3  Metrics
  4  Training utilities
  5  Evaluation utilities
  6  Decoder re-exports and loaders
  7  Calibration draws (task 4.1)
  8  Bootstrap statistics
  9  EEGNet baseline
"""

# ---------------------------------------------------------------------------
# 1. Imports and paths
# ---------------------------------------------------------------------------

import copy
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score

TASK4_DIR = os.path.dirname(os.path.abspath(__file__))  # .../task4
PROJECT_ROOT = os.path.dirname(TASK4_DIR)  # repository root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")  # earlier baseline artifacts
TASK4_RESULTS = os.path.join(TASK4_DIR, "results")
CHECKPOINT_DIR = os.path.join(TASK4_DIR, "checkpoints")
TRAINING_LOG_DIR = os.path.join(TASK4_DIR, "training_logs")
DRAWS_PATH = os.path.join(TASK4_RESULTS, "draws.json")
BIN_S = 0.05  # 50 ms bins

os.makedirs(TASK4_RESULTS, exist_ok=True)  # notebooks write their CSVs here

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import EECS182_NHP_data_loader as lab_loader
import EECS182_NHP_features as lab_features

LOADER_PATHS = lab_loader.Paths(
    root=DATA_DIR,
    export_dir=os.path.join(DATA_DIR, "export_outputs", "h5_export_outputs"),
    trial_dir=os.path.join(DATA_DIR, "export_outputs", "trials"),
    kin_dir=os.path.join(DATA_DIR, "kinematics_v3", "bins50ms"),
)

SEMISYNTHETIC_EXPORT_DIR = os.path.join(
    DATA_DIR, "semi_synthetic", "parrm_amp10", "h5_export_outputs"
)
AMP10_VARIANT = "amp10"  # feature_variant label for rows scored on the 10x recording

PARRM_DIR = os.path.join(DATA_DIR, "parrm_outputs", "parrm_h5")

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "EECS182_data_manifest.csv")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "splits.json")
MANIFEST_TABLE = lab_loader.load_manifest(LOADER_PATHS, MANIFEST_PATH)
SESSION_TABLE = MANIFEST_TABLE.set_index(
    "session"
)  # metadata, sessions stored row-wise
SPLITS = lab_loader.load_splits(
    LOADER_PATHS, SPLITS_PATH
)  # fixed train/val/test and calibration splits

LAG_COUNT = 9

# ---------------------------------------------------------------------------
# 2. Data, splits and loaders
# ---------------------------------------------------------------------------


def load_json(path):
    """Read one JSON file into Python objects."""
    with open(path) as file:
        contents = json.load(file)
    return contents


def trials_in_split_part(session, part):
    session_table = lab_loader.load_session(
        session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE
    )
    part_trials = lab_loader.split_trials(SPLITS, session_table, part)
    return part_trials


TARGET_VELOCITY_COLS = ("Index_vx_cm", "Index_vy_cm")  # target velocities
NORMALIZATION_PAD_S = 1.0  # extra seconds on each side of a trial
MAD_TO_STD = 1.4826  # MAD to standard deviation


def select_trial_rows(session_table, trials):
    """Trial rows for `trials`, or every included trial when `trials` is None."""
    if trials is None:
        included_rows = session_table.included
        return included_rows
    trial_numbers = {int(trial) for trial in trials}
    trial_mask = session_table.trials.trial.isin(trial_numbers)
    selected_rows = session_table.trials[trial_mask]
    ordered_rows = selected_rows.sort_values("t_go")
    return ordered_rows


def keep_bins_inside_trials(trial_rows, bin_ids):
    """Mask the 50 ms bins that sit wholly inside the trial rows' time span."""
    first_time = float(trial_rows.t_go.min() - NORMALIZATION_PAD_S)
    last_time = float(trial_rows.t_reward.max() + NORMALIZATION_PAD_S)
    bin_start = bin_ids * BIN_S
    bin_end = (bin_ids + 1) * BIN_S
    inside = (bin_start >= first_time) & (bin_end <= last_time)
    return inside


def normalize_features(session, variant, normalization_trials):
    """z-score every feature by the median and MAD over the normalization trials."""
    feature_buffer = lab_features.load_features(
        session, LOADER_PATHS, variant=variant, normalize=False
    )
    session_table = lab_loader.load_session(
        session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE
    )
    normalization_rows = select_trial_rows(session_table, normalization_trials)
    normalization_mask = keep_bins_inside_trials(
        normalization_rows, feature_buffer["bin_id"]
    )
    reference_features = feature_buffer["X"][normalization_mask]
    feature_median = np.nanmedian(reference_features, axis=0)
    feature_scale = MAD_TO_STD * np.nanmedian(
        np.abs(reference_features - feature_median), axis=0
    )
    feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
    normalized_values = (feature_buffer["X"] - feature_median) / feature_scale
    normalized_features = feature_buffer | {"X": normalized_values.astype(np.float32)}
    return normalized_features


def load_trial_dataset(session, trials, variant, normalization_trials, lag_count):
    """Per-trial bin tensors, one row per 50 ms bin inside each trial.

    features   (n_trials, n_bins_per_trial, n_channels * n_feature_types * (lag_count + 1))
    targets    (n_trials, n_bins_per_trial, 2)  Index fingertip vx, vy per bin
    valid_bins (n_trials, n_bins_per_trial)     bins with both features and kinematics
    bin_ids    (n_trials, n_bins_per_trial)     index on the global 50 ms grid
    trials     (n_trials,)
    """
    normalized_features = normalize_features(session, variant, normalization_trials)
    session_table = lab_loader.load_session(
        session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE
    )
    trial_rows = select_trial_rows(session_table, trials)
    kinematics = lab_loader.load_kinematics(session, LOADER_PATHS)
    (
        trial_bin_features,
        trial_bin_velocity,
        trial_bin_valid,
        trial_bin_ids,
    ) = lab_features.trial_arrays(
        normalized_features,
        trial_rows,
        kin=kinematics,
        align="t_go",
        pre=0.5,
        post=1.5,
        n_lags=lag_count,
        kin_cols=TARGET_VELOCITY_COLS,
    )
    trial_numbers = trial_rows.trial.to_numpy()
    trial_dataset = dict(
        features=trial_bin_features,
        targets=trial_bin_velocity,
        valid_bins=trial_bin_valid,
        bin_ids=trial_bin_ids,
        trials=trial_numbers,
    )
    return trial_dataset


def load_split_dataset(session, part, variant, lag_count):
    trials = trials_in_split_part(session, part)
    if part in ("train", "val"):
        normalization_trials = trials
    else:
        normalization_trials = SPLITS["calibration"][session]["pool"]
    split_dataset = load_trial_dataset(
        session, trials, variant, normalization_trials, lag_count
    )
    return split_dataset


def same_day_control_session(stimulation_on_session):
    """The same-day no-stimulation recording paired with a stimulated session.

    `splits.json` stores this as `off_reference`; the name here says what the session is
    used for. Nine stimulated sessions have no same-day control and raise.
    """
    control_session = SPLITS["off_reference"][stimulation_on_session]
    if control_session is None:
        raise ValueError(
            f"{stimulation_on_session} has no same-day no-stimulation control session"
        )
    return control_session


def load_same_day_control_dataset(stimulation_on_session, variant, lag_count):
    """Trials from the same-day no-stimulation recording, as a trial dataset."""
    control_session = same_day_control_session(stimulation_on_session)
    control_session_table = lab_loader.load_session(
        control_session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE
    )
    control_trials = control_session_table.included.trial.astype(int).tolist()
    control_dataset = load_trial_dataset(
        control_session, control_trials, variant, control_trials, lag_count
    )
    return control_dataset


def valid_bin_rows(dataset):
    """Flatten a trial dataset to one row per valid 50 ms bin.

    Returns bin features (n_valid_bins, n_features) and bin velocity (n_valid_bins, 2).
    """
    valid_bin_mask = dataset["valid_bins"]
    bin_features = dataset["features"][valid_bin_mask]
    bin_velocity = dataset["targets"][valid_bin_mask]
    return bin_features, bin_velocity


def semisynthetic_paths(export_dir=SEMISYNTHETIC_EXPORT_DIR):
    """Loader paths that read LFP from the 10x semi-synthetic export folder.

    Trials and kinematics are unchanged by the amplification, so only export_dir moves.
    """
    paths = lab_loader.Paths(
        root=DATA_DIR,
        export_dir=export_dir,
        trial_dir=os.path.join(DATA_DIR, "export_outputs", "trials"),
        kin_dir=os.path.join(DATA_DIR, "kinematics_v3", "bins50ms"),
    )
    return paths


def parrm_paths(parrm_dir=PARRM_DIR):
    """Loader paths that can read the PARRM-cleaned LFP via source='parrm'."""
    paths = lab_loader.Paths(
        root=DATA_DIR,
        export_dir=os.path.join(DATA_DIR, "export_outputs", "h5_export_outputs"),
        trial_dir=os.path.join(DATA_DIR, "export_outputs", "trials"),
        parrm_dir=parrm_dir,
        kin_dir=os.path.join(DATA_DIR, "kinematics_v3", "bins50ms"),
    )
    return paths


def load_lfp_dataset(session, trials, paths=None, source="export"):
    """Per-trial LFP tensors: (n_trials, n_channels, n_bins * samples_per_bin)."""
    session_table = lab_loader.load_session(
        session,
        paths or LOADER_PATHS,
        with_lfp=True,
        manifest=MANIFEST_TABLE,
        source=source,
    )
    lab_loader.normalize(session_table)
    included_trials = session_table.trials[session_table.trials.include == 1]
    session_table.trials = included_trials
    kinematics = lab_loader.load_kinematics(session, LOADER_PATHS)
    examples = lab_loader.trial_examples(
        session_table,
        kinematics,
        align="t_go",
        pre=0.5,
        post=1.5,
        trials=trials,
        kin_cols=TARGET_VELOCITY_COLS,
    )
    flattened_lfp = [
        example["lfp"].reshape(example["lfp"].shape[0], -1) for example in examples
    ]
    trial_channels_by_samples = np.stack(flattened_lfp).astype(np.float32)
    trial_bin_velocity = np.stack([example["kin"] for example in examples]).astype(
        np.float32
    )
    trial_bin_valid = np.stack([example["kin_valid"] for example in examples]).astype(
        bool
    )
    lfp_dataset = dict(
        features=trial_channels_by_samples,
        targets=trial_bin_velocity,
        valid_bins=trial_bin_valid,
    )
    return lfp_dataset


def load_lfp_split(session, part, paths=None, source="export"):
    trials = trials_in_split_part(session, part)
    lfp_dataset = load_lfp_dataset(session, trials, paths, source)
    return lfp_dataset


STIMULATION_ON = 1  # manifest 'stim' flag for recordings with stimulation on
STIMULATION_OFF = 0  # recordings with no stimulation at all


def sessions_in_split(part, stimulation_flag):
    """Sessions in `part` whose manifest `stim` flag equals `stimulation_flag`."""
    split_sessions = SPLITS["sessions"][part]
    selected_sessions = []
    for session in split_sessions:
        session_stimulation = int(SESSION_TABLE.loc[session, "stim"])
        if session_stimulation == stimulation_flag:
            selected_sessions.append(session)
    return selected_sessions


STIMULATION_OFF_TRAINING_SESSIONS = sessions_in_split("train", STIMULATION_OFF)
CALIBRATION_SIZES = (2, 5, 10, 20)  # calibration trial counts

# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


def r2_per_target(bin_velocity, predicted_velocity, valid_bin_mask=None):
    """R² for each velocity component over the selected 50 ms bins."""
    bin_velocity = np.asarray(bin_velocity, np.float64)
    predicted_velocity = np.asarray(predicted_velocity, np.float64)
    if valid_bin_mask is not None:
        valid_bin_mask = np.asarray(valid_bin_mask, bool)
        bin_velocity = bin_velocity[valid_bin_mask]
        predicted_velocity = predicted_velocity[valid_bin_mask]
    if len(bin_velocity) == 0:
        return np.full(bin_velocity.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = r2_score(
            bin_velocity,
            predicted_velocity,
            multioutput="raw_values",
            force_finite=False,
        )
    return np.where(np.isfinite(scores), scores, np.nan)


def mean_r2(bin_velocity, predicted_velocity, valid_bin_mask=None):
    """Average R² over the velocity components."""
    per_target_r2 = r2_per_target(bin_velocity, predicted_velocity, valid_bin_mask)
    mean_score = float(np.nanmean(per_target_r2))
    return mean_score


# ---------------------------------------------------------------------------
# 4. Training utilities
# ---------------------------------------------------------------------------

ADAM_LR = 1e-3  # Adam learning rate for biases and normalisation parameters
MUON_LR = 0.02  # Muon learning rate for the weight matrices
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
LR_FLOOR = 0.01  # cosine decay ends at this fraction of the initial rate
TRIAL_BATCH_SIZE = 16


def save_checkpoint(model, name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{name}.pt"))


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_mse(predicted_velocity, bin_velocity, valid_bin_mask):
    """MSE over the bins whose kinematics exist; NaN bins never enter the loss."""
    if not valid_bin_mask.any():
        return predicted_velocity.sum() * 0.0
    return F.mse_loss(predicted_velocity[valid_bin_mask], bin_velocity[valid_bin_mask])


def to_tensors(numpy_arrays):
    return [
        torch.as_tensor(
            np.asarray(numpy_array),
            dtype=(
                torch.bool if np.asarray(numpy_array).dtype == bool else torch.float32
            ),
        )
        for numpy_array in numpy_arrays
    ]


def predict(model, bin_features):
    """Run a model over bin rows (n_bins, n_features) or trials (n_trials, ...)."""
    device = select_device()
    model = model.to(device).eval()
    inputs = torch.as_tensor(np.asarray(bin_features), dtype=torch.float32).to(device)
    with torch.inference_mode():
        return model(inputs).cpu().numpy()


def orthogonal_polar_factor(gradient, steps=MUON_NS_STEPS, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    orthogonal = gradient.float()
    orthogonal = orthogonal / (orthogonal.norm() + eps)
    if orthogonal.shape[0] > orthogonal.shape[1]:
        orthogonal = orthogonal.T
    for _ in range(steps):
        gram = orthogonal @ orthogonal.T
        orthogonal = a * orthogonal + (b * gram + c * (gram @ gram)) @ orthogonal
    return orthogonal.T if gradient.shape[0] > gradient.shape[1] else orthogonal


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=MUON_LR,
        momentum=MUON_MOMENTUM,
        nesterov=True,
        ns_steps=MUON_NS_STEPS,
    ):
        super().__init__(
            params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = (
                    parameter.grad.reshape(parameter.shape[0], -1)
                    if parameter.ndim > 2
                    else parameter.grad
                )
                buffer = self.state[parameter].setdefault(
                    "momentum_buffer", torch.zeros_like(gradient)
                )
                buffer.mul_(group["momentum"]).add_(gradient)
                update = (
                    gradient.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                orthogonal = orthogonal_polar_factor(update, group["ns_steps"])
                # sqrt(max(1, rows / cols)) matches the RMS of an Adam step at the same lr.
                step_size = (
                    group["lr"]
                    * max(1.0, orthogonal.shape[0] / orthogonal.shape[1]) ** 0.5
                )
                parameter.add_(
                    orthogonal.reshape(parameter.shape).to(parameter.dtype),
                    alpha=-step_size,
                )


def cosine_lr_multiplier(epoch, epochs):
    return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * epoch / epochs))


def set_learning_rates(optimizers, base_rates, factor):
    for optimizer, optimizer_rates in zip(optimizers, base_rates):
        for group, base_rate in zip(optimizer.param_groups, optimizer_rates):
            group["lr"] = base_rate * factor


def train_epoch(
    model, dataset_tensors, loss_fn, optimizers, sample_order, batch_size, device
):
    """One pass in which `sample_order` indexes the rows of every dataset tensor."""
    model.train()
    loss_sum = 0.0
    for start in range(0, len(sample_order), batch_size):
        batch_rows = sample_order[start : start + batch_size]
        bin_features_batch, *target_batch = [
            tensor[batch_rows].to(device) for tensor in dataset_tensors
        ]
        loss = loss_fn(model(bin_features_batch), *target_batch)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        loss_sum += float(loss.detach()) * len(batch_rows)
    return loss_sum / len(sample_order)


def predict_trial_bins(model, trial_dataset):
    """Predict (n_trials, n_bins, 2) from a per-trial dataset."""
    trial_features = trial_dataset["features"]
    predicted_velocity = predict(model, trial_features)
    return predicted_velocity


# ---------------------------------------------------------------------------
RAW_VARIANT = "export_none"
PARRM_VARIANT = "parrm_none"

# The `r2_off_*` result columns keep their short spelling because they are part of the
# committed CSV/parquet schema. There, `off` means the same-day no-stimulation control.


def concat_trial_datasets(per_session_datasets):
    """Concatenate per-session trial datasets along the trial axis."""
    feature_blocks = [dataset["features"] for dataset in per_session_datasets]
    velocity_blocks = [dataset["targets"] for dataset in per_session_datasets]
    valid_blocks = [dataset["valid_bins"] for dataset in per_session_datasets]
    combined_dataset = dict(
        features=np.concatenate(feature_blocks),
        targets=np.concatenate(velocity_blocks),
        valid_bins=np.concatenate(valid_blocks),
    )
    return combined_dataset


def stack_session_bin_rows(sessions, part, variant, lag_count):
    """Stack every valid bin row from several sessions.

    Returns bin features (n_bins_total, n_features) and velocity (n_bins_total, 2).
    """
    per_session_bin_rows = []
    for session in sessions:
        session_dataset = load_split_dataset(session, part, variant, lag_count)
        bin_rows = valid_bin_rows(session_dataset)
        per_session_bin_rows.append(bin_rows)
    bin_features_blocks, bin_velocity_blocks = zip(*per_session_bin_rows)
    bin_features = np.concatenate(bin_features_blocks)
    bin_velocity = np.concatenate(bin_velocity_blocks)
    return bin_features, bin_velocity


def load_calibration_bin_rows(
    session, trials, normalization_trials, variant, lag_count
):
    trial_dataset = load_trial_dataset(
        session, trials, variant, normalization_trials, lag_count
    )
    bin_rows = valid_bin_rows(trial_dataset)
    return bin_rows


def metrics_row(
    session,
    method,
    feature_label,
    bin_velocity,
    predicted_velocity,
    control_bin_velocity,
    predicted_control_velocity,
    k=0,
    draw=0,
    seed=0,
):
    r2 = r2_per_target(bin_velocity, predicted_velocity)
    control_r2 = r2_per_target(control_bin_velocity, predicted_control_velocity)
    return dict(
        method=method,
        feature_variant=feature_label,
        session=session,
        am_freq=float(SESSION_TABLE.loc[session, "am_freq_hz"]),
        k=k,
        draw=draw,
        seed=seed,
        r2_vx=float(r2[0]),
        r2_vy=float(r2[1]),
        r2_ave=float(np.nanmean(r2)),
        r2_off_vx=float(control_r2[0]),
        r2_off_vy=float(control_r2[1]),
        r2_off_ave=float(np.nanmean(control_r2)),
        n_bins=len(bin_velocity),
    )


def score_window_decoder(
    decoder,
    session,
    method,
    feature_label,
    query_bin_rows,
    control_bin_rows,
    k=0,
    draw=0,
):
    """Score a decoder that consumes one row per 50 ms bin."""
    query_bin_features, query_bin_velocity = query_bin_rows
    control_bin_features, control_bin_velocity = control_bin_rows
    predicted_query_velocity = decoder.predict(query_bin_features)
    predicted_control_velocity = decoder.predict(control_bin_features)
    row = metrics_row(
        session,
        method,
        feature_label,
        query_bin_velocity,
        predicted_query_velocity,
        control_bin_velocity,
        predicted_control_velocity,
        k,
        draw,
    )
    return row


def save_training_log(summary, method, seed, training_log_dir):
    if not training_log_dir:
        return
    os.makedirs(training_log_dir, exist_ok=True)
    with open(os.path.join(training_log_dir, f"{method}_seed{seed}.json"), "w") as file:
        json.dump(dict(method=method, seed=seed, **summary), file)


def concat_stimulation_off_sessions(load_part, part, excluded_session=None):
    """Concatenate the no-stimulation sessions in `part`, skipping one if asked."""
    per_session_datasets = []
    for session in STIMULATION_OFF_TRAINING_SESSIONS:
        if session == excluded_session:
            continue
        session_dataset = load_part(session, part)
        per_session_datasets.append(session_dataset)
    combined_dataset = concat_trial_datasets(per_session_datasets)
    return combined_dataset


def score_trial_model(
    model, session, method, feature_label, seed, query_trials, control_trials
):
    """Score a model that maps a whole trial (bins or channels x samples) to bins."""
    query_valid_bins = query_trials["valid_bins"]
    control_valid_bins = control_trials["valid_bins"]
    predicted_query_velocity = predict_trial_bins(model, query_trials)[query_valid_bins]
    predicted_control_velocity = predict_trial_bins(model, control_trials)[
        control_valid_bins
    ]
    query_bin_velocity = query_trials["targets"][query_valid_bins]
    control_bin_velocity = control_trials["targets"][control_valid_bins]
    row = metrics_row(
        session,
        method,
        feature_label,
        query_bin_velocity,
        predicted_query_velocity,
        control_bin_velocity,
        predicted_control_velocity,
        seed=seed,
    )
    return row


# ---------------------------------------------------------------------------
# 6. Decoder re-exports and loaders
# ---------------------------------------------------------------------------

# The decoder definitions live in the notebooks that own them:
#   task4_2_eegnet.ipynb      MLPRegressor, GRURegressor, CausalEEGNet
#   task4_3_parrm_ridge.ipynb RidgeRegressor, KNNRegressor and the tuning grids
# Those marked cells are executed here so `from utils import CausalEEGNet` keeps working
# for the Kaggle runners and the figure scripts, from a single definition.
NOTEBOOK_MODEL_NOTEBOOKS = ("task4_2_eegnet.ipynb", "task4_3_parrm_ridge.ipynb")
NOTEBOOK_MODEL_CELL_MARKER = "MODELS:"

NOTEBOOK_MODEL_NAMES = (
    "RidgeRegressor",
    "KNNRegressor",
    "MLPRegressor",
    "GRURegressor",
    "CausalEEGNet",
    "RIDGE_PENALTIES",
    "KNN_NEIGHBOR_GRID",
    "CALIBRATION_TUNE_TRIALS",
    "MLP_DROPOUT_P",
    "GRU_HIDDEN_SIZE",
)


def _notebook_model_cells():
    """Source of every code cell marked ``MODELS:`` in the two model notebooks."""
    sources = []
    for notebook_name in NOTEBOOK_MODEL_NOTEBOOKS:
        with open(os.path.join(TASK4_DIR, notebook_name)) as notebook_file:
            notebook = json.load(notebook_file)
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and NOTEBOOK_MODEL_CELL_MARKER in source:
                sources.append(source)
    return sources


def _export_notebook_models(names=NOTEBOOK_MODEL_NAMES):
    namespace = dict(np=np, torch=torch, nn=nn, __name__="task4_notebook_models")
    for source in _notebook_model_cells():
        exec(compile(source, "<notebook model cell>", "exec"), namespace)
    missing = [name for name in names if name not in namespace]
    if missing:
        raise ImportError(
            "task4 model definitions live in task4_2_eegnet.ipynb and "
            f"task4_3_parrm_ridge.ipynb; missing {missing}. Run kaggle/build_dataset.py "
            "so the Kaggle dataset carries both notebooks."
        )
    return {name: namespace[name] for name in names}


globals().update(_export_notebook_models())


def load_gru_datasets(variant, sessions):
    """GRU trial datasets at no lag history: (n_trials, 40 bins, 280 features)."""

    def load_part(session, part):
        split_dataset = load_split_dataset(session, part, variant, 0)
        return split_dataset

    train_trials = concat_stimulation_off_sessions(load_part, "train")
    validation_trials = concat_stimulation_off_sessions(load_part, "val")
    test_trials = {}
    for session in sessions:
        query_trials = load_part(session, "query")
        control_trials = load_same_day_control_dataset(session, variant, 0)
        test_trials[session] = (query_trials, control_trials)
    return train_trials, validation_trials, test_trials


def evaluate_gru(model, seed, test_trials, feature_label):
    rows = []
    for session, trial_pair in test_trials.items():
        query_trials, control_trials = trial_pair
        row = score_trial_model(
            model, session, "gru", feature_label, seed, query_trials, control_trials
        )
        rows.append(row)
    return rows


def load_eegnet_checkpoint(name, checkpoint_dir=None):
    """Load a saved CausalEEGNet state dict, e.g. 'task4_2_eegnet_seed0'."""
    path = os.path.join(checkpoint_dir or CHECKPOINT_DIR, f"{name}.pt")
    model = CausalEEGNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


# ---------------------------------------------------------------------------
# 7. Calibration draws (task 4.1)
# ---------------------------------------------------------------------------

DRAW_SEED = 20261106
DRAWS_PER_SIZE = 10

QUERY_DEFINITION = "fixed"


def included_on_trials(session):
    """Trial numbers of every included trial in a recording."""
    session_table = lab_loader.load_session(
        session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE
    )
    included_rows = session_table.included
    trial_numbers = included_rows.trial.astype(int).tolist()
    return sorted(trial_numbers)


def calibration_pool_and_query(session):
    """The declared calibration pool and query for one session, from splits.json."""
    entry = SPLITS["calibration"][session]
    return sorted(int(t) for t in entry["pool"]), sorted(int(t) for t in entry["query"])


def distinct_subsets(pool, k, n_draws, rng):
    """Up to n_draws distinct k-subsets of pool, redrawn until no subset repeats.

    C(len(pool), k) caps how many exist, so k equal to the pool size gives one.
    """
    target = min(n_draws, math.comb(len(pool), k))
    seen, subsets, attempts = set(), [], 0
    while len(subsets) < target and attempts < 1000 * target:
        subset = frozenset(rng.sample(pool, k))
        attempts += 1
        if subset not in seen:
            seen.add(subset)
            subsets.append(sorted(subset))
    return subsets


def sample_draws_for_session(
    session,
    ks=CALIBRATION_SIZES,
    n_draws=DRAWS_PER_SIZE,
    seed=DRAW_SEED,
    query_definition=QUERY_DEFINITION,
):
    """The k-trial calibration draws for one stimulation-on session."""
    trials = included_on_trials(session)
    rng = random.Random(
        f"{seed}:{session}"
    )  # str seeds hash with sha512, so runs repeat

    if query_definition == "fixed":
        pool, query = calibration_pool_and_query(session)
        if not set(query) <= set(trials):
            raise ValueError(
                f"{session}: the declared query leaves the included ON trials"
            )
        draws_by_k = {
            str(k): [
                dict(draw=subset, query=query)
                for subset in distinct_subsets(pool, k, n_draws, rng)
            ]
            for k in ks
        }
        return dict(pool=pool, query=query, subsets=draws_by_k)

    if query_definition != "remaining":
        raise ValueError(f"unknown query_definition {query_definition!r}")

    orders = []  # one shuffled trial order per draw index
    for _ in range(n_draws):
        shuffled_trials = list(trials)
        rng.shuffle(shuffled_trials)
        orders.append(shuffled_trials)

    # A draw is the first k entries of an order; its query is everything left over.
    draws_by_k = {
        str(k): [
            dict(
                draw=shuffled_trials[:k],
                query=sorted(set(trials) - set(shuffled_trials[:k])),
            )
            for shuffled_trials in orders
        ]
        for k in ks
    }
    return dict(pool=None, query=None, subsets=draws_by_k)


def build_draws(
    sessions=None,
    ks=CALIBRATION_SIZES,
    n_draws=DRAWS_PER_SIZE,
    seed=DRAW_SEED,
    query_definition=QUERY_DEFINITION,
):
    if sessions is None:
        sessions = sessions_in_split("test", STIMULATION_ON)

    by_session = {}
    for session in sessions:
        sampled = sample_draws_for_session(session, ks, n_draws, seed, query_definition)
        by_session[session] = dict(
            condition=str(SESSION_TABLE.loc[session, "condition"]),
            am_freq_hz=float(SESSION_TABLE.loc[session, "am_freq_hz"]),
            off_reference=SPLITS["off_reference"].get(
                session
            ),  # same-day no-stimulation session
            on_trials=included_on_trials(session),  # every ON trial of the session
            pool=sampled["pool"],  # calibration pool, or None when 'remaining'
            query=sampled["query"],  # fixed query trials, or None when 'remaining'
            subsets=sampled[
                "subsets"
            ],  # str(k) -> entries of {'draw': [...], 'query': [...]}
        )

    payload = dict(
        version=3,
        created_by="task4/utils.py",
        unit="trial",
        query_definition=query_definition,
        ks=list(ks),
        n_draws=n_draws,
        seed=seed,
        sessions=by_session,
    )
    payload["checks"] = verify_draws(payload)  # refuse to hand back an invalid payload
    return payload


def verify_draws(payload):
    failures, notes = [], []
    fixed = payload.get("query_definition") == "fixed"
    for session, entry in payload["sessions"].items():
        all_on = set(entry["on_trials"])
        pool = set(entry["pool"]) if entry.get("pool") else all_on
        query = set(entry["query"]) if entry.get("query") else None
        if len(all_on) != len(entry["on_trials"]):
            failures.append(f"{session}: on_trials holds duplicates")
        if query is not None:
            if not query <= all_on:
                failures.append(f"{session}: the query leaves the ON trials")
            if pool & query:
                failures.append(f"{session}: the pool and the query overlap")

        for k in payload["ks"]:
            draws = entry["subsets"][str(k)]
            if len(draws) > payload["n_draws"]:
                failures.append(
                    f'{session}: k={k} has {len(draws)} draws, more than {payload["n_draws"]}'
                )
            if k > len(pool):
                failures.append(f"{session}: k={k} exceeds the {len(pool)}-trial pool")

            for index, draw in enumerate(draws):
                calibration, scored = set(draw["draw"]), set(draw["query"])
                label = f"{session}: k={k} draw {index}"
                if len(calibration) != k:
                    failures.append(f"{label} is not {k} distinct trials")
                if not calibration <= pool:
                    failures.append(f"{label} leaves the calibration pool")
                if calibration & scored:
                    failures.append(f"{label} reuses a query trial")
                if fixed and scored != query:
                    failures.append(f"{label} does not use the declared query")
                if not fixed and calibration | scored != all_on:
                    failures.append(f"{label} misses ON trials")

            distinct = len({frozenset(d["draw"]) for d in draws})
            if distinct < payload["n_draws"]:
                notes.append(
                    f"{session}: k={k} has {distinct} distinct draws among "
                    f'{payload["n_draws"]}; only {math.comb(len(pool), k)} exist'
                )

    return dict(passed=not failures, failures=failures, notes=notes)


def save_draws(payload, out=DRAWS_PATH):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as file:
        json.dump(payload, file, indent=1)


def iter_draws(draws, session):
    """Yield (calibration_size, draw_index, draw_record) for one session."""
    for calibration_size in draws["ks"]:
        subsets = draws["sessions"][session]["subsets"][str(calibration_size)]
        for draw_index, draw_record in enumerate(subsets):
            yield calibration_size, draw_index, draw_record


def drawn_trial_windows(session, draw_record, variant, lag_count):
    """Calibration bin rows: (n_valid_bins, n_features) and (n_valid_bins, 2)."""
    calibration_trials = draw_record["draw"]
    trial_dataset = load_trial_dataset(
        session, calibration_trials, variant, calibration_trials, lag_count
    )
    bin_rows = valid_bin_rows(trial_dataset)
    return bin_rows


def query_trial_windows(session, draw_record, variant, lag_count):
    """Query bin rows: (n_valid_bins, n_features) and (n_valid_bins, 2)."""
    query_trials = draw_record["query"]
    calibration_trials = draw_record["draw"]
    trial_dataset = load_trial_dataset(
        session, query_trials, variant, calibration_trials, lag_count
    )
    bin_rows = valid_bin_rows(trial_dataset)
    return bin_rows


# ---------------------------------------------------------------------------
# 8. Bootstrap statistics
# ---------------------------------------------------------------------------


def bootstrap_sample_means(values, resample_count):
    """Row means of `resample_count` bootstrap resamples of `values`."""
    values = np.asarray(values, float)
    generator = np.random.default_rng(0)
    resampled = generator.choice(
        values, size=(resample_count, len(values)), replace=True
    )
    resampled_means = resampled.mean(axis=1)
    return resampled_means


def bootstrap_confidence_interval(values, resample_count=2000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    resampled = bootstrap_sample_means(values, resample_count)
    return float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def two_level_bootstrap_confidence_interval(groups, resample_count=2000, seed=0):
    """95% CI for a mean over sessions, resampling sessions and the draws inside each.

    `groups` holds one array of per-draw scores per session, so the interval covers both
    which sessions were tested and which calibration sets were drawn inside them.
    """
    arrays = [np.asarray(group, float) for group in groups]
    arrays = [array[np.isfinite(array)] for array in arrays]
    arrays = [array for array in arrays if array.size]
    if len(arrays) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    resampled = np.empty(resample_count)
    for index in range(resample_count):
        sessions = rng.integers(0, len(arrays), len(arrays))
        resampled[index] = np.mean(
            [
                arrays[session][
                    rng.integers(0, arrays[session].size, arrays[session].size)
                ].mean()
                for session in sessions
            ]
        )
    return float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


# ---------------------------------------------------------------------------
# 9. EEGNet baseline
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ = 1000
MAX_TRAINING_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
EARLY_STOP_DELTA = 1e-4
STIMULATION_FREQ_HZ = (3.0, 20.0)
STIMULATION_FREQ_MAG = (0.5, 3.0)
EEGNET_METHOD = "eegnet"
# The stored label stays "eegnet_stimulation_tones" so existing result files still resolve.
EEGNET_FREQ_AUGMENTATION = "eegnet_stimulation_tones"
EEGNET_ENSEMBLE_METHOD = "eegnet_ensemble"
ENSEMBLE_SEED = -1  # seed column value for a score averaged over every seed
EEGNET_DEFAULT_SETTINGS = dict(
    temporal_filters_n=8,
    depth_multiplier=2,
    kernel_samples=500,
    separable_samples=62,
    dropout=0.5,
)


def same_day_control_lfp_dataset(stimulation_on_session):
    """LFP trials from the same-day no-stimulation recording of an ON session."""
    control_session = same_day_control_session(stimulation_on_session)
    control_dataset = load_lfp_dataset(control_session, None)
    return control_dataset


def stimulation_off_train_dataset():
    """LFP trials from the no-stimulation recordings in the train split."""
    training_trials = concat_stimulation_off_sessions(load_lfp_split, "train")
    return training_trials


def stimulation_off_validation_dataset():
    """LFP trials from the no-stimulation recordings in the validation split."""
    validation_sessions = sessions_in_split("val", STIMULATION_OFF)
    per_session_datasets = []
    for session in validation_sessions:
        session_dataset = load_lfp_dataset(session, None)
        per_session_datasets.append(session_dataset)
    validation_trials = concat_trial_datasets(per_session_datasets)
    return validation_trials


def stimulation_on_test_pairs():
    """session -> (query trial dataset, same-day no-stimulation control dataset)."""
    test_sessions = sessions_in_split("test", STIMULATION_ON)
    test_trials_by_session = {}
    for session in test_sessions:
        query_trials = load_lfp_split(session, "query")
        control_trials = same_day_control_lfp_dataset(session)
        test_trials_by_session[session] = (query_trials, control_trials)
    return test_trials_by_session


def amp10_test_datasets(sessions=None):
    """Query trials read from the 10x semi-synthetic recording.

    Only the ON recording is amplified, so the same-day control and the targets are the
    same arrays stimulation_on_test_pairs() uses.
    """
    if not os.path.isdir(SEMISYNTHETIC_EXPORT_DIR):
        raise FileNotFoundError(
            f"No semi-synthetic exports at {SEMISYNTHETIC_EXPORT_DIR}; run\n"
            "  python semisynthetic/EECS182_make_semisynthetic_h5.py "
            "--root data --split val-test --h5-only"
        )
    selected_sessions = sessions or sessions_in_split("test", STIMULATION_ON)
    loader_paths = semisynthetic_paths()
    trials_by_session = {}
    for session in selected_sessions:
        query_trials = load_lfp_split(session, "query", loader_paths)
        control_trials = same_day_control_lfp_dataset(session)
        trials_by_session[session] = (query_trials, control_trials)
    return trials_by_session


def parrm_test_datasets(sessions=None, parrm_dir=None):
    """Query trials read from the PARRM-cleaned ON recording.

    The cleaned file only exists for stimulation-ON sessions, so the same-day control
    stays on the recorded export.
    """
    selected_sessions = sessions or sessions_in_split("test", STIMULATION_ON)
    if parrm_dir:
        loader_paths = parrm_paths(parrm_dir)
    else:
        loader_paths = parrm_paths()
    trials_by_session = {}
    for session in selected_sessions:
        query_trials = load_lfp_split(session, "query", loader_paths, source="parrm")
        control_trials = same_day_control_lfp_dataset(session)
        trials_by_session[session] = (query_trials, control_trials)
    return trials_by_session


PARRM_AMP10_REUSE_DIR = os.path.join(
    DATA_DIR, "semi_synthetic", "parrm_amp10_reuse", "parrm_h5"
)

PARRM_LFP_VARIANT = "parrm_lfp"
PARRM_REUSE_VARIANT = "parrm_amp10_reuse"

ARM_VARIANTS = {
    "recorded": "raw",
    "amp10": AMP10_VARIANT,
    "parrm_recorded": PARRM_LFP_VARIANT,
    "parrm_amp10_reuse": PARRM_REUSE_VARIANT,
}


def amp10_arms(sessions=None, arms=None):
    """Score-ready datasets for each input arm, keyed by arm name.

    'parrm_recorded' is PARRM re-estimated on the amplified signal: the estimate scales
    with the artifact, so subtracting it returns the recorded cleaned LFP.
    'parrm_amp10_reuse' subtracts one copy of the 1x template from the 10x signal, leaving
    eight copies.
    """
    selected_arms = arms or list(ARM_VARIANTS)
    datasets_by_arm = {}
    if "recorded" in selected_arms:
        datasets_by_arm["recorded"] = stimulation_on_test_pairs()
    if "amp10" in selected_arms:
        datasets_by_arm["amp10"] = amp10_test_datasets(sessions)
    if "parrm_recorded" in selected_arms:
        datasets_by_arm["parrm_recorded"] = parrm_test_datasets(sessions)
    if "parrm_amp10_reuse" in selected_arms:
        if not os.path.isdir(PARRM_AMP10_REUSE_DIR):
            raise FileNotFoundError(
                f"No reused-template files at {PARRM_AMP10_REUSE_DIR}; "
                "the 'parrm_amp10_reuse' arm needs the semi-synthetic build "
                "(see README, section 4.2.10)"
            )
        reuse_datasets = parrm_test_datasets(sessions, parrm_dir=PARRM_AMP10_REUSE_DIR)
        datasets_by_arm["parrm_amp10_reuse"] = reuse_datasets
    return datasets_by_arm


def train_with_early_stopping(
    model,
    train_trials,
    validation_trials,
    seed,
    max_epochs=MAX_TRAINING_EPOCHS,
    patience=EARLY_STOP_PATIENCE,
):
    """Train on trial tensors, stop on validation R², restore the best epoch."""
    
    # setup:
    device = select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)

    dataset_keys = ("features", "targets", "valid_bins")
    train_tensors = to_tensors([train_trials[key] for key in dataset_keys])

    # Validation is scored in numpy, once per epoch and outside autograd.
    validation_tensors = to_tensors(
        [validation_trials[key] for key in dataset_keys]
    )
    (
        validation_bin_features,
        validation_bin_velocity,
        validation_valid_bins,
    ) = [tensor.numpy() for tensor in validation_tensors]

    optimizers = [Muon([p for p in model.parameters() if p.ndim >= 2])]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    if vectors:
        optimizers.append(torch.optim.Adam(vectors, lr=ADAM_LR))

    init_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    best_r2, best_epoch = -np.inf, 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(
            model, train_tensors, masked_mse, optimizers,
            rng.permutation(len(train_tensors[0])), TRIAL_BATCH_SIZE, device,
        )
        val_r2 = mean_r2(
            validation_bin_velocity,
            predict(model, validation_bin_features),
            validation_valid_bins,
        )
        history.append({"epoch": epoch, "train_loss": loss, "val_r2": val_r2})

        if val_r2 > best_r2 + EARLY_STOP_DELTA:
            best_r2, best_epoch = val_r2, epoch
            best_weights = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

        set_learning_rates(
            optimizers, init_lrs, cosine_lr_multiplier(epoch, max_epochs)
        )

    model.load_state_dict(best_weights)
    model.eval()
    return model, {
        "training_log": history,
        "best_val_r2": best_r2,
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
    }


def eegnet_sweep_settings():
    settings = EEGNET_DEFAULT_SETTINGS
    return [
        settings,
        settings | dict(kernel_samples=250),
        settings | dict(kernel_samples=1000),
        settings | dict(dropout=0.25),
        settings | dict(dropout=0.75),
        settings | dict(depth_multiplier=1),
        settings | dict(kernel_samples=250, dropout=0.25),
        settings | dict(kernel_samples=1000, dropout=0.25),
    ]


def best_eegnet_settings(sweep):
    best = sweep.iloc[0]
    return dict(
        temporal_filters_n=int(best.temporal_filters_n),
        depth_multiplier=int(best.depth_multiplier),
        kernel_samples=int(best.kernel_samples),
        separable_samples=int(best.separable_samples),
        dropout=float(best.dropout),
    )


def add_stimulation_tones(
    dataset,
    seed,
    strength_range=STIMULATION_FREQ_MAG,
    frequencies=STIMULATION_FREQ_HZ,
):
    """Add one sine per trial to the LFP tensor.

    dataset["features"] is (n_trials, n_channels, n_samples); every channel of a trial
    gets the same frequency and phase, scaled by a per-channel gain.
    """
    generator = np.random.default_rng(seed)
    trial_channels_by_samples = dataset["features"].copy()
    n_trials, n_channels, n_samples = trial_channels_by_samples.shape
    sample_times = np.arange(n_samples) / SAMPLE_RATE_HZ

    for trial in range(n_trials):
        frequency = generator.choice(frequencies)
        strength = generator.uniform(*strength_range)
        phase = generator.uniform(0, 2 * np.pi)
        channel_gain = generator.uniform(0.5, 1.5, size=(n_channels, 1))
        trial_channels_by_samples[trial] += (
            strength
            * channel_gain
            * np.sin(2 * np.pi * frequency * sample_times + phase)
        )

    return dataset | dict(features=trial_channels_by_samples)


def score_ensemble(
    models,
    session,
    method,
    feature_label,
    query_trials,
    control_trials,
    seed=ENSEMBLE_SEED,
):
    """Average the per-bin velocity predicted by every model in the ensemble."""
    query_valid_bins = query_trials["valid_bins"]
    control_valid_bins = control_trials["valid_bins"]
    query_predictions = [predict(model, query_trials["features"]) for model in models]
    control_predictions = [
        predict(model, control_trials["features"]) for model in models
    ]
    predicted_query_velocity = np.mean(query_predictions, axis=0)
    predicted_control_velocity = np.mean(control_predictions, axis=0)
    query_bin_velocity = query_trials["targets"][query_valid_bins]
    control_bin_velocity = control_trials["targets"][control_valid_bins]
    row = metrics_row(
        session,
        method,
        feature_label,
        query_bin_velocity,
        predicted_query_velocity[query_valid_bins],
        control_bin_velocity,
        predicted_control_velocity[control_valid_bins],
        seed=seed,
    )
    return row


def scores_per_day(scores, column):
    scored = scores.assign(day=scores.session.str.split("-").str[1])
    per_session = scored.groupby(["method", "day", "session"], as_index=False)[
        column
    ].mean()
    return per_session.groupby(["method", "day"], as_index=False)[column].mean()


def day_level_summary(scores, column):
    per_day = scores_per_day(scores, column)
    rows = []
    for method, method_days in per_day.groupby("method"):
        values = method_days[column].to_numpy()
        low, high = bootstrap_confidence_interval(values)
        rows.append(
            dict(
                method=method,
                column=column,
                days=len(values),
                mean=float(np.mean(values)),
                low=low,
                high=high,
            )
        )
    return pd.DataFrame(rows)


def write_baseline_parquet(scores, path):
    columns = [
        "method",
        "feature_variant",
        "session",
        "am_freq",
        "k",
        "draw",
        "seed",
        "r2_vx",
        "r2_vy",
        "r2_ave",
        "r2_off_vx",
        "r2_off_vy",
        "r2_off_ave",
        "n_bins",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scores[columns].to_parquet(path, index=False)
