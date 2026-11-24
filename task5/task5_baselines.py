"""A strong labels-only baseline: predict velocity without reading any brain signal.

The raw average of k context trials is noisy when k is small, so it is too easy to beat.
This baseline instead shrinks that average toward the typical trajectory of the task
sequence, bin by bin:

    prediction(t) = (sum of context velocity at t + lam * typical(t)) / (n valid at t + lam)

`typical` and `lam` are learned from training-day OFF sessions only.
"""

import numpy as np

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data

SHRINKAGE_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e9)


def mean_by_bin(velocity, valid):
    """Mean over trials at each bin, using valid bins only: (40, 2)."""
    counts = valid.sum(axis=0)[:, None]
    totals = (np.nan_to_num(velocity) * valid[..., None]).sum(axis=0)
    return totals / np.maximum(counts, 1)


def shrink(context_velocity, context_valid, typical, lam):
    """Shrink the context mean toward the typical trajectory."""
    counts = context_valid.sum(axis=0)[:, None]
    totals = (np.nan_to_num(context_velocity) * context_valid[..., None]).sum(axis=0)
    return (totals + lam * typical) / np.maximum(counts + lam, 1e-9)


class LabelsOnlyBaseline:
    def __init__(self, seed=0):
        self.sessions = {}
        for session in data.usable_sessions('train', 'no_stim'):
            arrays = data.load_session_arrays(session)
            fit = data.part_mask(session, arrays['trial'], 'train')
            self.sessions[session] = dict(
                sequence=str(C.manifest().loc[session, 'sequence']),
                fit_velocity=arrays['velocity'][fit], fit_valid=arrays['valid'][fit],
                held_velocity=arrays['velocity'][~fit], held_valid=arrays['valid'][~fit])
        self.typical = {sequence: self._typical(sequence) for sequence in
                        {entry['sequence'] for entry in self.sessions.values()}}
        self.lam = self._choose_shrinkage(seed)

    def _typical(self, sequence, exclude=None):
        chosen = [e for name, e in self.sessions.items() if e['sequence'] == sequence and name != exclude]
        return mean_by_bin(np.concatenate([e['fit_velocity'] for e in chosen]),
                           np.concatenate([e['fit_valid'] for e in chosen]))

    def _choose_shrinkage(self, seed):
        """Pick the shrinkage with the lowest error on held-out blocks, leaving each session out of its prior."""
        rng = np.random.default_rng(seed)
        errors = np.zeros(len(SHRINKAGE_GRID))
        for name, entry in self.sessions.items():
            typical = self._typical(entry['sequence'], exclude=name)
            valid = entry['held_valid']
            for k in cfg.K_VALUES:
                for _ in range(1 if k == 0 else 5):
                    rows = rng.choice(len(entry['fit_velocity']), k, replace=False)
                    for index, lam in enumerate(SHRINKAGE_GRID):
                        guess = shrink(entry['fit_velocity'][rows], entry['fit_valid'][rows], typical, lam)
                        errors[index] += ((entry['held_velocity'] - guess)[valid] ** 2).mean()
        return float(SHRINKAGE_GRID[int(errors.argmin())])

    def predict(self, session, context_velocity_cm, context_valid, shape):
        """Labels-only velocity for all query trials of a session, in cm/s."""
        sequence = str(C.manifest().loc[session, 'sequence'])
        typical = self.typical.get(sequence)
        if typical is None:     # a sequence with no training OFF session
            typical = np.mean(list(self.typical.values()), axis=0)
        return np.broadcast_to(shrink(context_velocity_cm, context_valid, typical, self.lam), shape)
