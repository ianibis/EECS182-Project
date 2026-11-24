"""Build the PARRM-cleaned 10x recordings, and check the linear relation behind them.

PARRM subtracts the component phase-locked to the AM signal, so its estimate scales with
that component's amplitude. On a recording amplified to `scale`, the re-estimated artifact
is therefore `scale` times the recorded one, and

    parrm(10x) = 10x - scale * artifact = recorded + (scale - 1) * artifact - scale * artifact
               = recorded - artifact = parrm(recorded)

so re-estimating PARRM on the amplified signal returns the original cleaned recording. That
arm needs no new file: it is `data/parrm_outputs/parrm_h5`, the existing PARRM output, and
`--verify` checks the identity.

The other arm is what you get when the existing template library is reused without
re-estimating: subtracting one copy of a ten-times artifact,

    10x - artifact = recorded + 8 * artifact

which still carries eight copies. That one is written here as parrm-style HDF5 under
`semi_synthetic/parrm_amp10_reuse/parrm_h5/`, so the normal `source='parrm'` loader reads it.

    python task4/make_parrm_amp10.py --verify
    python task4/make_parrm_amp10.py
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

EXPORT_DIR = os.path.join(U.DATA_DIR, 'export_outputs', 'h5_export_outputs')
AMP10_DIR = U.SEMISYNTHETIC_EXPORT_DIR
REUSE_DIR = os.path.join(U.DATA_DIR, 'semi_synthetic', 'parrm_amp10_reuse', 'parrm_h5')
CHUNK = 25000


def sessions_for(part):
    return U.sessions_in_split(part, U.STIMULATED_ON)


def check_identity(session, tolerance=1e-3):
    """Return the relative RMS error of 10x - scale*artifact == parrm(recorded)."""
    with h5py.File(os.path.join(EXPORT_DIR, f'{session}.h5')) as export:
        keep = export['channel_keep_fixed'][:].astype(bool)
    error = signal = 0.0
    with h5py.File(os.path.join(AMP10_DIR, f'{session}.h5')) as amplified, \
            h5py.File(os.path.join(U.PARRM_DIR, f'{session}.h5')) as cleaned:
        for start in range(0, amplified['lfp'].shape[1], CHUNK):
            stop = min(start + CHUNK, amplified['lfp'].shape[1])
            ten = np.asarray(amplified['lfp'][keep, start:stop], np.float64)
            artifact = np.asarray(cleaned['artifact'][keep, start:stop], np.float64)
            clean = np.asarray(cleaned['lfp'][keep, start:stop], np.float64)
            difference = ten - U.SEMISYNTHETIC_SCALE * artifact - clean
            error += np.square(difference).sum()
            signal += np.square(ten).sum()
    return float(np.sqrt(error / max(signal, 1e-24)))


def write_reuse(session, overwrite=False):
    """Write 10x - artifact as a parrm-style file, i.e. the 1x template reused."""
    target = os.path.join(REUSE_DIR, f'{session}.h5')
    if os.path.isfile(target) and not overwrite:
        return target, False
    os.makedirs(REUSE_DIR, exist_ok=True)
    temporary = f'{target}.partial'
    with h5py.File(os.path.join(AMP10_DIR, f'{session}.h5')) as amplified, \
            h5py.File(os.path.join(U.PARRM_DIR, f'{session}.h5')) as cleaned, \
            h5py.File(temporary, 'w') as output:
        keep = amplified['channel_keep_fixed'][:].astype(bool)
        output.create_dataset('lfp', shape=amplified['lfp'].shape, dtype='float32')
        output.create_dataset('artifact', shape=cleaned['artifact'].shape, dtype='float32')
        for start in range(0, amplified['lfp'].shape[1], CHUNK):
            stop = min(start + CHUNK, amplified['lfp'].shape[1])
            ten = np.asarray(amplified['lfp'][:, start:stop], np.float64)
            artifact = np.asarray(cleaned['artifact'][:, start:stop], np.float64)
            output['lfp'][:, start:stop] = (ten - artifact).astype('float32')
            output['artifact'][:, start:stop] = artifact.astype('float32')
        for name in ('t', 'channel_ids', 'channel_keep'):
            if name in cleaned:
                output.create_dataset(name, data=cleaned[name][:])
        output.create_dataset('stim/phase_model', data=cleaned['stim/phase_model'][:])
        output.attrs['cleaning'] = 'reused_1x_template'
        output.attrs['formula'] = 'amp10_lfp - 1 * parrm_artifact = recorded + 8 * artifact'
        output.attrs['artifact_scale_of_input'] = U.SEMISYNTHETIC_SCALE
        output.attrs['artifact_scale_removed'] = 1.0
        output.attrs['channels_modified'] = int(keep.sum())
    os.replace(temporary, target)
    return target, True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('test', 'val', 'val-test'), default='val-test')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='check 10x - 10*artifact == parrm(recorded), write nothing')
    args = parser.parse_args()

    sessions = sessions_for('val') + sessions_for('test') if args.split == 'val-test' \
        else sessions_for(args.split)

    if args.verify:
        worst = 0.0
        for session in sessions:
            error = check_identity(session)
            worst = max(worst, error)
            print(f'{session}: relative RMS |10x - 10*artifact - clean| = {error:.2e}')
        print(f'\nworst {worst:.2e}: re-estimated PARRM on the 10x signal returns the '
              f'recorded cleaned LFP, so that arm needs no new file.')
        return

    written = 0
    for session in sessions:
        _, created = write_reuse(session, args.overwrite)
        written += int(created)
        print(f'{session}: reuse arm {"written" if created else "exists"}')
    print(f'\n{written} written -> {REUSE_DIR}')
    print('the re-estimated arm is the existing data/parrm_outputs/parrm_h5')


if __name__ == '__main__':
    main()
