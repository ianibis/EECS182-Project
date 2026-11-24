"""Assemble the Kaggle dataset folder for task 4.2.

Copies the code and the 36 sessions the decoders touch (14 train + 3 validation + 14
test + 5 same-day OFF references) into kaggle/dataset/, mirroring the repository layout
so task4/utils.py resolves its paths there. Data files are hardlinked, so the folder
costs no extra disk.

    python kaggle/build_dataset.py && kaggle datasets create -p kaggle/dataset
"""
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO, 'kaggle', 'dataset')
SLUG = 'eecs182-lfp'

sys.path.insert(0, os.path.join(REPO, 'task4'))
import utils as U  # noqa: E402  (needs the repository layout, which is what we are copying)

SESSIONS = sorted(set(U.sessions_in_split('train', 0)) | set(U.sessions_in_split('val', 0))
                  | set(U.sessions_in_split('test', 1))
                  | {U.SPLITS['off_reference'][s] for s in U.sessions_in_split('test', 1)})

FILES = ['EECS182_NHP_data_loader.py', 'EECS182_NHP_features.py', 'EECS182_data_manifest.csv',
         'splits.json', 'conditions_eACS_AM-kHz_v2.csv', os.path.join('data', 'manifest.csv'),
         os.path.join('task4', 'utils.py')]


def link(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    for name in FILES:
        link(os.path.join(REPO, name), os.path.join(DEST, name))
    for session in SESSIONS:
        link(os.path.join(REPO, 'data/export_outputs/h5_export_outputs', f'{session}.h5'),
             os.path.join(DEST, 'data/export_outputs/h5_export_outputs', f'{session}.h5'))
        link(os.path.join(REPO, 'data/kinematics_v3/bins50ms', f'{session}_bins50ms.csv'),
             os.path.join(DEST, 'data/kinematics_v3/bins50ms', f'{session}_bins50ms.csv'))
        link(os.path.join(REPO, 'data/export_outputs/trials', f'{session}_trials.csv'),
             os.path.join(DEST, 'data/export_outputs/trials', f'{session}_trials.csv'))

    metadata = dict(title='EECS182 LFP task 4', id=f'kevinseriousying/{SLUG}',
                    licenses=[dict(name='CC0-1.0')])
    with open(os.path.join(DEST, 'dataset-metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=1)

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(DEST) for name in names)
    print(f'{len(SESSIONS)} sessions, {total / 1e9:.2f} GB -> {DEST}')


if __name__ == '__main__':
    main()
