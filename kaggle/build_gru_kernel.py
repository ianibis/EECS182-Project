"""Build the Kaggle kernel folder for the GRU run from train_gru.py.

    python kaggle/build_gru_kernel.py && kaggle kernels push -p kaggle/kernel_gru
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, 'kernel_gru')
SOURCE = open(os.path.join(HERE, 'train_gru.py')).read()

METADATA = dict(
    id='kevinseriousying/task4-gru-lfp',
    title='task4-gru-lfp',
    code_file='task4_gru_kaggle.ipynb',
    language='python',
    kernel_type='notebook',
    is_private=True,
    enable_gpu=True,
    # The run is pure torch over the mounted dataset, so internet can stay off and the
    # kernel schedules in the (much shorter) offline GPU queue.
    enable_internet=False,
    dataset_sources=['kevinseriousying/eecs182-lfp'],
    competition_sources=[],
    kernel_sources=[],
)

INTRO = """# Task 4 — GRU decoder on cached spectral features

Sweeps twelve GRU settings on stimulation-off validation trials, then trains five seeds at the
winner and scores every stimulated test session plus its matched no-stimulation recording.
This is the sequence baseline the EEGNet is compared against.

Data comes from the attached `eecs182-lfp` dataset (36 sessions, 3.7 GB, now including the
`features/export_none` caches the GRU reads). Outputs land in `/kaggle/working/task4_gru`:
`gru_sweep.csv`, `task4_gru_final.csv` (per session and seed, plus the ensemble),
`task4_gru_seeds.csv`, `task4_gru_config.json` and the seed-0 checkpoint.
"""


def code(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=source.strip('\n').splitlines(keepends=True))


notebook = dict(
    cells=[
        dict(cell_type='markdown', metadata={}, source=INTRO.splitlines(keepends=True)),
        code('''import os

root = None
for dirpath, dirs, files in os.walk('/kaggle/input'):
    if os.path.basename(dirpath) == 'task4' and 'utils.py' in files:
        root = os.path.dirname(dirpath)
        break
    if dirpath[len('/kaggle/input'):].count(os.sep) >= 5:
        dirs[:] = []
os.environ['LFP_ROOT'] = root
print('LFP_ROOT =', root)
print('features present:', os.path.isdir(os.path.join(root, 'data', 'features', 'export_none')))'''),
        code(SOURCE),
        code('!cd /kaggle/working && zip -qr task4_gru_run.zip task4_gru && '
             'ls -la /kaggle/working'),
    ],
    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                  language_info=dict(name='python')),
    nbformat=4,
    nbformat_minor=5,
)

os.makedirs(KERNEL, exist_ok=True)
with open(os.path.join(KERNEL, 'kernel-metadata.json'), 'w') as handle:
    json.dump(METADATA, handle, indent=1)
with open(os.path.join(KERNEL, 'task4_gru_kaggle.ipynb'), 'w') as handle:
    json.dump(notebook, handle, indent=1)
print('wrote', KERNEL, len(SOURCE.splitlines()), 'lines of training code')
