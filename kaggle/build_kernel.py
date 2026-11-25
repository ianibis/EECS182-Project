"""Build the Kaggle kernel folder from train_task4_2.py.

    python kaggle/build_kernel.py && kaggle kernels push -p kaggle/kernel
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, 'kernel')
SOURCE = open(os.path.join(HERE, 'train_task4_2.py')).read()

METADATA = dict(
    id='kevinseriousying/task4-2-eegnet-lfp',
    title='task4-2-eegnet-lfp',
    code_file='task4_2_kaggle.ipynb',
    language='python',
    kernel_type='notebook',
    is_private=True,
    enable_gpu=True,
    enable_internet=True,
    dataset_sources=['kevinseriousying/eecs182-lfp'],
    competition_sources=[],
    kernel_sources=[],
)

INTRO = """# Task 4.2 — CausalEEGNet on 1 kHz LFP

Trains the task 4.2 decoder from `task4/task4_2_eegnet.ipynb` twice — once on plain
stimulation-off trials and once with synthetic 3/20 Hz frequencies added — five randomised seeds
each, then scores every stimulated test session on its own input and on the same day's
stimulation-off recording.

Data comes from the attached `eecs182-lfp` dataset (36 sessions, 3.4 GB). Outputs land in
`/kaggle/working/task4_2`: ten checkpoints, ten training logs, `task4_2_eegnet_off.csv`
(84 score rows) and `task4_2_summary.csv`. Expect roughly 30–45 min on a T4/P100.
"""

def code(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=source.strip('\n').splitlines(keepends=True))


notebook = dict(
    cells=[
        dict(cell_type='markdown', metadata={}, source=INTRO.splitlines(keepends=True)),
        code('!pip install -q braindecode==2.0.0'),
        code('import os\nos.environ["LFP_ROOT"] = "/kaggle/input/eecs182-lfp"\n\n' + SOURCE),
        code('!cd /kaggle/working && zip -qr task4_2_run.zip task4_2 && ls -la /kaggle/working'),
    ],
    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                  language_info=dict(name='python')),
    nbformat=4,
    nbformat_minor=5,
)

os.makedirs(KERNEL, exist_ok=True)
with open(os.path.join(KERNEL, 'kernel-metadata.json'), 'w') as handle:
    json.dump(METADATA, handle, indent=1)
with open(os.path.join(KERNEL, 'task4_2_kaggle.ipynb'), 'w') as handle:
    json.dump(notebook, handle, indent=1)
print('wrote', KERNEL, len(SOURCE.splitlines()), 'lines of training code')
