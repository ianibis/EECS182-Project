"""Download the Kaggle run and install its artifacts into task4/.

    python kaggle/fetch_results.py
"""
import os
import shutil
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SLUG = 'kevinseriousying/task4-2-eegnet-lfp'
OUT = os.path.join(HERE, 'out')
RUN = os.path.join(OUT, 'task4_2')

subprocess.run(['kaggle', 'kernels', 'output', SLUG, '-p', OUT], check=True)
archive = os.path.join(OUT, 'task4_2_run.zip')
if os.path.exists(archive):
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(OUT)

for name, target in (('checkpoints', 'checkpoints'), ('training_logs', 'training_logs'),
                     ('task4_2_eegnet_off.csv', 'results/task4_2_eegnet_off.csv'),
                     ('task4_2_summary.csv', 'results/task4_2_summary.csv')):
    source = os.path.join(RUN, name)
    if os.path.isdir(source):
        shutil.copytree(source, os.path.join(REPO, 'task4', target), dirs_exist_ok=True)
    elif os.path.exists(source):
        shutil.copy2(source, os.path.join(REPO, 'task4', target))
print('installed task4/checkpoints, task4/training_logs and the two result tables')
