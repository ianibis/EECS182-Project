"""Load the notebook's shared cells so scripts can reuse its helpers.

Cells are located by a unique substring of their source, so inserting or
reordering notebook cells does not break the scripts.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, 'cs182_decoder_grouped.ipynb')

# Anchors for the cells every script needs: imports, data, metrics, training,
# evaluation helpers, and the ridge model.
SHARED = (
    'import glob',
    'PROJECT_ROOT =',
    'MIN_R2_GAP',
    'def select_device',
    'def collect_windows',
    'class RidgeRegressor',
    'ALPHAS =',
)

KNN = ('N_PCA =', 'KNN_TUNE_K =')
NEURAL = ('class MLPRegressor', 'class GRURegressor', 'class EEGNetRegressor',
          'def load_lfp_split')


def _cell_index(cells, anchor):
    """Return the index of the first code cell containing the anchor."""
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and anchor in ''.join(cell['source']):
            return index
    raise LookupError(f'no code cell contains {anchor!r}')


def load(anchors=SHARED, extra=(), notebook=NOTEBOOK):
    """Execute the anchored notebook cells in notebook order and return the namespace."""
    with open(notebook) as file:
        cells = json.load(file)['cells']
    indices = sorted({_cell_index(cells, anchor) for anchor in anchors + tuple(extra)})
    namespace = {'__name__': '__main__'}
    for index in indices:
        source = ''.join(cells[index]['source'])
        exec(compile(source, f'cell-{index}', 'exec'), namespace)
    return namespace
