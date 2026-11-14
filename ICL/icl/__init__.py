"""Section 5: in-context decoder for stimulation recalibration.

Importing this package makes the team's loader and feature modules, which live at the repo
root, importable. They are imported here without writing bytecode, so no __pycache__ appears
in the repo root.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import EECS182_NHP_data_loader  # noqa: E402,F401
    import EECS182_NHP_features  # noqa: E402,F401
finally:
    sys.dont_write_bytecode = _write_bytecode
