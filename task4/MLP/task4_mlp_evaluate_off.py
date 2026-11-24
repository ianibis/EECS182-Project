"""Evaluate a single MLP trained on stimulation-OFF recordings."""
import argparse
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'task3'))

from task3_paths import MLP_RESULTS, mlp_checkpoint

from task4_mlp_evaluate import evaluate

# The conventional Task 4 calibration recipe, fixed before evaluation.
CALIBRATION = (0.0001, 20, {
    'full': [(10, 0.1), (10, 0.1)],
    'time': [(10, 1.0), (10, 1.0)],
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--checkpoint', type=Path,
                        default=mlp_checkpoint('off'))
    parser.add_argument('--compare-gru', action='store_true',
                        help='Add paired scores from an existing Task 4 GRU results CSV')
    parser.add_argument('--ablation', action='store_true',
                        help='Also score frozen and ridge methods, with and without correction')
    parser.add_argument('--ridge-penalty', type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.ridge_penalty) or args.ridge_penalty <= 0:
        parser.error('--ridge-penalty must be finite and positive')

    torch.set_num_threads(4)
    evaluate(args.split, args.device, args.ridge_penalty if args.ablation else None,
             runs=('off',), checkpoint=args.checkpoint, calibration=CALIBRATION,
             compare_gru=args.compare_gru, output_root=MLP_RESULTS / 'off_only')


if __name__ == '__main__':
    main()
