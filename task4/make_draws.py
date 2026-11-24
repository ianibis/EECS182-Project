"""Command-line entry point for the task 4.1 calibration draws.

Thin wrapper over `utils.build_draws`, so the notebook and the CLI cannot drift
apart. Run from the repository root:

    .venv/bin/python task4/make_draws.py
    .venv/bin/python task4/make_draws.py --n-draws 20 --seed 7 --force
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils as U

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=U.DRAWS_PATH)
    parser.add_argument('--ks', type=int, nargs='+', default=list(U.CALIBRATION_SIZES))
    parser.add_argument('--n-draws', type=int, default=U.DRAWS_PER_SIZE)
    parser.add_argument('--seed', type=int, default=U.DRAW_SEED)
    parser.add_argument('--sessions', nargs='+', help='subset of test session ids')
    parser.add_argument('--query-definition', default=U.QUERY_DEFINITION,
                        choices=('fixed', 'remaining'),
                        help="'fixed' scores the declared query, 'remaining' scores what the draw leaves")
    parser.add_argument('--force', action='store_true', help='overwrite an existing --out file')
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f'{args.out} exists; pass --force to overwrite it')

    payload = U.build_draws(sessions=args.sessions, ks=sorted(args.ks),
                            n_draws=args.n_draws, seed=args.seed,
                            query_definition=args.query_definition)
    U.save_draws(payload, args.out)

    checks = payload['checks']
    print(f'wrote {args.out}')
    print(f"  sessions={len(payload['sessions'])} ks={payload['ks']} "
          f"n_draws={payload['n_draws']} seed={payload['seed']} "
          f"query={payload['query_definition']}")
    print(f"  checks passed={checks['passed']}")
    for failure in checks['failures'][:10]:
        print(f'    FAIL {failure}')
    for note in checks['notes'][:5]:
        print(f'    note: {note}')
    return 0 if checks['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
