"""Command-line entry point: exact full calculation is the default."""
import argparse
from pathlib import Path
import sympy as s
from . import KIN, PHI2, RPHI2, PHI4, solve, run, evaluate


def main():
    parser=argparse.ArgumentParser(description='Action-first symbolic in-in pipeline')
    parser.add_argument('--action', required=True,
          help='e.g. -KIN/2-m**2*PHI2/2-RPHI2/12-lam*PHI4/24')
    parser.add_argument('--output', default='full_results')
    parser.add_argument('--family', nargs='+',choices=('scalar','tensor'),default=['scalar','tensor'])
    parser.add_argument('--workers', type=int, default=1, help='parallel processes (numerics default: half the cores)')
    parser.add_argument('--restart', action='store_true',help='Erase previous results in the output directory')
    parser.add_argument('--quick', action='store_true',help='Only create the small preview from the previous version')
    parser.add_argument('--symbolic', action='store_true',help='Only the exact symbolic Wick jobs (v0.2 behaviour)')
    parser.add_argument('--value', action='append', default=[], metavar='NAME=NUMBER',
                        help='numerical value of an action symbol, e.g. --value m=1 --value lam=0.1 (units H_e)')
    parser.add_argument('--q', type=float, nargs='+', default=[0.1, 0.1316, 0.1732, 0.228, 0.3],
                        help='Fourier modes of the metric perturbation (units a_e H_e)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='numerical setting, e.g. --set pmax=8 --set np=60 --set nend=5.5 --set mu_R=2')
    parser.add_argument('--stop-after',type=int, default=None,
                        help='For testing only: stop after N pending diagrams')
    args=parser.parse_args()
    syms={k.name:k for k in (KIN,PHI2,RPHI2,PHI4)}
    for name in ('m','lam','lambda'):
        syms[name]=s.Symbol(name,real=True)
    act=s.sympify(args.action,locals=syms)
    vals = dict(v.split('=', 1) for v in args.value)
    sets = {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        try: sets[k] = int(v)
        except ValueError:
            try: sets[k] = float(v)
            except ValueError: sets[k] = None if v == 'None' else v
    if not args.quick and not args.symbolic:
        res = evaluate(act, {k: float(v) for k, v in vals.items()}, families=args.family, q_grid=tuple(args.q),
                       output_dir=args.output, workers=args.workers if args.workers > 1 else None, **sets)
        print(res.summary())
        return
    if args.quick:
        result=run(act,args.output,families=args.family)
        print('Preview:',result.files)
    else:
        result=solve(act,args.output,families=args.family,workers=args.workers,
                     resume=not args.restart,stop_after=args.stop_after)
        print('Saved diagrams:',result.path/'diagrams.sqlite')
        print('Exact exported terms:',result.path/'expanded_wick.jsonl')
        print('Progress:',result.status)

if __name__=='__main__':
    main()
