"""Automatic local order-lambda covariant mass/vacuum counterterm vertices.

Implements the mass subdivergence and double-bubble vacuum counterterms of
Pasted text(7).txt at xi=1/6.  The independent *free* gravitational one-loop
counterterm must also be varied in the full metric.  That is a distributional
renormalization step and is not substituted by the expressions here.

The counterterms are proportional to the interaction coupling lambda. The
returned dictionary coefficients have lambda removed to match (lambda,eps)
order conventions in the in-in Dyson enumerator.
"""
from dataclasses import dataclass
import sympy as sp
from .symbols import P,PS,SH,a,T

@dataclass(frozen=True)
class CountertermInsertions:
    interaction: dict
    observable: dict
    mass_shift_per_lambda: sp.Expr
    vacuum_shift_per_lambda: sp.Expr


def local_counterterms(action,family):
    if family not in ('scalar','tensor'):
        raise ValueError('family should be scalar or tensor')
    if action.coupling == 0:
        return CountertermInsertions({}, {}, sp.S.Zero, sp.S.Zero)
    zeta=sp.Symbol('zeta',real=True)
    f=sp.Function('f_sw')(T)
    m2=action.mass_squared
    dm=f*m2/(32*sp.pi**2*zeta)
    dv=f*m2**2/(8*(16*sp.pi**2)**2*zeta**2)
    determinant={0:sp.S.One}
    if family=='scalar':
        determinant.update({1:-2*PS,2:-2*PS**2})
    else:
        determinant.update({1:sp.S.Zero,2:-SH/4})
    volume=a*a*dm*P**2/2+a**4*dv
    H={(1,j):sp.expand(J*volume) for j,J in determinant.items() if J!=0}
    # Normal-observer physical density contains no explicit metric factors
    # in the Phi² and unit-operator counterterms (at fixed Phi = phi/a).
    O={(1,0):dm*P**2/(2*a**2)+dv}
    return CountertermInsertions(H,O,dm,dv)
