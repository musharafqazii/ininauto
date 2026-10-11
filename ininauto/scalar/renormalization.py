"""Covariant proper-time subtraction *structure*, not a numerical UV regulator.

The heat-kernel pole coefficients and the renormalized coefficient identities
below match the supplied derivation.  A complete evaluation of the finite
state-dependent Hadamard/CTP kernels is NOT implemented. The pipeline never
passes bare Wick integrands off as finite observables.
"""
from dataclasses import dataclass
import sympy as s
from ..core.symbols import PS,SH,a

@dataclass(frozen=True)
class Renormalization:
    free_pole_density: s.Expr
    mass_counterterm: s.Expr
    vacuum_counterterm: s.Expr
    finite_coefficients: dict
    conditions: str


def prescription(action,family):
    zeta=s.Symbol('zeta',real=True)
    mu=s.Symbol('mu_R',positive=True)
    m2=action.mass_squared
    lam=action.coupling
    curvature=s.Symbol('A2_curvature') # (Riem^2-Ric^2+Box R)/180
    # Euclidean one-loop pole density in Gamma_0 (with sqrt(g_E) removed)
    free_pole=-(m2**2/s.Integer(2)+curvature)/(32*s.pi**2*zeta)
    # Counterterm mass squared with regulator and switch f_sw.
    switch=s.Function('f_sw')(s.Symbol('tau',real=True))
    dm2=lam*switch*m2/(32*s.pi**2*zeta)
    # This vacuum CT accompanies figure-eight + mass subdivergence.
    ct_vac=lam*switch*m2**2/(8*(16*s.pi**2)**2*zeta**2)
    d0,d1,d2=s.symbols('d0 d1 d2')
    U0,U1,U2=s.symbols('U0 U1 U2')
    M0,M1,M2=s.symbols('M0 M1 M2')
    if family=='scalar':
        b1=3*PS;b2=s.Rational(15,2)*PS**2
        j1=-2*PS;j2=-2*PS**2
    else:
        b1=0;b2=SH/4
        j1=0;j2=-SH/4
    out={
      (0,0):U0,
      (0,1):U1+b1*U0,
      (0,2):U2+b1*U1+b2*U0,
      (1,0):switch*d0**2/8+M0/(4*a**4),
      (1,1):switch*d0*d1/4+(M1+b1*M0)/(4*a**4),
      (1,2):switch*(d0*d2/4+d1**2/8)+(M2+b1*M1+b2*M0)/(4*a**4),
    }
    return Renormalization(free_pole,dm2,ct_vac,
        {k:s.expand(v) for k,v in out.items()},
        'Uj are the metric derivatives of the renormalized CTP free functional; '
        'dj are the FULL-metric Hadamard-subtracted <Phi^2> coefficients; '
        'Mj are causal renormalized metric-variance response integrals including '
        'diagonal contact terms. They are formal unless the state, geometry, '
        'initial preparation and regulator have been supplied and evaluated.')
