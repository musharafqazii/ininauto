"""Hilbert normal-observer energy density (not the canonical Hamiltonian density).

This uses the conformal improvement xi=1/6; rho=T_00/(-g_00).
For scalar geometry Q_F=(gamma_00 Box - nabla_0 nabla_0)phi^2.
Canonical substitution is performed after constructing the Hilbert observable.
"""
from dataclasses import dataclass
import sympy as s
from ..core.symbols import *
from ..core.geometry import scalar_geometry

@dataclass
class EnergyDensity:
    velocity: s.Expr
    canonical: s.Expr
    coefficients: dict


def hilbert_density(family,canonical,action):
    if family=='scalar':
        g=scalar_geometry()
        A=1+2*E*PS; B=1-2*E*PS
        Bp=-2*E*PST
        Fprime=2*P*V
        lapF=2*G+2*P*LAP
        gradBgradF=-4*E*P*MIX
        Q=s.Rational(3,2)*Bp/B*Fprime -A/B*lapF-A/(2*B**2)*gradBgradF
        rho=V**2/(2*A)+G/(2*B)+M*P**2/2 \
            +(g.einstein00*P**2+Q)/(6*A) +action.coupling*P**4/24
    elif family=='tensor':
        rho=(V**2+G+M*P**2)/2-(G+P*LAP)/3
        rho+=E*(-HG/s.Integer(6) +HESS/3)
        rho+=E**2*(CG/6 -CHESS/3-DIVC/3+HDER/6 \
              +(R3/s.Integer(12)-HP2/s.Integer(48))*P**2 -SHP*P*V/6)
        rho+=action.coupling*P**4/24
    else: raise ValueError(f'Unknown family: {family}')
    vel=trunc(rho/a**4)
    can=trunc(vel.subs(V,canonical.velocity))
    out={}
    for i in ((0,) if action.coupling==0 else (0,1)):
        part=s.expand(can) if action.coupling==0 else s.expand(can).coeff(action.coupling,i)
        for j in range(3):
            c=s.expand(part.coeff(E,j))
            if c!=0: out[(i,j)]=c
    return EnergyDensity(vel,can,out)
