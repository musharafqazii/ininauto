"""Explicit canonical Legendre transform, never H_I = -L_I for velocity terms."""
from dataclasses import dataclass
import sympy as s
from ..core.symbols import *

@dataclass
class Canonical:
    momentum: s.Expr
    velocity: s.Expr
    full_density: s.Expr
    free_density: s.Expr
    interaction_coeffs: dict


def legendre(lagrangian, coupling):
    mom=s.expand(s.diff(lagrangian,V))
    C=s.diff(mom,V); J=s.expand(mom-C*V)
    v=trunc((PI-J)/C)
    H=trunc(PI*v-lagrangian.subs(V,v))
    base=s.expand(H.subs(E,0).subs(coupling,0)) if coupling!=0 else s.expand(H.subs(E,0))
    terms={}
    # H_{lambda^i eps^j}, with coupling symbol extracted; physical couplings removed.
    expanded=s.expand(H)
    if coupling!=0 and not coupling.is_Symbol:
        raise ValueError('quartic coupling must be zero or a single symbolic parameter')
    for i in ((0,) if coupling==0 else (0,1)):
        component=expanded if coupling==0 else s.expand(expanded.coeff(coupling,i))
        for j in range(3):
            c=s.expand(component.coeff(E,j))
            if c!=0:
                terms[(i,j)]=c
    return Canonical(mom,v,H,base,terms)
