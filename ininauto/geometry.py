"""Prescribed metric families and their curvature invariants through O(eps**2).

The scalar curvature is calculated from the ADM scalar metric. Tensor formulas
assume transverse traceless h_ij. No second-order independent metric perturbation.
"""
from dataclasses import dataclass
import sympy as s
from .symbols import *

@dataclass
class Geometry:
    family: str
    jacobian: s.Expr
    kinetic: s.Expr
    gradient_scalar: s.Expr
    ricci: s.Expr
    einstein00: s.Expr
    spatial_curvature: s.Expr
    component_h_gradient: s.Expr = s.S.Zero
    component_h2_gradient: s.Expr = s.S.Zero


def scalar_geometry():
    A=1+2*E*PS; B=1-2*E*PS
    j=s.sqrt(A*B**3)
    ct=s.sqrt(B**3/A)
    cg=s.sqrt(A*B)
    Ap=2*E*PST; Bp=-2*E*PST
    Bpp=-2*E*PSTT
    lapA=2*E*PSL
    lapB=-2*E*PSL
    gradA2=4*E**2*PSG
    gradB2=4*E**2*PSG
    gradAB=-4*E**2*PSG
    R3=-2*lapB/B**2+s.Rational(3,2)*gradB2/B**3
    R=R3+3*Bpp/(A*B)-s.Rational(3,2)*Bp*Ap/(A**2*B) \
      -lapA/(A*B)+gradA2/(2*A**2*B)-gradAB/(2*A*B**2)
    G00=A*R3/2+3*Bp**2/(4*B**2)
    return Geometry('scalar',trunc(j),trunc(ct),trunc(cg),trunc(R),trunc(G00),trunc(R3))


def tensor_geometry():
    # Ricci_{3}^{(2)} derived from TT Christoffels, not a gauge-independent identity.
    j=1-E**2*SH/4
    ricci=E**2*(R3-SHPP-s.Rational(3,4)*HP2)
    # G00 of the *conformal* metric gamma, without the FLRW Hubble terms.
    G00=E**2*(R3/2-HP2/8)
    return Geometry('tensor',j,j,1-E**2*SH/4,ricci,G00,E**2*R3,
                    E*HG,E**2*CG)
