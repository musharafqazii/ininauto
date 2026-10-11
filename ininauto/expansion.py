"""Conformal action reduction and coefficientwise integration by parts."""
import sympy as s
from .symbols import *
from .geometry import scalar_geometry,tensor_geometry


def ibp_scalar(expr):
    """Remove psi_tt and Lap psi modulo integrals of total derivatives.

    Exact algebraic identity in the action integral for coefficients f(psi,phi):
    ∫f psi_tt = -∫(df/dpsi psi_t² + df/dphi psi_t phi_t), and similarly
    for the spatial Laplacian. Smooth compact preparation is assumed.
    """
    f=s.diff(expr,PSTT)
    expr=s.expand(expr.subs(PSTT,0)-s.diff(f,PS)*PST**2-s.diff(f,P)*PST*V)
    f=s.diff(expr,PSL)
    return s.expand(expr.subs(PSL,0)-s.diff(f,PS)*PSG-s.diff(f,P)*MIX)


def reduced_lagrangian(action,family='scalar'):
    g=scalar_geometry() if family=='scalar' else tensor_geometry()
    # M2=a^2 m^2; the coupling may depend on the user's chosen action symbol.
    lambda_ = action.coupling
    if family=='scalar':
        L= g.kinetic*V**2/2 - g.gradient_scalar*G/2 \
            -g.jacobian*M*P**2/2 - g.jacobian*g.ricci*P**2/12 \
            -g.jacobian*lambda_*P**4/24
        L=ibp_scalar(trunc(L))
    else:
        # The canonical tensor interaction carries h_ij d_i phi d_j phi.
        L=g.kinetic*V**2/2 -g.jacobian*(G+M*P**2)/2 + E*HG/2 \
          -E**2*CG/2 - E**2*(R3/12-SHPP/12-HP2/16)*P**2 \
          -g.jacobian*lambda_*P**4/24
    return trunc(L)
