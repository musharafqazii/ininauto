"""Actual local mode-by-mode subtraction for the homogeneous free sector.

This implements the standard fourth-adiabatic-order vacuum-energy subtraction
and second-adiabatic-order scalar variance subtraction for a conformally
coupled scalar in spatially flat FLRW. It is a *separate renormalization
scheme* from the source's covariant proper-time finite-part prescription.
It covers only lambda=0, h=0; it is NOT a subtraction for inhomogeneous
metric perturbations, quartic CTP diagrams, or their contact distributions.
"""
from dataclasses import dataclass
import sympy as s

@dataclass(frozen=True)
class AdiabaticSubtraction:
    omega: s.Expr
    W2: s.Expr
    variance_subtraction: s.Expr
    energy_subtraction: s.Expr
    mode_equation: s.Eq
    energy_finite_integral: s.Expr


def homogeneous_subtraction(mass_squared=None, scale_factor=None):
    """Returns symbolic renormalized energy-integral, given a(tau).

    The integrand is (|u'|²+omega²|u|²)/2 - E_ad^(0..4).
    Total density is int_0^infty dk k²/(2*pi²*a⁴) * integrand.
    All k-integrals and mode functions stay symbolic until background and
    normalized Hadamard state are supplied.
    """
    t,k=s.symbols('tau k',real=True,positive=True)
    a=scale_factor if scale_factor is not None else s.Function('a')(t)
    m2=mass_squared if mass_squared is not None else s.Symbol('m2',positive=True)
    w=s.Function('omega')(t)
    wp=s.diff(w,t)
    wpp=s.diff(w,t,2)
    W2=-wpp/(4*w**2)+3*wp**2/(8*w**3)
    W2p=s.diff(W2,t)
    energy=s.expand(w/2+wp**2/(16*w**3)+W2**2/(4*w)
        +wp*W2p/(8*w**3)-3*wp**2*W2/(16*w**4))
    variance=s.expand(1/(2*w)-W2/(2*w**2))
    subst={s.diff(w,t,n):s.diff(s.sqrt(k*k+m2*a*a),t,n)
           for n in (3,2,1)}
    subst[w]=s.sqrt(k*k+m2*a*a)
    u=s.Function('u')(k,t)
    up=s.Function('u_bar')(k,t)
    om=s.sqrt(k*k+m2*a*a)
    e_exact=(s.diff(u,t)*s.diff(up,t)+om**2*u*up)/2
    e_sub=energy.subs(subst, simultaneous=True)
    delta=s.Integral(k**2*(e_exact-e_sub)/(2*s.pi**2*a**4),(k,0,s.oo))
    return AdiabaticSubtraction(om,W2.subs(subst,simultaneous=True),
         variance.subs(subst,simultaneous=True),e_sub,
         s.Eq(s.diff(u,t,2)+om**2*u,0),delta)
