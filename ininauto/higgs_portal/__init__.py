"""Higgs portal: conformal dark scalar Phi + Higgs-like scalar h coupled by V_portal = (lambda_p/4) h^2 Phi^2.

Input is the covariant action only, e.g.

    S = -KIN/2 - m^2 PHI2/2 - RPHI2/12 - lam PHI4/24                 (dark field Phi, xi = 1/6)
        -KINH/2 - m_h^2 H2/2 - RH2/12 - lam_h H4/24                  (Higgs field h,  xi = 1/6)
        -lam_p H2PHI2/4                                              (portal)

Both fields are taken conformally coupled so that both live in the conformal frame of ininauto:
phi = a Phi, chi = a h, M^2 = a^2 m^2, M_h^2 = a^2 m_h^2.  Since sqrt(-g) h^2 Phi^2 = sqrt(-gamma) chi^2 phi^2,
the portal is

    L_portal = -sqrt(-gamma) (lambda_p/4) phi^2 chi^2,   H_portal = +sqrt(-gamma) (lambda_p/4) phi^2 chi^2,
    rho_portal = (lambda_p/4) phi^2 chi^2 / a^4          (rho = T_00/(-g_00) contains +V, no metric factor).

Everything else (metric expansion to O(eps^2), Legendre transform, Hilbert energy density, Weyl ordering, Wick
contractions) is the single-field machinery applied to each field; the Higgs-sector jets are the dark-sector jets
with phi -> chi.  Coefficients are keyed (l, p, j): l = power of the self-couplings (lambda, lambda_h), p = power
of lambda_p, j = power of the metric perturbation eps.

Modules: action (building blocks, PortalAction), fields (chi jets, M_h^2), system (portal_system, in-in terms, checks).

    sys = portal_system(S, "scalar")     # the package itself is also callable: higgs_portal(S, "scalar")
"""
from .action import KINH, H2, RH2, H4, H2PHI2, PortalAction, portal_action
from .fields import MH, MH2, PC, VC, PIC, GC, LAPC, TO_CHI, to_chi
from .system import PortalSystem, portal_system, portal_inin, portal_diagrams, checks

__all__ = ["KINH", "H2", "RH2", "H4", "H2PHI2", "PortalAction", "portal_action", "MH", "MH2", "PC", "VC", "PIC",
           "GC", "LAPC", "TO_CHI", "to_chi", "PortalSystem", "portal_system", "portal_inin", "portal_diagrams", "checks"]


import sys as _sys
import types as _types


class _CallableModule(_types.ModuleType):
    """higgs_portal(S, family) == portal_system(S, family)  (keeps the 0.6 call style working)."""

    def __call__(self, action, family="scalar"):
        return portal_system(action, family)


_sys.modules[__name__].__class__ = _CallableModule
