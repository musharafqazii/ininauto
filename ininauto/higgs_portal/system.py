"""Canonical two-field system (Lagrangian, Hamiltonian, energy density) and its in-in Wick terms."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as s

from ..core.symbols import *                 # noqa: F401,F403
from ..core.geometry import scalar_geometry, tensor_geometry
from ..core.wick import FieldAtom, GaussianState, wick_vev, nested_commutator_vev
from ..scalar.expansion import reduced_lagrangian
from ..scalar.hamiltonian import legendre
from ..scalar.stress import hilbert_density
from ..scalar.operators import to_local, to_vertex, to_polynomial
from .fields import TO_CHI, _PHI_JETS, PC, VC, PIC, to_chi
from .action import PortalAction


# ------------------------------------------------------------------------------------------- canonical system
@dataclass
class PortalSystem:
    family: str
    action: PortalAction
    lagrangian: s.Expr          # conformal-frame Lagrangian density in jets, through eps^2
    velocities: dict            # {phi': f(pi, ...), chi': g(pi_chi, ...)}
    hamiltonian: s.Expr         # full canonical Hamiltonian density
    H: dict                     # (l, p, j) -> Hamiltonian coefficient ((0,0,0) is the free reference H_0)
    rho_velocity: s.Expr        # rho in velocity variables (phi', chi')
    rho_canonical: s.Expr
    rho: dict                   # (l, p, j) -> coefficient of rho

    @property
    def couplings(self):
        return self.action.dark.coupling, self.action.higgs.coupling, self.action.portal

    def vertices(self):
        """Interaction vertices (everything but the free H_0)."""
        return {k: v for k, v in self.H.items() if k != (0, 0, 0)}


def _split(expr, lam, lam_h, lam_p):
    out = {}
    X = s.expand(expr)
    for p in (0, 1):
        if lam_p == 0:
            Xp = X if p == 0 else s.S.Zero
        else:
            Xp = s.expand(X.coeff(lam_p, p))
        zero = {g: 0 for g in (lam, lam_h) if g != 0}
        X0 = s.expand(Xp.subs(zero)) if zero else Xp
        for l, Y in ((0, X0), (1, s.expand(Xp - X0))):
            for j in range(3):
                c = s.expand(Y.coeff(E, j))
                if c != 0:
                    out[(l, p, j)] = c
    return out


def portal_system(action, family="scalar"):
    """Lagrangian, Hamiltonian and energy density of the two-field Higgs-portal system on perturbed FLRW
    (family 'scalar' = Newtonian gauge, 'tensor' = TT), through O(eps^2) and first order in each coupling."""
    A = action if isinstance(action, PortalAction) else PortalAction.parse(action)
    g = scalar_geometry() if family == "scalar" else tensor_geometry()
    lam, lam_h, lam_p = A.dark.coupling, A.higgs.coupling, A.portal

    L_dark = reduced_lagrangian(A.dark, family)
    L_h1 = reduced_lagrangian(A.higgs, family)              # written in phi-jets, renamed below
    L = trunc(L_dark + to_chi(L_h1) - g.jacobian * lam_p * P**2 * PC**2 / 4)

    sol = {}
    for vel, mom in ((V, PI), (VC, PIC)):
        pm = s.expand(s.diff(L, vel))
        C = s.diff(pm, vel)
        J = s.expand(pm - C * vel)
        if any(x.has(V) or x.has(VC) for x in (C, J)):
            raise ValueError("velocity mixing between the two fields is not supported")
        sol[vel] = trunc((mom - J) / C)
    Hfull = trunc(PI * sol[V] + PIC * sol[VC] - L.subs(sol, simultaneous=True))

    rho_d = hilbert_density(family, legendre(L_dark, lam), A.dark).velocity
    rho_h = to_chi(hilbert_density(family, legendre(L_h1, lam_h), A.higgs).velocity)
    rho_v = s.expand(rho_d + rho_h + lam_p * P**2 * PC**2 / (4 * a**4))
    rho_c = trunc(rho_v.subs(sol, simultaneous=True))
    return PortalSystem(family, A, L, sol, Hfull, _split(Hfull, lam, lam_h, lam_p), rho_v, rho_c,
                        _split(rho_c, lam, lam_h, lam_p))


# ------------------------------------------------------------------------------------------- in-in terms
def _legs(expr):
    poly = to_polynomial(expr)
    return max((sum(isinstance(x, FieldAtom) for x in m.atoms) for m in poly.monos), default=0)


def portal_inin(system: PortalSystem, observable, vertices=(), *, state=None):
    """(-i)^n < [[...[rho_obs(x, tau), H_v1(t1)], ...], H_vn(tn)] >  (tau >= t1 >= ... >= tn), unintegrated in time.

    observable, vertices: keys (l, p, j) of system.rho / system.H.  Two species: phi (dark) and chi (Higgs),
    mode functions u_k and u^{(chi)}_k."""
    state = state or GaussianState()
    o = to_local(system.rho[observable], T, "x")
    if not vertices:
        return wick_vev([o], state=state)
    times = [s.Symbol(f"t{i}", real=True) for i in range(1, len(vertices) + 1)]
    hs = [to_vertex(system.H[v].subs(T, t), t) for v, t in zip(vertices, times)]
    return (-s.I) ** len(vertices) * nested_commutator_vev([o] + hs, connected=True, state=state)


def portal_diagrams(system: PortalSystem, target=(0, 1, 0), max_insertions=2, max_legs=6):
    """All in-in contributions of degree `target` = (l, p, j); returns a list of (label, WickResult or None).
    Contributions with more than max_legs quantum legs are listed but not evaluated (None); max_legs=8 also
    evaluates the double insertions (about 2 min each)."""
    V = system.vertices()
    out = []
    for n in range(max_insertions + 1):
        for od in system.rho:
            for seq in product(V, repeat=n):
                deg = tuple(od[i] + sum(v[i] for v in seq) for i in range(3))
                if deg != target:
                    continue
                label = "rho" + "".join(map(str, od)) + "".join(f" H{''.join(map(str, v))}(t{i})"
                                                               for i, v in enumerate(seq, 1))
                legs = _legs(system.rho[od]) + sum(_legs(V[v]) for v in seq)
                res = portal_inin(system, od, seq) if (max_legs is None or legs <= max_legs) else None
                out.append((label, res))
    return out


def checks(system: PortalSystem):
    """Fast consistency checks of the portal sector."""
    lam, lam_h, lam_p = system.couplings
    g = scalar_geometry() if system.family == "scalar" else tensor_geometry()
    Lp = s.expand(system.lagrangian.coeff(lam_p, 1))
    Hp = s.expand(system.hamiltonian.coeff(lam_p, 1))
    rp = s.expand(system.rho_canonical.coeff(lam_p, 1))
    out = [
        ("H_portal = -L_portal (no derivative coupling)", s.expand(Hp + Lp) == 0),
        ("H_portal = sqrt(-gamma) phi^2 chi^2 / 4", s.expand(Hp - trunc(g.jacobian * P**2 * PC**2 / 4)) == 0),
        ("rho_portal = phi^2 chi^2 / (4 a^4)", s.expand(rp - P**2 * PC**2 / (4 * a**4)) == 0),
    ]
    A0 = system.action.dark
    dark_H = legendre(reduced_lagrangian(A0, system.family), A0.coupling).full_density
    Hd = s.expand(system.hamiltonian.subs(lam_p, 0).xreplace({x: 0 for x in TO_CHI.values()}))
    out.append(("lambda_p = 0, chi = 0: dark sector equals the single-field result", s.expand(Hd - dark_H) == 0))
    A1 = system.action.higgs
    higgs_H = to_chi(legendre(reduced_lagrangian(A1, system.family), A1.coupling).full_density)
    Hh = s.expand(system.hamiltonian.subs(lam_p, 0).xreplace({x: 0 for x in _PHI_JETS}))
    out.append(("lambda_p = 0, phi = 0: Higgs sector equals the renamed single-field result",
                s.expand(Hh - higgs_H) == 0))
    return out
