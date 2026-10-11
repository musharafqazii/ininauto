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
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as s
from sympy.printing.pretty.stringpict import prettyForm

from .symbols import *                      # noqa: F401,F403  (jets, E, M, T, a, trunc)
from .symbols import TexSymbol, _TEX, _UNI
from .action import KIN, PHI2, RPHI2, PHI4, ScalarAction
from .geometry import scalar_geometry, tensor_geometry
from .expansion import reduced_lagrangian
from .hamiltonian import legendre
from .stress import hilbert_density
from .operators import REGISTRY, to_local, to_vertex, to_polynomial
from .wick import Poly, Monomial, FieldAtom, GaussianState, wick_vev, nested_commutator_vev, WickResult


# ------------------------------------------------------------------------------------------- Higgs-sector symbols
class MH2(s.Function):
    """M_h^2(tau) = a(tau)^2 m_h^2, printed M_h^2(tau)."""
    is_real = True

    def _latex(self, printer, exp=None):
        base = r"M_{h}^{2}\!\left(" + printer._print(self.args[0]) + r"\right)"
        return base if exp is None else r"\left(" + base + r"\right)^{" + str(exp) + "}"

    def _pretty(self, printer):
        return prettyForm("M_h²(" + str(printer._print(self.args[0])) + ")")


MH = MH2(T)

_PHI_JETS = [P, V, PI, G, LAP, MIX, HG, CG, HESS, CHESS, DIVC, HDER]
_SPECIAL = {"phi": ("chi", r"\chi", "χ"), "v": ("v_chi", r"\chi'", "χ'"), "pi": ("pi_chi", r"\pi_{\chi}", "π_χ")}


def _chi_jet(sym):
    if sym.name in _SPECIAL:
        name, tex, uni = _SPECIAL[sym.name]
    else:
        name = sym.name.replace("phi", "chi")
        tex = _TEX[sym.name].replace(r"\phi", r"\chi")
        uni = _UNI[sym.name].replace("φ", "χ")
    return TexSymbol(name, tex=tex, uni=uni, real=True)


TO_CHI = {j: _chi_jet(j) for j in _PHI_JETS}
PC, VC, PIC, GC, LAPC = (TO_CHI[x] for x in (P, V, PI, G, LAP))


def _rename_poly(poly, old="phi", new="chi"):
    return Poly(tuple(Monomial(m.coefficient,
                               tuple(FieldAtom(new, at.dt, at.dx, at.indices)
                                     if isinstance(at, FieldAtom) and at.name == old else at for at in m.atoms))
                      for m in poly.monos))


REGISTRY.update({TO_CHI[j]: _rename_poly(REGISTRY[j]) for j in _PHI_JETS})


def to_chi(expr):
    """Rename a single-field (dark-sector) jet expression to the Higgs sector: phi -> chi, M^2 -> M_h^2."""
    return s.sympify(expr).xreplace({**TO_CHI, M: MH})


# ------------------------------------------------------------------------------------------- action language
KINH = TexSymbol("KINH", tex=r"\left(\nabla h\right)^{2}", uni="(∇h)²")
H2 = TexSymbol("H2", tex=r"h^{2}", uni="h²")
RH2 = TexSymbol("RH2", tex=r"R\,h^{2}", uni="Rh²")
H4 = TexSymbol("H4", tex=r"h^{4}", uni="h⁴")
H2PHI2 = TexSymbol("H2PHI2", tex=r"h^{2}\Phi^{2}", uni="h²Φ²")

_BLOCKS = {"KIN": KIN, "PHI2": PHI2, "RPHI2": RPHI2, "PHI4": PHI4,
           "KINH": KINH, "H2": H2, "RH2": RH2, "H4": H4, "H2PHI2": H2PHI2}


@dataclass(frozen=True)
class PortalAction:
    expression: s.Expr
    dark: ScalarAction          # Phi sector (mass_squared, xi, coupling = lambda)
    higgs: ScalarAction         # h sector, written with the dark building blocks (coupling = lambda_h)
    portal: s.Expr              # lambda_p

    @classmethod
    def parse(cls, expression):
        e = s.sympify(expression, locals=dict(_BLOCKS))
        e = s.expand(e.xreplace({x: _BLOCKS[x.name] for x in e.free_symbols
                                 if x.name in _BLOCKS and not isinstance(x, TexSymbol)}))
        c = {k: e.coeff(b) for k, b in _BLOCKS.items()}
        rest = s.simplify(e - sum(c[k] * b for k, b in _BLOCKS.items()))
        if rest != 0:
            raise ValueError(f"Action contains unsupported operators: {rest}")
        dark = ScalarAction.parse(c["KIN"] * KIN + c["PHI2"] * PHI2 + c["RPHI2"] * RPHI2 + c["PHI4"] * PHI4)
        higgs = ScalarAction.parse(c["KINH"] * KIN + c["H2"] * PHI2 + c["RH2"] * RPHI2 + c["H4"] * PHI4)
        lam_p = s.expand(-4 * c["H2PHI2"])
        for name, g in (("lambda", dark.coupling), ("lambda_h", higgs.coupling), ("lambda_p", lam_p)):
            if g != 0 and not g.is_Symbol:
                raise ValueError(f"{name} must be zero or a single symbol")
        return cls(e, dark, higgs, lam_p)


def portal_action(m=None, m_h=None, lam=None, lam_h=0, lam_p=None):
    """-KIN/2 - m^2 PHI2/2 - RPHI2/12 - lam PHI4/24 - KINH/2 - m_h^2 H2/2 - RH2/12 - lam_h H4/24 - lam_p H2PHI2/4."""
    m = s.Symbol("m", positive=True) if m is None else s.sympify(m)
    m_h = s.Symbol("m_h", positive=True) if m_h is None else s.sympify(m_h)
    lam = s.Symbol("lambda", real=True) if lam is None else s.sympify(lam)
    lam_p = s.Symbol("lambda_p", real=True) if lam_p is None else s.sympify(lam_p)
    half = s.Rational(1, 2)
    return (-half * KIN - m**2 * PHI2 / 2 - RPHI2 / 12 - lam * PHI4 / 24
            - half * KINH - m_h**2 * H2 / 2 - RH2 / 12 - s.sympify(lam_h) * H4 / 24 - lam_p * H2PHI2 / 4)


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


def higgs_portal(action, family="scalar"):
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
