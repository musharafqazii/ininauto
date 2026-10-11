"""Action -> numerical engine bridge.

The numerical back end (``ininauto.numerics.engine``) integrates the retarded mode hierarchy
F_k = u + eps v + eps^2 z + lambda (l + eps w + eps^2 r) and evaluates <rho> as the velocity bilinear
sum_k E(F_k, F_k) (equivalent to the canonical in-in series, see the reference derivation).  Its kernels are
written out explicitly for speed.  This module re-derives every one of them from the *input action* and the
symbolic pipeline of this package, and refuses to run the numerics if anything differs:

* mode operators  D_0 + eps P_1 + eps^2 P_2  from the Euler-Lagrange equation of the reduced Lagrangian
  (explicit fields, sympy.euler_equations), projected exactly as the engine uses them
  (first order: side band p -> p+q; second order: zero mode of psi^2 / h^2, ensemble mean);
* the normal-observer energy density in velocity variables, orders eps^0, eps^1, eps^2;
* the quartic vertices and their determinant factors j_1, j_2 (Hartree tadpoles Sigma_1, Sigma_2);
* m^2 and the conformal coupling, read from the action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sympy as s

from .action import ScalarAction
from .expansion import reduced_lagrangian
from .hamiltonian import legendre
from .stress import hilbert_density
from .symbols import (E, T as TAU, M, a, P, V, G, LAP, PS, PST, PSG, PSL, MIX, SH, SHP, HP2, R3, HG, CG,
                      HESS, CHESS, DIVC, HDER, SHPP)


@dataclass
class BridgeReport:
    family: str
    checks: dict = field(default_factory=dict)

    @property
    def ok(self):
        return all(v == 0 for v in self.checks.values())

    def __str__(self):
        lines = [f"[{self.family}] action -> engine kernels: {'ALL MATCH' if self.ok else 'MISMATCH'}"]
        for k, v in self.checks.items():
            lines.append(f"   {k:45s} {'ok' if v == 0 else 'residual: ' + str(v)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- explicit fields
t, x, y, z = s.symbols('t x y z', real=True)
X3 = (x, y, z)
_phi = s.Function('phi')(t, x, y, z)
_psi = s.Function('psi')(t, x, y, z)
_hp = s.Function('hp')(t, z)                       # h_xx = -h_yy = hp(t, z): a transverse-traceless wave along z
_M2 = s.Function('M2')(t)
_lam = s.Symbol('lam')


def _grad(f): return [s.diff(f, c) for c in X3]
def _dot(u, v): return sum(i*j for i, j in zip(u, v))


def _explicit_scalar(expr):
    rep = {V: s.diff(_phi, t), P: _phi, G: _dot(_grad(_phi), _grad(_phi)), MIX: _dot(_grad(_psi), _grad(_phi)),
           PS: _psi, PST: s.diff(_psi, t), PSG: _dot(_grad(_psi), _grad(_psi)), PSL: sum(s.diff(_psi, c, 2) for c in X3),
           LAP: sum(s.diff(_phi, c, 2) for c in X3)}
    return expr.subs(M, _M2).subs(TAU, t).subs(rep, simultaneous=True)


def _hmat():
    return s.Matrix([[_hp, 0, 0], [0, -_hp, 0], [0, 0, 0]])


def _explicit_tensor(expr):
    h = _hmat(); C = h*h
    gp = _grad(_phi)
    hdt = h.diff(t); hdtt = h.diff(t, 2)
    sh = sum(h[i, j]**2 for i in range(3) for j in range(3))
    r3 = (sum(h[i, j]*sum(s.diff(h[i, j], c, 2) for c in X3) for i in range(3) for j in range(3))
          + s.Rational(3, 4)*sum(s.diff(h[i, j], X3[k])**2 for i in range(3) for j in range(3) for k in range(3))
          - s.Rational(1, 2)*sum(s.diff(h[i, k], X3[j])*s.diff(h[i, j], X3[k]) for i in range(3) for j in range(3) for k in range(3)))
    rep = {V: s.diff(_phi, t), P: _phi, G: _dot(gp, gp), LAP: sum(s.diff(_phi, c, 2) for c in X3),
           SH: sh, SHP: sum(h[i, j]*hdt[i, j] for i in range(3) for j in range(3)),
           SHPP: sum(h[i, j]*hdtt[i, j] for i in range(3) for j in range(3)),
           HP2: sum(hdt[i, j]**2 for i in range(3) for j in range(3)), R3: r3,
           HG: sum(h[i, j]*gp[i]*gp[j] for i in range(3) for j in range(3)),
           CG: sum(C[i, j]*gp[i]*gp[j] for i in range(3) for j in range(3)),
           HESS: sum(h[i, j]*_phi*s.diff(_phi, X3[i], X3[j]) for i in range(3) for j in range(3)),
           CHESS: sum(C[i, j]*_phi*s.diff(_phi, X3[i], X3[j]) for i in range(3) for j in range(3)),
           DIVC: sum(_phi*h[i, k]*s.diff(h[k, j], X3[i])*gp[j] for i in range(3) for j in range(3) for k in range(3)),
           HDER: sum(_phi*h[j, k]*s.diff(h[j, k], X3[i])*gp[i] for i in range(3) for j in range(3) for k in range(3))}
    return expr.subs(M, _M2).subs(TAU, t).subs(rep, simultaneous=True)


def _el(lagr):
    """Euler-Lagrange expression dL/dphi - d_mu dL/d(d_mu phi), for an explicit Lagrangian density."""
    eq = s.euler_equations(lagr, [_phi], [t, x, y, z])[0]
    return s.expand(eq.lhs)


# --------------------------------------------------------------------------- engine kernels (as implemented)
def _engine_scalar(q, p, f, T):
    """Engine's D1 (side band) and D2 (zero mode, mean <psi^2> = T^2), acting on f(t) e^{ipx}; 'f2' = f''."""
    fp, fpp = s.diff(f, t), s.diff(f, t, 2)
    Tp, Tpp = s.diff(T, t), s.diff(T, t, 2)
    D1 = -4*T*fpp - 4*Tp*fp - (Tpp + 2*_M2*T + q**2*T/3)*f
    D2 = 6*T**2*fpp + 12*T*Tp*fp + (2*T*Tpp + Tp**2 - q**2*T**2/3 - 2*_M2*T**2 - 2*p**2*T**2)*f
    return D1, D2


def _engine_tensor(q, p2, f, T):
    """Engine's D2 for the tensor (zero mode, isotropic mean, <S_h> = T^2), flat conformal frame."""
    fp, fpp = s.diff(f, t), s.diff(f, t, 2)
    Tp, Tpp = s.diff(T, t), s.diff(T, t, 2)
    src = fpp + (p2 + _M2)*f
    return -T**2/4*src - T*Tp/2*fp + p2*T**2/3*f + (-q**2*T**2/24 - T*Tpp/6 - Tp**2/8)*f


def verify(action: ScalarAction, family: str) -> BridgeReport:
    rep = BridgeReport(family)
    L = reduced_lagrangian(action, family)
    can = legendre(L, action.coupling)
    rho = hilbert_density(family, can, action)
    lam = action.coupling
    # ---- conformal coupling and mass come from the action
    rep.checks['xi = 1/6'] = s.simplify(action.xi - s.Rational(1, 6))
    # ---- quartic vertices: H_I^(lambda eps^j) = j_j phi^4/24 with the engine's determinant factors
    jexp = {('scalar', 1): -2*PS, ('scalar', 2): -2*PS**2, ('tensor', 1): 0, ('tensor', 2): -SH/4}
    if lam != 0:
        rep.checks['quartic vertex A = phi^4/24'] = s.expand(can.interaction_coeffs.get((1, 0), 0) - P**4/24)
        for j in (1, 2):
            rep.checks[f'quartic vertex j_{j} (Hartree Sigma_{j})'] = s.expand(
                can.interaction_coeffs.get((1, j), 0) - jexp[(family, j)]*P**4/24)
    vel = s.expand(rho.velocity*a**4)
    # ---- velocity energy density (what the engine's bilinears E_0, E_1, E_2 polarise)
    if family == 'scalar':
        eng = {0: V**2/2 + G/6 - P*LAP/3 + M*P**2/2,
               1: -PS*V**2 - PST*P*V + PS*G - PS*(2*G + 2*P*LAP)/3 + P**2*PSL/3 + P*MIX/3,
               2: 2*PS**2*V**2 + s.Rational(2, 3)*PS**2*G - s.Rational(4, 3)*PS**2*P*LAP + s.Rational(4, 3)*PS*P**2*PSL
                  + s.Rational(4, 3)*PS*P*MIX + P**2*(PST**2 + PSG)/2}
    else:
        eng = {0: V**2/2 + G/6 - P*LAP/3 + M*P**2/2,
               1: -HG/2 + (2*HG + 2*HESS)/6,
               2: CG/2 + (R3/12 - HP2/48)*P**2 - SHP*P*V/6 - (2*DIVC + 2*CG + 2*CHESS)/6 + HDER/6}
    for j in range(3):
        got = vel.coeff(E, j).subs(lam, 0) if lam != 0 else vel.coeff(E, j)
        rep.checks[f'energy density rho_{j} (velocity form)'] = s.expand(got - eng[j])
    if lam != 0:
        rep.checks['energy density quartic term lambda phi^4/24'] = s.expand(vel.coeff(lam) - P**4/24)
    # ---- mode operators from the Euler-Lagrange equation of the action
    Lfree = L.subs(lam, 0) if lam != 0 else L
    q, p, T0 = s.symbols('q p T0', positive=True)
    f = s.Function('f')(t); T = s.Function('T')(t)
    if family == 'scalar':
        EL = _el(_explicit_scalar(Lfree))
        # first order: psi = T e^{iqx}, phi = f e^{ipx}; the engine evolves the e^{i(p+q)x} side band
        e1 = EL.coeff(E, 1).subs({_psi: T*s.exp(s.I*q*x), _phi: f*s.exp(s.I*p*y)}).doit()
        P1 = -s.simplify(e1*s.exp(-s.I*(q*x + p*y)))
        D1, D2 = _engine_scalar(q, p, f, T)
        rep.checks['mode operator P_1 (side band)'] = s.simplify(s.expand(P1 - D1))
        # second order: psi = sqrt2 T cos(qx) -> zero mode of psi^2 terms acting on f e^{ipy}
        e2 = EL.coeff(E, 2).subs({_psi: s.sqrt(2)*T*s.cos(q*x), _phi: f*s.exp(s.I*p*y)}).doit()
        e2 = s.expand(e2*s.exp(-s.I*p*y))
        zero = s.integrate(e2, (x, 0, 2*s.pi/q))*q/(2*s.pi)
        rep.checks['mode operator P_2 (zero mode, mean)'] = s.simplify(s.expand(-zero - D2))
        # D_0
        e0 = EL.coeff(E, 0).subs({_phi: f*s.exp(s.I*p*y)}).doit()
        rep.checks['free operator D_0'] = s.simplify(s.expand(-e0*s.exp(-s.I*p*y)
                                                               - (s.diff(f, t, 2) + (p**2 + _M2)*f)))
    else:
        EL = _el(_explicit_tensor(Lfree))
        # first order: h_xx = -h_yy = T e^{iqz}; phi = f e^{i(px x + py y + pz z)}: P_1 phi = -h_ij p_i p_j phi
        px, py, pz = s.symbols('p_x p_y p_z', real=True)
        ph = f*s.exp(s.I*(px*x + py*y + pz*z))
        e1 = EL.coeff(E, 1).subs({_hp: T*s.exp(s.I*q*z), _phi: ph}).doit()
        P1 = -s.simplify(e1*s.exp(-s.I*(px*x + py*y + (pz + q)*z)))
        rep.checks['mode operator P_1 = -h_ij p_i p_j'] = s.simplify(s.expand(P1 + T*(px**2 - py**2)*f))
        # second order, zero mode: h = T cos(qz) (<S_h> = T^2), then isotropic average over the direction of p
        e2 = EL.coeff(E, 2).subs({_hp: T*s.cos(q*z), _phi: ph}).doit()
        e2 = s.expand(e2*s.exp(-s.I*(px*x + py*y + pz*z)))
        zero = s.integrate(e2, (z, 0, 2*s.pi/q))*q/(2*s.pi)
        th, ph_ = s.symbols('theta varphi', real=True)
        pp = s.Symbol('P', positive=True)
        sub = {px: pp*s.sin(th)*s.cos(ph_), py: pp*s.sin(th)*s.sin(ph_), pz: pp*s.cos(th)}
        iso = s.integrate(s.integrate(s.expand(-zero).subs(sub)*s.sin(th), (ph_, 0, 2*s.pi)), (th, 0, s.pi))/(4*s.pi)
        rep.checks['mode operator P_2 (zero mode, isotropic mean)'] = s.simplify(s.expand(iso - _engine_tensor(q, pp**2, f, T)))
    return rep


def verify_all(action, families=('scalar', 'tensor')):
    act = action if isinstance(action, ScalarAction) else ScalarAction.parse(action)
    return {f: verify(act, f) for f in families}
