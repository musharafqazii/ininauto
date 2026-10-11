"""The compact Wick form: every rewriting used by simplify_result is checked numerically, pointwise."""
import cmath
import random

import numpy as np
import sympy as sp

from ininauto import KIN, PHI2, RPHI2, PHI4
from ininauto.action import ScalarAction
from ininauto.expansion import reduced_lagrangian
from ininauto.hamiltonian import legendre
from ininauto.stress import hilbert_density
from ininauto.operators import to_local, to_vertex
from ininauto.symbols import T, M, a
from ininauto import wick as W

k1, q1 = sp.Symbol('k1'), sp.Symbol('q1')


def _setup():
    m, lam = sp.symbols('m lambda', real=True)
    A = ScalarAction.parse(-KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24)
    H = legendre(reduced_lagrangian(A, 'scalar'), A.coupling)
    rho = hilbert_density('scalar', H, A)
    t1 = sp.Symbol('t1', real=True)
    R0 = to_local(rho.coefficients[(0, 0)], T, 'x')
    H1 = to_vertex(H.interaction_coeffs[(0, 1)].subs(T, t1), t1)
    return -sp.I * W.nested_commutator_vev([R0, H1], connected=True, state=W.GaussianState()), t1


def _num(expr, K, Q, times):
    """Evaluate an integrand at numeric momenta with smooth test functions of |p| (isotropic modes) and a source
    with psi~(-q) = psi~(q)^*."""
    vec = {k1: np.array(K), q1: np.array(Q)}

    def v(p):
        p = sp.expand(p)
        return sum(float(c) * vec[s] for s, c in p.as_coefficients_dict().items())

    def mode(n, P, t, conj):
        p = np.linalg.norm(P)
        f = lambda tt: cmath.exp(-p * p) * (1 + p) * cmath.exp(-1j * (1 + p) * tt)
        val = (-1j * (1 + p)) ** int(n) * f(times[t])
        return val.conjugate() if conj else val

    def src(n, Q_, t):
        base = cmath.exp(-np.dot(Q_, Q_)) * (1 + 1j * Q_[2])
        tt = times[t]
        return base * [cmath.cos(tt), -cmath.sin(tt), -cmath.cos(tt)][int(n)]

    e = expr.replace(W.U, lambda s_, p, n, t: sp.sympify(mode(n, v(p), t, False)))
    e = e.replace(W.Ub, lambda s_, p, n, t: sp.sympify(mode(n, v(p), t, True)))
    e = e.replace(W.SP, lambda p, q: sp.sympify(float(np.dot(v(p), v(q)))))
    e = e.replace(W.SQ, lambda lab, q, n, t, *r: sp.sympify(src(n, v(q), t)))
    e = e.replace(M.func, lambda t: 1 + times[t] ** 2).replace(a.func, lambda t: 1 + 0.1 * times[t])
    return complex(sp.N(e))


def test_relabelling_is_a_change_of_variables():
    res, t1 = _setup()
    times = {T: 0.7, t1: 0.3}
    rng = random.Random(3)
    terms = sp.Add.make_args(res.terms[0].integrand)[:12]
    for m_ in terms:
        for rule in ({k1: -k1}, {k1: -k1 - q1}, {k1: k1 + 2 * q1}):
            K = [rng.uniform(-1, 1) for _ in range(3)]
            Q = [rng.uniform(-1, 1) for _ in range(3)]
            # relabelled integrand at k' equals the original at k = rule(k')
            Kp = np.array(K)
            Korig = (np.array(K) * float(rule[k1].coeff(k1)) + float(rule[k1].coeff(q1)) * np.array(Q))
            lhs = _num(W._relabel(m_, rule), Kp, Q, times)
            rhs = _num(m_, Korig, Q, times)
            assert abs(lhs - rhs) < 1e-10 * (1 + abs(rhs))


def test_conjugation_rule_for_real_sources():
    res, t1 = _setup()
    times = {T: 0.7, t1: 0.3}
    m_ = sp.Add(*sp.Add.make_args(res.terms[0].integrand)[:6])
    K, Q = [0.3, -0.2, 0.5], [0.1, 0.4, -0.6]
    # conj(B(k, q)) = conj_expr(B)(k, -q)   (psi~(q)^* = psi~(-q), then q -> -q)
    lhs = _num(m_, K, Q, times).conjugate()
    rhs = _num(W._conjugate_expr(m_, [q1]), K, [-x for x in Q], times)
    assert abs(lhs - rhs) < 1e-10 * (1 + abs(lhs))


def test_simplified_form_is_shorter_and_consistent():
    res, _ = _setup()
    st = W.simplify_result(res)[0]
    n_orig = len(sp.Add.make_args(res.terms[0].integrand))
    assert st.paired != 0 and st.plain == 0
    assert len(sp.Add.make_args(st.paired)) < n_orig
    tex = W.latex_result(res)
    assert r"\mathrm{Im}" in tex and "SQ" not in tex and "Ub" not in tex
    raw = W.latex_result(res, simplify=False)
    assert raw.count("u^{*}") > tex.count("u^{*}")


def test_quartic_vertex_and_free_density_read_naturally():
    m, lam = sp.symbols('m lambda', real=True)
    A = ScalarAction.parse(-KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24)
    H = legendre(reduced_lagrangian(A, 'scalar'), A.coupling)
    rho = hilbert_density('scalar', H, A)
    t1 = sp.Symbol('t1', real=True)
    R0 = to_local(rho.coefficients[(0, 0)], T, 'x')
    free = W.latex_result(W.wick_vev([R0], state=W.GaussianState()))
    assert r"\left|u'_{k_{1}}\!\left(\tau\right)\right|^{2}" in free
    Hl = to_vertex(H.interaction_coeffs[(1, 0)].subs(T, t1), t1)
    tad = W.latex_result(-sp.I * W.nested_commutator_vev([R0, Hl], connected=True, state=W.GaussianState()))
    assert r"\mathrm{Im}" in tad and r"\left|u_{k_{1}}\!\left(t_{1}\right)\right|^{2}" in tad
