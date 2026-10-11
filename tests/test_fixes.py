"""Regression tests for the two v0.2 bugs (composite momentum products; time-dependent M^2 at vertices)."""
import numpy as np
import sympy as sp
from ininauto.wick import dot, SP, U, Ub, SQ, F, S, local, vertex, wick_vev
from ininauto.action import ScalarAction, KIN, PHI2, RPHI2
from ininauto.expansion import reduced_lagrangian
from ininauto.hamiltonian import legendre
from ininauto.stress import hilbert_density
from ininauto.operators import to_local, to_vertex
from ininauto.symbols import T, a, M

k1, q1 = sp.symbols('k1 q1')


def test_dot_composite_momenta():
    assert sp.expand(dot(q1 - k1, q1 - k1) - (SP(k1, k1) - 2*SP(k1, q1) + SP(q1, q1))) == 0
    assert sp.expand(dot(k1 + q1, k1 - q1) - (SP(k1, k1) - SP(q1, q1))) == 0


def test_laplacian_on_shifted_leg():
    t, t1 = sp.symbols('t t1')
    r = wick_vev([local(F('phi')*F('phi', dx=('i', 'i')), t, 'x'), vertex(S('psi')*F('phi')*F('phi'), t1)], connected=True)
    e = r.terms[0].integrand
    e = e.subs({f: 1 for f in e.atoms(sp.Function) if f.func in (U, Ub, SQ)})
    assert sp.expand(e + 2*(SP(k1, k1) - 2*SP(k1, q1) + SP(q1, q1))) == 0


def _scalar_ops():
    m = sp.Symbol('m', real=True)
    act = ScalarAction.parse(-KIN/2 - m**2*PHI2/2 - RPHI2/12)
    h = legendre(reduced_lagrangian(act, 'scalar'), act.coupling)
    return h, hilbert_density('scalar', h, act)


def test_vertex_mass_carries_vertex_time():
    h, rho = _scalar_ops()
    t1 = sp.Symbol('t1', real=True)
    v = to_vertex(h.interaction_coeffs[(0, 1)].subs(T, t1), t1)
    r = wick_vev([to_local(rho.coefficients[(0, 0)], T, 'x'), v], connected=True)
    e = sp.expand(r.terms[0].integrand)
    assert e.has(M.subs(T, t1)) and e.has(M)
    assert not e.has(M**2) and not e.has(M.subs(T, t1)**2)


def test_rho0_H1_against_reference():
    """<rho_0(tau,x) H_1(t1)> = 2 E(k,q-k;tau) V(k,q-k,q;t1)  (reference derivation, symmetric under k <-> q-k)."""
    h, rho = _scalar_ops()
    t1 = sp.Symbol('t1', real=True)
    r = wick_vev([to_local(rho.coefficients[(0, 0)], T, 'x'), to_vertex(h.interaction_coeffs[(0, 1)].subs(T, t1), t1)],
                 connected=True)
    expr = r.terms[0].integrand
    rng = np.random.default_rng(7)
    kv, qv = rng.normal(size=3), rng.normal(size=3)
    tau, t1v, av = 0.83, 0.41, 1.3
    Mt = {tau: 0.7, t1v: 0.45}
    vec = {k1: kv, q1: qv}
    vals = {}
    def mode(kind, kk, dt, tt):
        key = (kind, round(float(np.linalg.norm(kk)), 12), dt, tt)
        if key not in vals: vals[key] = complex(rng.normal(), rng.normal())
        return vals[key]
    lin = lambda e: sum(float(sp.expand(e).coeff(s)) * v for s, v in vec.items())
    def ev(e, k_of):
        reps = {}
        for f in e.atoms(sp.Function):
            if f.func in (U, Ub):
                _, kk, dt, tt = f.args
                reps[f] = mode(f.func.__name__, k_of(lin(kk)), int(dt), float(tt.subs({T: tau, t1: t1v})))
            elif f.func == SP: reps[f] = float(np.dot(lin(f.args[0]), lin(f.args[1])))
            elif f.func == SQ: reps[f] = {0: 0.37, 1: -0.52}[int(f.args[2])]
            elif f == M: reps[f] = Mt[tau]
            elif f == M.subs(T, t1): reps[f] = Mt[t1v]
            elif f == a: reps[f] = av
        return complex(e.subs(reps))
    # the integrand is defined up to the loop relabelling k1 -> q1 - k1: compare the symmetrised values
    k = kv; p = qv - kv
    pkg = 0.5*(ev(expr, lambda x: x) + ev(expr.subs(k1, q1 - k1), lambda x: x))
    u = lambda kk, dt, tt: mode('U', kk, dt, tt); ub = lambda kk, dt, tt: mode('Ub', kk, dt, tt)
    E = (u(k, 1, tau)*u(p, 1, tau) + (Mt[tau] + (k@k + p@p - k@p)/3)*u(k, 0, tau)*u(p, 0, tau))/(2*av**4)
    V = (2*0.37*ub(k, 1, t1v)*ub(p, 1, t1v) - 0.52/2*(ub(k, 0, t1v)*ub(p, 1, t1v) + ub(p, 0, t1v)*ub(k, 1, t1v))
         - Mt[t1v]*0.37*ub(k, 0, t1v)*ub(p, 0, t1v) - 0.37/6*(qv@(k + p))*ub(k, 0, t1v)*ub(p, 0, t1v))
    assert abs(pkg - 2*E*V) < 1e-10*abs(2*E*V)
