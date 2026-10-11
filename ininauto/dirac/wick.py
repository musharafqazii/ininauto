"""In-in expectation values of Dirac bilinears with explicit fermionic Wick contractions.

Free field in the conformal frame (time-dependent mass M = a(tau) m):
    Psi(x) = int d^3p/(2pi)^3 sum_r [ b_{p,r} U_{p,r}(tau) e^{ip.x} + d^dag_{p,r} V_{p,r}(tau) e^{-ip.x} ],
    U_{p,r} = (u_A(p,tau) h_r, r u_B(p,tau) h_r)/sqrt(2)   (Dirac representation, sigma.p h_r = r |p| h_r),
    i u_A' = M u_A + |p| u_B,   i u_B' = |p| u_A - M u_B,   |u_A|^2 + |u_B|^2 = 2,   V = i gamma^2 U^*.
Spin sums (exact):
    P_p(t, t')   = sum_r U_{p,r}(t) U_{p,r}^dag(t')
                 = (1/2) [[u_A u_A'^* 1, u_A u_B'^* sigma.p/|p|], [u_B u_A'^* sigma.p/|p|, u_B u_B'^* 1]]
    N_p(t, t')   = sum_r V_{p,r}(t) V_{p,r}^dag(t') = gamma^2 P_p(t,t')^* gamma^2^dag
    <Psi(x) Psi^dag(y)>   = int_p e^{ip(x-y)} P_p(tau_x, tau_y)
    <Psi^dag(y) Psi(x)>^T = int_p e^{ip(x-y)} N_{-p}(tau_x, tau_y)

Fermion loop rule (verified against a brute-force Grassmann Wick expansion in the tests): for an ordered product
of bilinears O_i = Psi^dag K_i Psi (Weyl ordered, (Psi^dag K Psi - Psi K^T Psi^dag)/2 at a point), the connected
expectation is a sum over cyclic orders of the operators,
    <O_1 ... O_n>_c = - sum_cycles Tr[ K_{c1} G(c1,c2) K_{c2} G(c2,c3) ... K_{cn} G(cn,c1) ],
    G(i,j) = P (i stands left of j in the product),  G(i,j) = -N_{-p} (i stands right of j),
    G(i,i) = (P_p - N_{-p})/2 (a single Weyl-ordered bilinear).
Disconnected pieces cancel in nested commutators.  Metric perturbations are Fourier sources
psi(y) = int_q psi~(q, t) e^{iq.y}; momentum is conserved at every integrated vertex.
For Hermitian operators <w>^* = <reverse(w)>, so each nested commutator is written with Re/Im of half its words.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations, combinations_with_replacement, product

import sympy as sp

from .core import (Psi, Psib, GAM, G0, sig, X, x, y, z, tau, a, psi as psi_fn, _mi, _tex, DiracField)
from ..wick import SQ

# ---------------------------------------------------------------------------------------- printable objects
class _Mode(sp.Function):
    """mode function: args (|p| magnitude symbol, time)."""
    _name = "u"
    _conj = False

    def _latex(self, printer, exp=None):
        body = (r"u_{%s}" % self._name) + ("^{*}" if self._conj else "") + r"\!\left(" + printer._print(self.args[0]) + \
            ", " + printer._print(self.args[1]) + r"\right)"
        return body if exp is None else r"\left[" + body + r"\right]^{" + str(exp) + "}"


class uA(_Mode):
    _name, _conj = "A", False


class uB(_Mode):
    _name, _conj = "B", False


class uAc(_Mode):
    _name, _conj = "A", True


class uBc(_Mode):
    _name, _conj = "B", True


_CONJ = {uA: uAc, uAc: uA, uB: uBc, uBc: uB}


def conj_modes(e):
    """Complex conjugate of an expression built from mode functions, real symbols and numbers."""
    e = e.replace(lambda u: isinstance(u, _Mode), lambda u: _CONJ[type(u)](*u.args))
    return e.subs(sp.I, -sp.I)


class Mag(sp.Symbol):
    """|p| for an edge momentum p (printed as |k + q_1| ...)."""
    _tex_map = {}

    def _latex(self, printer):
        return Mag._tex_map.get(self.name, self.name)


# ---------------------------------------------------------------------------------------- vectors
def vec(name):
    return sp.Matrix(sp.symbols(f"{name}_x {name}_y {name}_z", real=True))


SIGMA = sig


def sigma_dot(p):
    return sum((p[i] * SIGMA[i] for i in range(3)), sp.zeros(2))


def _block(A, B, C, D):
    M = sp.zeros(4, 4)
    M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:] = A, B, C, D
    return M


G2 = GAM[2]


def P_matrix(p, mag, t1, t2):
    """P_p(t1, t2) (positive-frequency spin sum)."""
    S = sigma_dot(p) / mag
    A1, B1, A2c, B2c = uA(mag, t1), uB(mag, t1), uAc(mag, t2), uBc(mag, t2)
    I2 = sp.eye(2)
    return _block(A1 * A2c * I2, A1 * B2c * S, B1 * A2c * S, B1 * B2c * I2) / 2


def N_matrix_minus(p, mag, t1, t2):
    """N_{-p}(t1, t2) = gamma^2 P_{-p}(t1,t2)^* gamma^2^dag."""
    Pm = P_matrix(-p, mag, t1, t2)
    Pc = Pm.applyfunc(conj_modes)
    return (G2 * Pc * G2.H).applyfunc(sp.expand)


# ---------------------------------------------------------------------------------------- operators
@dataclass
class Bilinear:
    """sum over terms coef * (d^beta Psib_a) (d^alpha Psi_b) at one point; sources already Fourier transformed."""
    terms: list            # (coef, a, beta_spatial, b, alpha_spatial)
    sources: list          # list of q-vector names used by this operator
    time: sp.Symbol
    label: str


def _point_coords(k):
    if k == 0:
        return (tau, x, y, z)
    return tuple(sp.Symbol(f"{c}{k}", real=True) for c in ("t", "x", "y", "z"))


def make_bilinear(expr, k, qstart, label=""):
    """Parse a bilinear density at point k (0 = observation point x, k >= 1 vertex y_k at time t_k).
    Metric functions become Fourier amplitudes SQ(name, q_j, n_t, t) times (i q_j)^spatial derivatives;
    returns (Bilinear, next free source index)."""
    coords = _point_coords(k)
    tk = coords[0]
    terms = []
    nsrc = None
    for term in sp.Add.make_args(sp.expand(expr)):
        lb = rb = None
        coef = sp.S.One
        srcs = []
        for fac in sp.Mul.make_args(term):
            base, ex = fac.as_base_exp()
            core_ = base.expr if isinstance(base, sp.Derivative) else base
            if core_ in Psib:
                lb = (Psib.index(core_), _mi(base))
            elif core_ in Psi:
                rb = (Psi.index(core_), _mi(base))
            elif isinstance(core_, sp.core.function.AppliedUndef) and core_.func.__name__ != "a":
                srcs.extend([(core_, _mi(base))] * int(ex))
            else:
                coef *= fac
        if lb is None or rb is None:
            raise ValueError("not a bilinear term: " + str(term))
        if lb[1][0] or rb[1][0]:
            raise ValueError("time derivatives of Psi are not allowed (canonical form expected)")
        if nsrc is None:
            nsrc = len(srcs)
        elif nsrc != len(srcs):
            raise ValueError("terms with different numbers of metric factors in one operator")
        # the source momenta q_j are dummy integration variables entering only through their sum at this vertex:
        # assign them symmetrically (average over the orderings of the metric factors)
        from itertools import permutations as _perms
        perms = list(_perms(range(len(srcs)))) or [()]
        avg = sp.S.Zero
        for perm in perms:
            c_ = sp.S.One
            for j, idx in enumerate(perm):
                f_, mi = srcs[idx]
                qn = f"q{qstart + j}"
                qv = vec(qn)
                nm = str(f_.func)
                if nm.startswith("h") and len(nm) == 3:
                    amp = SQ(sp.Symbol("h"), sp.Symbol(qn), mi[0], tk, int(nm[1]), int(nm[2]))
                else:
                    amp = SQ(sp.Symbol(nm), sp.Symbol(qn), mi[0], tk)
                c_ *= amp * sp.Mul(*[(sp.I * qv[i]) ** mi[i + 1] for i in range(3)])
            avg += c_
        coef = sp.expand(coef * avg / len(perms))
        coef = coef.subs(dict(zip(X, coords)), simultaneous=True)
        terms.append((coef, lb[0], lb[1][1:], rb[0], rb[1][1:]))
    nsrc = nsrc or 0
    return Bilinear(terms, [f"q{qstart + j}" for j in range(nsrc)], tk, label), qstart + nsrc


def K_matrix(op: Bilinear, p_in, p_out):
    """Matrix of the bilinear between an incoming propagator (momentum p_in, acts on Psi^dag) and an outgoing one
    (p_out, acts on Psi): derivatives give (-i p_in) on Psi^dag and (i p_out) on Psi."""
    K = sp.zeros(4)
    for coef, ai, be, bi, al in op.terms:
        f = coef
        for i in range(3):
            if be[i]:
                f *= (-sp.I * p_in[i]) ** be[i]
            if al[i]:
                f *= (sp.I * p_out[i]) ** al[i]
        K[ai, bi] += f
    return K


# ---------------------------------------------------------------------------------------- loop evaluation
@dataclass
class WordValue:
    word: tuple            # operator order (indices into ops)
    integrand: sp.Expr
    momenta: list          # integration momenta (names): loop 'k' and sources
    mags: dict             # Mag symbol -> vector expression (for display)


_MAG_COUNTER = [0]


def _mag_for(pvec, names):
    """Magnitude symbol |p| for a vector expression p = sum c_j v_j (v_j named vectors)."""
    key = []
    tex = []
    for nm in names:
        v = vec(nm)
        c = sp.simplify(pvec[0].coeff(v[0]))
        if c != 0:
            key.append((nm, c))
    sym = "Mag_" + "_".join(f"{nm}{c}" for nm, c in key)
    sym = sym.replace("-", "m").replace("/", "d")
    ex = " + ".join(("" if c == 1 else ("-" if c == -1 else sp.latex(c))) + _vtex(nm) for nm, c in key).replace("+ -", "- ")
    Mag._tex_map[sym] = (r"\left|" + ex + r"\right|") if len(key) > 1 or (key and key[0][1] not in (1, -1)) else _vtex(key[0][0])
    return Mag(sym, positive=True)


def _vec_mag(nm):
    sym = f"Mag_{nm}1"
    Mag._tex_map[sym] = _vtex(nm)
    return Mag(sym, positive=True)


def _vtex(nm):
    return nm if len(nm) == 1 else f"{nm[0]}_{{{nm[1:]}}}"


def loop_word(ops, word, equal_point=False):
    """Connected expectation <O_{w1} ... O_{wn}> as an integrand (loop momentum k, sources q_j)."""
    n = len(word)
    pos = {op: i for i, op in enumerate(word)}
    kv = vec("k")
    names = ["k"] + [q for o in ops for q in o.sources]
    total = sp.S.Zero
    if n == 1:
        op = ops[word[0]]
        mag = _mag_for(kv, names)
        K = K_matrix(op, kv, kv)
        G = (P_matrix(kv, mag, op.time, op.time) - N_matrix_minus(kv, mag, op.time, op.time)) / 2
        return sp.expand(-(K * G).trace()), names
    rest = [o for o in range(len(ops)) if o != 0]
    for perm in permutations(rest):
        cyc = (0,) + perm
        # momentum on edge leaving cyc[j]
        p_out = {}
        p = kv
        p_out[cyc[0]] = p
        for j in range(1, n):
            Q = sum((vec(q) for q in ops[cyc[j]].sources), sp.zeros(3, 1))
            p = (p - Q).applyfunc(sp.expand)
            p_out[cyc[j]] = p
        p_in = {cyc[j]: p_out[cyc[j - 1]] for j in range(n)}
        M = sp.eye(4)
        for j in range(n):
            i_, nxt = cyc[j], cyc[(j + 1) % n]
            M = M * K_matrix(ops[i_], p_in[i_], p_out[i_])
            pe = p_out[i_]
            mag = _mag_for(pe, names)
            if pos[i_] < pos[nxt]:
                G = P_matrix(pe, mag, ops[i_].time, ops[nxt].time)
            else:
                G = -N_matrix_minus(pe, mag, ops[i_].time, ops[nxt].time)
            M = (M * G).applyfunc(sp.expand)
        total += -M.trace()
    return sp.expand(total), names


# ---------------------------------------------------------------------------------------- nested commutators
def nested_words(n):
    """Words and signs of <[[...[O_0, O_1], O_2] ...]>."""
    words = [(1, (0,))]
    for j in range(1, n):
        words = [(s_, w + (j,)) for s_, w in words] + [(-s_, (j,) + w) for s_, w in words]
    return words


@dataclass
class DiracTerm:
    label: str
    pieces: list           # (prefactor, kind in {'plain','Re','Im'}, integrand)
    names: list            # integration momenta
    times: list

    def latex(self, reduce=True):
        return latex_term(self, reduce=reduce)

    def _repr_latex_(self):
        return "$\\displaystyle " + self.latex() + "$"


def inin_term(D: DiracField, rho_order, h_orders=(), label=None):
    """(-i)^k < [[...[rho_n(x), H_{m1}(t1)], H_{m2}(t2)] ...] >  for the Dirac field D (tau >= t1 >= t2 ...)."""
    ops = []
    op, qn = make_bilinear(D.rho_n(rho_order), 0, 1, label=f"rho{rho_order}")
    ops.append(op)
    for j, mo in enumerate(h_orders, start=1):
        op, qn = make_bilinear(D.interaction(mo), j, qn, label=f"H{mo}")
        ops.append(op)
    n = len(ops)
    words = nested_words(n)
    done, pieces = set(), []
    pref = (-sp.I) ** (n - 1)
    names = None
    for s_, w in words:
        if w in done:
            continue
        rw = tuple(reversed(w))
        s_r = dict((ww, ss) for ss, ww in words).get(rw)
        val, names = loop_word(ops, w)
        done.add(w); done.add(rw)
        if rw == w:
            pieces.append((pref * s_, "plain", val))
        elif s_r == s_:
            pieces.append((2 * pref * s_, "Re", val))
        else:
            pieces.append((2 * sp.I * pref * s_, "Im", val))
    # turn i*Im / i*Re prefactors into real ones where possible
    out = []
    for c, kind, v in pieces:
        c = sp.nsimplify(c)
        out.append((c, kind, v))
    lab = label or (rf"\langle {'[' * (n - 1)}\rho_{{{rho_order}}}" +
                    "".join(rf", H_{{{mo}}}(t_{{{j}}})]" for j, mo in enumerate(h_orders, 1)) + r"\rangle")
    return DiracTerm(lab, out, names, [o.time for o in ops])


# ---------------------------------------------------------------------------------------- invariant reduction
def _components(names):
    return {nm: vec(nm) for nm in names}


def _shape_split(expr, comp_syms, tensor_syms):
    """Group a polynomial-in-components integrand by its non-polynomial factor."""
    groups = {}
    for t in sp.Add.make_args(sp.expand(expr)):
        poly, shape = sp.S.One, sp.S.One
        for f in sp.Mul.make_args(t):
            b, e = f.as_base_exp()
            if b in comp_syms or b in tensor_syms or f.is_number:
                poly *= f
            else:
                shape *= f
        groups[shape] = groups.get(shape, sp.S.Zero) + poly
    return groups


def _rand_vec(rng):
    return [Fraction(rng.randint(-7, 7), rng.randint(1, 5)) for _ in range(3)]


def _cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _tt(qv, rng):
    while True:
        r = _rand_vec(rng)
        e1 = _cross(qv, r)
        if any(e1):
            break
    e2 = _cross(qv, e1)
    n1 = sum(c * c for c in e1)
    n2 = sum(c * c for c in e2)
    al, be = Fraction(rng.randint(-5, 5), 3), Fraction(rng.randint(-5, 5), 4)
    H = [[al * (e1[i] * e1[j] / n1 - e2[i] * e2[j] / n2) + be * (e1[i] * e2[j] + e2[i] * e1[j]) for j in range(3)] for i in range(3)]
    return H


@dataclass
class _Inv:
    name: str          # LaTeX
    deg: tuple         # multidegree (vectors..., tensors...)
    fn: object         # numeric evaluator(vectors dict, tensors dict) -> Fraction


def _invariants(vnames, tnames, ttex=None):
    nv = len(vnames)
    invs = []

    def dg(vs=(), ts=()):
        d = [0] * (nv + len(tnames))
        for v in vs:
            d[vnames.index(v)] += 1
        for t in ts:
            d[nv + tnames.index(t)] += 1
        return tuple(d)

    def dot(u, v):
        return sum(p * q for p, q in zip(u, v))

    def mv(H, v):
        return [sum(H[i][j] * v[j] for j in range(3)) for i in range(3)]

    for i, u in enumerate(vnames):
        for v in vnames[i:]:
            nm = (rf"{_vtex(u)}^{{2}}" if u == v else rf"\left({_vtex(u)}\cdot {_vtex(v)}\right)")
            invs.append(_Inv(nm, dg((u, v)), lambda V, T, u=u, v=v: dot(V[u], V[v])))
    for trip in [(vnames[i], vnames[j], vnames[l]) for i in range(nv) for j in range(i + 1, nv) for l in range(j + 1, nv)]:
        u, v, w = trip
        invs.append(_Inv(rf"\left[{_vtex(u)},{_vtex(v)},{_vtex(w)}\right]", dg(trip), lambda V, T, u=u, v=v, w=w: dot(V[u], _cross(V[v], V[w]))))
    ttex = ttex or {t: rf"\tilde h({t})" for t in tnames}
    for t in tnames:
        ht = ttex[t]
        for i, u in enumerate(vnames):
            for v in vnames[i:]:
                invs.append(_Inv(rf"\left({_vtex(u)}\cdot{ht}\cdot {_vtex(v)}\right)", dg((u, v), (t,)),
                                 lambda V, T, u=u, v=v, t=t: dot(V[u], mv(T[t], V[v]))))
        for i, u in enumerate(vnames):
            for j, v in enumerate(vnames):
                for l, w in enumerate(vnames):
                    if i < j:
                        invs.append(_Inv(rf"\left[{_vtex(u)},{_vtex(v)},{ht}{_vtex(w)}\right]", dg((u, v, w), (t,)),
                                         lambda V, T, u=u, v=v, w=w, t=t: dot(V[u], _cross(V[v], mv(T[t], V[w])))))
    for i, s_ in enumerate(tnames):
        for t in tnames[i:]:
            invs.append(_Inv(rf"\mathrm{{tr}}\left[{ttex[s_]}{ttex[t]}\right]", dg((), (s_, t)),
                             lambda V, T, s_=s_, t=t: sum(T[s_][i][j] * T[t][j][i] for i in range(3) for j in range(3))))
            for a_, u in enumerate(vnames):
                for v in (vnames[a_:] if s_ == t else vnames):        # v.h1h2.w != w.h1h2.v for h1 != h2
                    invs.append(_Inv(rf"\left({_vtex(u)}\cdot{ttex[s_]}{ttex[t]}\cdot {_vtex(v)}\right)",
                                     dg((u, v), (s_, t)),
                                     lambda V, T, u=u, v=v, s_=s_, t=t: dot(V[u], mv(T[s_], mv(T[t], V[v])))))
    return invs


def _monomials(invs, target):
    """All products of invariants (with repetition) whose multidegrees add up to target."""
    out = []
    target = tuple(target)

    def rec(start, rem, chosen):
        if all(r == 0 for r in rem):
            out.append(tuple(chosen))
            return
        for i in range(start, len(invs)):
            d = invs[i].deg
            if all(dd <= r for dd, r in zip(d, rem)) and any(d):
                rec(i, tuple(r - dd for r, dd in zip(rem, d)), chosen + [i])
    rec(0, target, [])
    return out


def reduce_to_invariants(poly, vnames, tensors, rng=None):
    """Write a rotation-invariant polynomial in vector components (and TT tensor components) through scalar
    products / triple products / h-contractions.  Returns a sympy expression in symbols named by LaTeX, or None."""
    rng = rng or random.Random(11)
    comp = {nm: vec(nm) for nm in vnames}
    tnames = list(tensors)
    tensor_map = {tn: tensors[tn][1] for tn in tnames}
    tq = {tn: tensors[tn][0] for tn in tnames}
    ttex = {tn: tensors[tn][2] for tn in tnames}
    # multidegree decomposition
    pieces = {}
    for t in sp.Add.make_args(sp.expand(poly)):
        d = [sum(sp.degree(t, c) for c in comp[nm]) for nm in vnames]
        d += [sum(sp.degree(t, c) for c in tensor_map[tn].values()) for tn in tnames]
        pieces[tuple(d)] = pieces.get(tuple(d), sp.S.Zero) + t
    invs = _invariants(vnames, tnames, ttex)
    result = sp.S.Zero
    allsyms = [c for nm in vnames for c in comp[nm]] + [s_ for tn in tnames for s_ in tensor_map[tn].values()]
    for d, P in pieces.items():
        if all(v == 0 for v in d):
            result += P
            continue
        monos = _monomials(invs, d)
        if not monos:
            return None
        nsamp = len(monos) + 4
        rows, rhs = [], []
        for _ in range(nsamp):
            V = {nm: _rand_vec(rng) for nm in vnames}
            T = {}
            for tn in tnames:
                T[tn] = _tt(V[tq[tn]], rng)
            vals = [sp.Rational(c.numerator, c.denominator) for nm in vnames for c in V[nm]]
            for tn in tnames:
                for (i, j), s_ in tensor_map[tn].items():
                    c = T[tn][i][j]
                    vals.append(sp.Rational(c.numerator, c.denominator))
            rhs.append(sp.expand(P.xreplace(dict(zip(allsyms, vals)))))
            row = []
            for mono in monos:
                v = Fraction(1)
                for idx in mono:
                    v *= invs[idx].fn(V, T)
                row.append(sp.Rational(v.numerator, v.denominator))
            rows.append(row)
        A = sp.Matrix(rows)
        b = sp.Matrix(rhs)
        try:
            sol, params = A.gauss_jordan_solve(b)
        except ValueError:
            return None
        sol = sol.subs({p_: 0 for p_ in params})
        for coef, mono in zip(sol, monos):
            if coef != 0:
                term = coef
                for idx in mono:
                    term *= sp.Symbol(invs[idx].name)
                result += term
    return sp.expand(result)


# ---------------------------------------------------------------------------------------- LaTeX
def _tensor_map(expr):
    """Tensor sources SQ(h, q, n, t, i, j) -> placeholder symbols, grouped per (q, n, t).
    Returns ({key: (q name, {(i, j): symbol}, latex name)}, replacement dict)."""
    tmap, rep = {}, {}
    for s_ in expr.atoms(SQ):
        if len(s_.args) > 4:
            qn = str(s_.args[1])
            n_, t_ = s_.args[2], s_.args[3]
            i, j = int(s_.args[4]) - 1, int(s_.args[5]) - 1
            key = f"{qn}|{n_}|{t_}"
            sym = sp.Symbol(f"H_{qn}_{i}{j}_{n_}_{t_}", real=True)
            primes = "'" * int(n_) if int(n_) <= 3 else "^{(%d)}" % int(n_)
            tex = r"\tilde h%s(%s,%s)" % (primes, _vtex(qn), sp.latex(t_))
            ent = tmap.setdefault(key, (qn, {}, tex))
            ent[1][(i, j)] = sym
            rep[s_] = sym
    return tmap, rep


def latex_term(term: DiracTerm, reduce=True):
    chunks = []
    for c, kind, val in term.pieces:
        if val == 0:
            continue
        names = term.names
        comp_syms = set(s_ for nm in names for s_ in vec(nm))
        tmap, rep = _tensor_map(val)
        v2 = val.xreplace(rep)
        tsyms = set(s_ for ent in tmap.values() for s_ in ent[1].values())
        groups = _shape_split(v2, comp_syms, tsyms)
        body = []
        for shape, poly in groups.items():
            red = reduce_to_invariants(poly, list(names), tmap) if reduce else None
            pt = red if red is not None else poly
            if red is not None:                      # v^2 -> |v|^2 so that it cancels against 1/|v| of the modes
                pt = pt.subs({sp.Symbol(rf"{_vtex(nm)}^{{2}}"): _vec_mag(nm) ** 2 for nm in names})
            if sp.expand(pt) == 0:
                continue
            full = sp.factor_terms(sp.cancel(sp.factor(pt) * shape))
            body.append(sp.latex(full))
        if not body:
            continue
        inner = (r"\begin{aligned}&" + r" \\ &+ ".join(body) + r"\end{aligned}") if len(body) > 3 else " + ".join(body)
        inner = inner.replace("+ -", "- ")
        measure = r"\,".join(r"\int\!\frac{d^3%s}{(2\pi)^3}" % _vtex(nm) for nm in names)
        qs = [nm for nm in names if nm != "k"]
        phase = (r"e^{i(" + "+".join(_vtex(q) for q in qs) + r")\cdot x}") if qs else ""
        core = measure + r"\," + phase + r"\left\{" + inner + r"\right\}"
        if kind == "plain":
            chunks.append(sp.latex(c) + r"\," + core if c != 1 else core)
        else:
            cr = c
            op = r"\mathrm{Re}" if kind == "Re" else r"\mathrm{Im}"
            if (c / sp.I).is_real and kind == "Im":
                # c * i Im(...)  : keep as is (complex prefactor); real case handled below
                pass
            chunks.append(sp.latex(cr) + r"\," + op + r"\left[" + core + r"\right]")
    return (" + ".join(chunks) if chunks else "0").replace("+ -", "- ")


