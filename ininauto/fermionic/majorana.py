"""Free Majorana field on perturbed FLRW, explicit to O(eps^2), with Majorana Wick contractions.

Majorana condition (Dirac representation): psi = psi^c = i gamma^2 psi^*, i.e. psi^*_a = sum_b C_ab psi_b with
C = (i gamma^2)^*.  Action  S_M = (1/2) S_Dirac[psi]  (the Majorana kinetic and mass terms carry a factor 1/2), so
in the canonical conformal-frame variables

    L_M = (1/2) L_D(Psi),   H_M = (1/2) H_D(Psi),   a^4 rho_M = (1/2) a^4 rho_D(Psi),   Psi Majorana,

with the same tetrad, spin connection, canonical normalisation and Hamilton equations as the Dirac field (the
derivation in core.py applies unchanged; only the field is self-conjugate).

Wick contractions.  With psi^dag = psi^T C^T every bilinear is a quadratic form in psi alone,
    (1/2) Psi^dag K Psi = (1/2) psi^T (C^T K) psi,
and the only two-point function is
    W(x, y) = < psi(x) psi^T(y) > = int_p e^{ip(x-y)} P_p(tau_x, tau_y) (i gamma^2)^T,
which is NOT zero for Majorana fields (<psi psi> contractions).  Products of quadratic forms are evaluated with the
full fermionic Wick theorem: every perfect matching of the 2n legs, sign (-1)^(crossings), connected single cycles
only (disconnected pieces cancel in commutators); a single Weyl-ordered operator uses (<psi_a psi_b> - <psi_b psi_a>)/2.
The tests check: vacuum energy -omega_k per momentum (half the Dirac sea), and that every in-in term equals
1/2 of the Dirac one for these C-even vertices -- an independent check of both Wick engines.
"""
from __future__ import annotations

from functools import cached_property
from itertools import permutations

import sympy as sp

from .dirac import DiracField, GAM, Psi, Psib, _mi
from .wick import (Bilinear, DiracTerm, P_matrix, make_bilinear, nested_words, vec, _mag_for)

IG2 = sp.I * GAM[2]
CM = IG2.conjugate()          # psi^*_a = sum_b CM_ab psi_b


class MajoranaField:
    """Free Majorana field: same geometry and canonical structure as DiracField, operators carry a factor 1/2."""

    def __init__(self, family="scalar", order=2, mass=None):
        self.dirac = DiracField(family, order) if mass is None else DiracField(family, order, mass)
        self.family, self.N, self.m = family, self.dirac.N, self.dirac.m

    @cached_property
    def lagrangian_series(self):
        return [sp.expand(c / 2) for c in self.dirac.lagrangian_series]

    def H(self, n):
        return sp.expand(self.dirac.H(n) / 2)

    def interaction(self, n):
        return self.H(n)

    def rho_n(self, n):
        return sp.expand(self.dirac.rho_n(n) / 2)

    def clifford(self, expr):
        return self.dirac.clifford(expr)

    def checks(self):
        out = [(name.replace("H ", "H_M ").replace("rho", "rho_M"), ok) for name, ok in self.dirac.checks()]
        out.append(("Majorana condition: (i gamma^2)(i gamma^2)^* = 1 (psi^cc = psi)", sp.simplify(IG2 * IG2.conjugate()) == sp.eye(4)))
        return out


# ---------------------------------------------------------------------------------------- Majorana loops
def _legs(op: Bilinear):
    """Turn (coef, a', beta, b, alpha) terms into a matrix-valued quadratic form psi_L^T A psi_R with derivative data:
    returns list of (A-matrix (4x4) times coef, beta, alpha) where psi^dag_{a'} = sum_a CM_{a' a} psi_a."""
    out = []
    for coef, ai, be, bi, al in op.terms:
        A = sp.zeros(4)
        for a_ in range(4):
            if CM[ai, a_] != 0:
                A[a_, bi] += coef * CM[ai, a_]
        out.append((A, be, al))
    return out


def _deriv(p, d, sign):
    f = sp.S.One
    for i in range(3):
        if d[i]:
            f *= (sign * sp.I * p[i]) ** d[i]
    return f


def _matchings(items):
    if not items:
        yield []
        return
    a_ = items[0]
    for j in range(1, len(items)):
        rest = items[1:j] + items[j + 1:]
        for mm in _matchings(rest):
            yield [(a_, items[j])] + mm


def majorana_word(ops, word):
    """Connected < O_{w1} ... O_{wn} > for Majorana quadratic forms (full Wick theorem with <psi psi> contractions)."""
    n = len(word)
    kv = vec("k")
    names = ["k"] + [q for o in ops for q in o.sources]
    forms = {i: _legs(op) for i, op in enumerate(ops)}
    # legs in operator-string order: (op, side) with side 0 = L (from psi^dag), 1 = R
    legs = [(op, s_) for op in word for s_ in (0, 1)]
    posleg = {lg: i for i, lg in enumerate(legs)}
    total = sp.S.Zero
    if n == 1:
        op = word[0]
        mag = _mag_for(kv, names)
        Wxx = P_matrix(kv, mag, ops[op].time, ops[op].time) * IG2.T          # <psi_a(x) psi_b(x)>, momentum k
        val = sp.S.Zero
        for A, be, al in forms[op]:
            # (1/2)(<psi_L psi_R> - <psi_R psi_L>):  first argument gets +ik, second -ik
            t1 = _deriv(kv, be, +1) * _deriv(kv, al, -1) * sum(A[a_, b_] * Wxx[a_, b_] for a_ in range(4) for b_ in range(4))
            t2 = _deriv(kv, al, +1) * _deriv(kv, be, -1) * sum(A[a_, b_] * Wxx[b_, a_] for a_ in range(4) for b_ in range(4))
            val += (t1 - t2) / 2
        return sp.expand(val), names
    for mt in _matchings(list(range(2 * n))):
        pairs = [(legs[p], legs[q]) for p, q in mt]
        if any(lp[0] == lq[0] for lp, lq in pairs):
            continue
        # follow the cycle from word[0] leaving through its R leg
        partner = {}
        for lp, lq in pairs:
            partner[lp] = lq
            partner[lq] = lp
        start = (word[0], 1)
        seq = []                       # (op, entry side, exit side)
        cur_exit = start
        visited = set()
        while True:
            op, side = cur_exit
            if op in visited:
                break
            visited.add(op)
            seq.append(op)
            nxt = partner[cur_exit]
            cur_exit = (nxt[0], 1 - nxt[1])
        if len(visited) != n:
            continue
        cross = sum(1 for (p1, q1) in mt for (p2, q2) in mt if p1 < p2 < q1 < q2)
        sign = (-1) ** cross
        # traversal: op seq[j] is entered through side ent[j] and left through 1 - ent[j]
        ent = {}
        cur_exit = start
        for j in range(n):
            op, side = cur_exit
            nxt = partner[cur_exit]
            ent[nxt[0]] = nxt[1]
            cur_exit = (nxt[0], 1 - nxt[1])
        ent[word[0]] = 0
        # momenta: loop momentum k leaves word[0]; p_out = p_in - Q at integrated vertices
        p_out, p = {}, kv
        p_out[seq[0]] = p
        for j in range(1, n):
            Q = sum((vec(q) for q in ops[seq[j]].sources), sp.zeros(3, 1))
            p = (p - Q).applyfunc(sp.expand)
            p_out[seq[j]] = p
        p_in = {seq[j]: p_out[seq[j - 1]] for j in range(n)}
        M = sp.eye(4)
        for j in range(n):
            i_, nx = seq[j], seq[(j + 1) % n]
            # vertex matrix oriented (entry index, exit index)
            V = sp.zeros(4)
            for A, be, al in forms[i_]:
                d_entry, d_exit = (be, al) if ent[i_] == 0 else (al, be)
                f = _deriv(p_in[i_], d_entry, -1) * _deriv(p_out[i_], d_exit, +1)
                V += f * (A if ent[i_] == 0 else A.T)
            M = M * V
            # propagator from exit leg of i_ to entry leg of nx
            pe = p_out[i_]
            mag = _mag_for(pe, names)
            exit_leg, entry_leg = (i_, 1 - ent[i_]), (nx, ent[nx])
            if posleg[exit_leg] < posleg[entry_leg]:
                G = P_matrix(pe, mag, ops[i_].time, ops[nx].time) * IG2.T
            else:
                G = (P_matrix(-pe, mag, ops[nx].time, ops[i_].time) * IG2.T).T
            M = (M * G).applyfunc(sp.expand)
        total += sign * M.trace()
    return sp.expand(total), names


def majorana_inin_term(MF: MajoranaField, rho_order, h_orders=(), label=None, method="wick"):
    """(-i)^k < [[...[rho_n(x), H_{m1}(t1)], ...] > for the Majorana field (same conventions as inin_term).

    method="wick"        explicit Majorana Wick contractions (including <psi psi>), default
    method="half_dirac"  1/2 of the Dirac result -- identical for these C-even vertices (verified term by term in
                         the tests); useful for the slow double insertion (0, (1, 1))."""
    if method == "half_dirac":
        from .wick import inin_term
        T = inin_term(MF.dirac, rho_order, h_orders, label=label)
        return DiracTerm(T.label, [(c / 2, k_, v) for c, k_, v in T.pieces], T.names, T.times)
    ops = []
    op, qn = make_bilinear(MF.rho_n(rho_order), 0, 1, label=f"rho{rho_order}")
    ops.append(op)
    for j, mo in enumerate(h_orders, start=1):
        op, qn = make_bilinear(MF.interaction(mo), j, qn, label=f"H{mo}")
        ops.append(op)
    n = len(ops)
    words = nested_words(n)
    pref = (-sp.I) ** (n - 1)
    done, pieces, names = set(), [], None
    sign_of = {w: s_ for s_, w in words}
    for s_, w in words:
        if w in done:
            continue
        rw = tuple(reversed(w))
        val, names = majorana_word(ops, w)
        done.add(w); done.add(rw)
        if rw == w:
            pieces.append((pref * s_, "plain", val))
        elif sign_of.get(rw) == s_:
            pieces.append((2 * pref * s_, "Re", val))
        else:
            pieces.append((2 * sp.I * pref * s_, "Im", val))
    lab = label or "Majorana"
    return DiracTerm(lab, [(sp.nsimplify(c), k_, v) for c, k_, v in pieces], names, [o.time for o in ops])
