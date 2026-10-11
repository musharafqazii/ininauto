"""Free Dirac field on perturbed FLRW: geometry, canonical structure, energy density, fermion Wick rules."""
import itertools
import random

import sympy as sp

from ininauto.dirac import DiracField, inin_term, P_matrix, N_matrix_minus, make_bilinear, loop_word
from ininauto.dirac import wick as DW


def test_scalar_family_structure_and_checks():
    D = DiracField("scalar", 2)
    for name, ok in D.checks():
        assert ok, name
    # H = Psi^dag[ sqrt(A/B) alpha.(-i grad) + sqrt(A) a m beta ]Psi with A = 1 + 2 eps psi, B = 1 - 2 eps psi
    tex1 = D.clifford(D.H(1))
    assert r"a(\tau) m \psi\,\gamma^{0}" in tex1 and r"- i \psi\,\gamma^{0}\gamma^{1}" in tex1
    tex2 = D.clifford(D.rho_n(2))
    assert r"\frac{15 a(\tau) m \psi^{2}}{2}\,\gamma^{0}" in tex2


def test_tensor_family_checks():
    D = DiracField("tensor", 1)
    for name, ok in D.checks():
        assert ok, name


def test_spin_sums_are_complete():
    k = DW.vec("k")
    t = sp.Symbol("t", real=True)
    K = sp.sqrt(sum(c**2 for c in k))
    P, N = P_matrix(k, K, t, t), N_matrix_minus(k, K, t, t)
    C = (P + N - sp.eye(4)).applyfunc(sp.expand)
    C = C.subs(DW.uB(K, t) * DW.uBc(K, t), 2 - DW.uA(K, t) * DW.uAc(K, t)).applyfunc(sp.simplify)
    assert C == sp.zeros(4)


def test_vacuum_energy_is_the_dirac_sea():
    D = DiracField("scalar", 0)
    op, _ = make_bilinear(D.H(0), 0, 1)
    val, _ = loop_word([op], (0,))
    Kk, M = sp.symbols("Kk M", positive=True)
    w = sp.sqrt(Kk**2 + M**2)
    mag = [s for s in val.free_symbols if isinstance(s, DW.Mag)][0]
    tt = sp.Symbol("tau", real=True)
    inst = {DW.uA(mag, tt): sp.sqrt(1 + M / w), DW.uAc(mag, tt): sp.sqrt(1 + M / w),
            DW.uB(mag, tt): sp.sqrt(1 - M / w), DW.uBc(mag, tt): sp.sqrt(1 - M / w)}
    k = DW.vec("k")
    e = val.subs(inst).subs({mag: Kk}).subs(D.m, 1).subs(sp.Function("a")(tt), M)
    e = e.subs({k[0]: 0, k[1]: 0, k[2]: Kk})
    assert sp.simplify(e + 2 * w) == 0                       # -2 omega per momentum (2 spins x (-w/2 - w/2))


def _brute_force(Ks, Pm, Nm, order):
    """Connected <O_{w1} ... O_{wn}> of bilinears psi^dag K psi from the fermionic Wick theorem (crossing signs)."""
    n = len(order)
    seq = []
    for pos_, op in enumerate(order):
        seq.append(("bar", op)); seq.append(("psi", op))
    total = 0
    idx = list(range(2 * n))

    def matchings(items):
        if not items:
            yield []
            return
        a_ = items[0]
        for j in range(1, len(items)):
            b_ = items[j]
            rest = items[1:j] + items[j + 1:]
            for mm in matchings(rest):
                yield [(a_, b_)] + mm
    for mt in matchings(idx):
        ok = True
        for p, q in mt:
            if seq[p][0] == seq[q][0] or seq[p][1] == seq[q][1]:
                ok = False
        if not ok:
            continue
        # connectedness: single cycle through all operators
        nxt = {}
        for p, q in mt:
            (tp, op_p), (tq, op_q) = seq[p], seq[q]
            if tp == "psi":
                nxt[op_p] = op_q
            else:
                nxt[op_q] = op_p
        cur, seen = order[0], set()
        while cur not in seen:
            seen.add(cur); cur = nxt[cur]
        if len(seen) != n:
            continue
        cross = sum(1 for (p1, q1), (p2, q2) in itertools.combinations(mt, 2) if (p1 < p2 < q1 < q2) or (p2 < p1 < q2 < q1))
        sign = (-1) ** cross
        # sum over spinor indices: build as matrix chain
        terms = 0
        for ab in itertools.product(range(2), repeat=2 * n):
            val = sign
            for o in range(n):
                val *= Ks[order[o]][ab[2 * o], ab[2 * o + 1]]
            for p, q in mt:
                (tp, op_p), (tq, op_q) = seq[p], seq[q]
                ip, iq = ab[p], ab[q]
                if tp == "psi":            # psi(op_p) left of psibar(op_q):  <psi psi^dag> = P(op_p, op_q)
                    val *= Pm[(op_p, op_q)][ip, iq]
                else:                       # psi^dag(op_p) left of psi(op_q): <psi^dag psi> = N(op_q, op_p)^T entry
                    val *= Nm[(op_q, op_p)][iq, ip]
            terms += val
        total += terms
    return sp.expand(total)


def test_fermion_loop_rule_matches_grassmann_wick():
    rng = random.Random(5)
    for n in (2, 3):
        Ks = [sp.Matrix(2, 2, lambda i, j: rng.randint(-3, 3)) for _ in range(n)]
        Pm = {(i, j): sp.Matrix(2, 2, lambda a_, b_: rng.randint(-3, 3)) for i in range(n) for j in range(n) if i != j}
        Nm = {(i, j): sp.Matrix(2, 2, lambda a_, b_: rng.randint(-3, 3)) for i in range(n) for j in range(n) if i != j}
        order = tuple(range(n))
        pos = {op: i for i, op in enumerate(order)}
        loop = 0
        for perm in itertools.permutations(range(1, n)):
            cyc = (0,) + perm
            M = sp.eye(2)
            for j in range(n):
                i_, nx = cyc[j], cyc[(j + 1) % n]
                G = Pm[(i_, nx)] if pos[i_] < pos[nx] else -Nm[(i_, nx)]
                M = M * Ks[i_] * G
            loop += -M.trace()
        assert sp.expand(loop - _brute_force(Ks, Pm, Nm, order)) == 0


def test_first_order_term_is_compact_and_reduced():
    D = DiracField("scalar", 1)
    T = inin_term(D, 0, (1,))
    tex = T.latex()
    assert r"\mathrm{Im}" in tex and "k_{x}" not in tex and "Mag_" not in tex


def test_majorana_checks_vacuum_energy_and_half_dirac():
    from ininauto.dirac import MajoranaField, majorana_inin_term
    MF = MajoranaField("scalar", 1)
    for name, ok in MF.checks():
        assert ok, name
    # vacuum energy: half the Dirac sea, -omega per momentum
    T = majorana_inin_term(MF, 0, ())
    val = T.pieces[0][2] * T.pieces[0][0]
    Kk, M = sp.symbols("Kk M", positive=True)
    w = sp.sqrt(Kk**2 + M**2)
    mag = [s for s in val.free_symbols if isinstance(s, DW.Mag)][0]
    tt = sp.Symbol("tau", real=True)
    inst = {DW.uA(mag, tt): sp.sqrt(1 + M / w), DW.uAc(mag, tt): sp.sqrt(1 + M / w),
            DW.uB(mag, tt): sp.sqrt(1 - M / w), DW.uBc(mag, tt): sp.sqrt(1 - M / w)}
    k = DW.vec("k")
    e = val.subs(inst).subs({mag: Kk}).subs(MF.m, 1).subs(sp.Function("a")(tt), M).subs({k[0]: 0, k[1]: 0, k[2]: Kk})
    assert sp.simplify(e + w) == 0
    # explicit Majorana Wick (with <psi psi> contractions) = 1/2 Dirac, term by term
    for args in [(1, ()), (0, (1,))]:
        TM = majorana_inin_term(MF, *args)
        TD = inin_term(MF.dirac, *args)
        for (cm, km, vm), (cd, kd, vd) in zip(TM.pieces, TD.pieces):
            assert km == kd and sp.expand(cm * vm - cd * vd / 2) == 0
