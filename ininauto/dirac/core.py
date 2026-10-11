"""Free Dirac field on perturbed FLRW, every step explicit, to O(eps^N) (default N = 2).

Conventions
-----------
* Spinor algebra uses the (+,-,-,-) metric g_D = -g (g is the package's (-,+,+,+) metric).  The tetrad is the same
  for both signs.  Dirac representation: gamma^0 = diag(1,1,-1,-1), gamma^i = [[0, sigma^i], [-sigma^i, 0]],
  {gamma^a, gamma^b} = 2 eta_D^{ab}.
* Physical metric g = a(tau)^2 gamma.  Families: 'scalar' gamma = diag(-(1+2 eps psi), (1-2 eps psi) delta_ij)
  (Newtonian gauge, equal potentials) and 'tensor' gamma = diag(-1, delta_ij + eps h_ij), h_ij symmetric and
  traceless (h33 = -h11 - h22 is substituted; transversality is used when results are simplified).
* Symmetric tetrad e^a_mu (e^0_0 = sqrt(gamma_D00), spatial block = symmetric square root), spin connection
  omega_mu^a_b = e^a_nu (d_mu e_b^nu + Gamma^nu_{mu lam} e_b^lam),  Omega_mu = (1/4) omega_mu_ab gamma^a gamma^b.
* Action  S = int sqrt(-g) [ (i/2)(psibar gamma^mu nabla_mu psi - (nabla_mu psibar) gamma^mu psi) - m psibar psi ].
  Conformal frame: psi = a^{-3/2} chi (exact; the mass becomes a(tau) m).  Canonical field
  Psi = f^{1/2} chi with f = sqrt(-gamma) e_0^0, so that L = (i/2)(Psi^dag Psi' - Psi'^dag Psi) - H exactly.
* Energy density for the normal observer (on-shell symmetric stress tensor):
      a^4 rho = (1/sqrt(-gamma)) [ (i/2)(Psi^dag Psi' - Psi'^dag Psi) + (i/2) Psi^dag (Omega_0 + gamma^0 Omega_0 gamma^0) Psi ]
  with Psi' = -i dH/dPsi^dag and Psi'^dag = i dH/dPsi (Hamilton's equations) -> canonical form.

All quantities are kept as truncated eps-series (lists of coefficients), so no product beyond eps^N is formed.
Spinor components are explicit functions Psi1..Psi4 (field) and Psib1..Psib4 (its Hermitian conjugate, treated as
independent); DiracField.clifford(...) prints a bilinear as Psi^dag [sum c_A Gamma_A] Psi.
"""
from __future__ import annotations

from functools import cached_property

import sympy as sp

from ..symbols import E as eps, T as tau, a

x, y, z = sp.symbols("x y z", real=True)
X = (tau, x, y, z)
m = sp.Symbol("m", positive=True)
psi = sp.Function("psi", real=True)(*X)


def hf(i, j):
    i, j = sorted((i, j))
    return sp.Function(f"h{i}{j}", real=True)(*X)


Psi = [sp.Function(f"Psi{n}")(*X) for n in range(1, 5)]
Psib = [sp.Function(f"Psib{n}")(*X) for n in range(1, 5)]

# ---------------------------------------------------------------------------------------- Dirac matrices
I2 = sp.eye(2)
sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]


def _block(A, B, C, D):
    M = sp.zeros(4, 4)
    M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:] = A, B, C, D
    return M


G0 = _block(I2, sp.zeros(2), sp.zeros(2), -I2)
GAM = [G0] + [_block(sp.zeros(2), s, -s, sp.zeros(2)) for s in sig]          # gamma^a (upper index)
ETA = sp.diag(1, -1, -1, -1)
G5 = sp.I * GAM[0] * GAM[1] * GAM[2] * GAM[3]


def clifford_basis():
    """16 basis matrices with LaTeX names (orthogonal under Tr(A^-1 B)/4)."""
    names, mats = ["1"], [sp.eye(4)]
    for a_ in range(4):
        names.append(r"\gamma^{%d}" % a_); mats.append(GAM[a_])
    for a_ in range(4):
        for b_ in range(a_ + 1, 4):
            names.append(r"\gamma^{%d}\gamma^{%d}" % (a_, b_)); mats.append(GAM[a_] * GAM[b_])
    for a_ in range(4):
        names.append(r"\gamma^{%d}\gamma^{5}" % a_); mats.append(GAM[a_] * G5)
    names.append(r"\gamma^{5}"); mats.append(G5)
    return names, mats


# ---------------------------------------------------------------------------------------- eps-series helpers
# A series is a list [c_0, ..., c_N] of sympy expressions or matrices (coefficients of eps^n).
def _x(v):
    return v.applyfunc(sp.expand) if isinstance(v, sp.MatrixBase) else sp.expand(v)


def s_const(v, N):
    zero = sp.zeros(*v.shape) if isinstance(v, sp.MatrixBase) else sp.S.Zero
    return [v] + [zero] * N


def s_add(A, B):
    return [_x(p + q) for p, q in zip(A, B)]


def s_scale(A, c):
    return [_x(c * p) for p in A]


def s_mul(A, B):
    N = len(A) - 1
    out = []
    for n in range(N + 1):
        acc = None
        for i in range(n + 1):
            p, q = A[i], B[n - i]
            if _is_zero(p) or _is_zero(q):
                continue
            t = p * q
            acc = t if acc is None else acc + t
        if acc is None:
            r = A[0] * B[0]
            acc = r * 0 if isinstance(r, sp.MatrixBase) else sp.S.Zero
        out.append(_x(acc))
    return out


def _is_zero(v):
    if isinstance(v, sp.MatrixBase):
        return all(e_ == 0 for e_ in v)
    return v == 0


def s_diff(A, var):
    return [_x(p.diff(var)) if isinstance(p, sp.MatrixBase) else sp.expand(sp.diff(p, var)) for p in A]


def s_pow(A, p):
    """(A_0 + t)^p for scalar series with A_0 = 1."""
    N = len(A) - 1
    t = [sp.S.Zero] + list(A[1:])
    out, term = s_const(sp.S.One, N), s_const(sp.S.One, N)
    for n in range(1, N + 1):
        term = s_mul(term, t)
        out = s_add(out, s_scale(term, sp.binomial(p, n)))
    return out


def m_pow(A, p):
    """(1 + T)^p for a matrix series with A_0 = identity."""
    N = len(A) - 1
    n_ = A[0].shape[0]
    t = [sp.zeros(n_)] + list(A[1:])
    out, term = s_const(sp.eye(n_), N), s_const(sp.eye(n_), N)
    for n in range(1, N + 1):
        term = s_mul(term, t)
        out = s_add(out, s_scale(term, sp.binomial(p, n)))
    return out


def s_entry(A, i, j):
    return [p[i, j] for p in A]


def s_sum(series_list, N, matrix_shape=None):
    acc = s_const(sp.zeros(*matrix_shape) if matrix_shape else sp.S.Zero, N)
    for s_ in series_list:
        acc = s_add(acc, s_)
    return acc


def s_expr(A):
    return sum(eps**n * c for n, c in enumerate(A))


def trunc(e, N):
    e = sp.expand(e)
    if not e.has(eps):
        return e
    return sp.Add(*[e.coeff(eps, n) * eps**n for n in range(N + 1)])


def euler(L, f):
    """Euler-Lagrange expression dL/df - d_mu dL/d(d_mu f) + ... (any derivative order)."""
    L = sp.expand(L)
    ders = sorted({D for D in L.atoms(sp.Derivative) if D.expr == f}, key=lambda D: -D.derivative_count)
    dm = {D: sp.Dummy() for D in ders}
    f0 = sp.Dummy()
    Ls = L.xreplace(dm).xreplace({f: f0})
    back = {v: k for k, v in dm.items()}
    back[f0] = f
    out = sp.diff(Ls, f0).xreplace(back)
    for D, s_ in dm.items():
        part = sp.diff(Ls, s_).xreplace(back)
        if part == 0:
            continue
        vs = [v for v, n in D.variable_count for _ in range(n)]
        out += (-1) ** len(vs) * sp.diff(part, *vs)
    return sp.expand(out)


def dagger(e):
    """Hermitian conjugate of a bilinear: Psi <-> Psib, i -> -i (metric functions and a(tau) are real)."""
    swap = {}
    for p_, q_ in zip(Psi, Psib):
        swap[p_.func] = q_.func
        swap[q_.func] = p_.func
    e = e.replace(lambda u: isinstance(u, sp.core.function.AppliedUndef) and u.func in swap,
                  lambda u: swap[u.func](*u.args))
    return sp.expand(e.subs(sp.I, -sp.I))


# ---------------------------------------------------------------------------------------- the field
class DiracField:
    """Free Dirac field on the perturbed FLRW family ('scalar' or 'tensor'), expanded to eps^order."""

    def __init__(self, family="scalar", order=2, mass=m):
        self.family, self.N, self.m = family, int(order), mass
        N = self.N
        X1 = sp.zeros(4)
        if family == "scalar":
            # gD = -gamma = diag(1 + 2 eps psi, -(1 - 2 eps psi) delta)
            X1 = sp.diag(2 * psi, 2 * psi, 2 * psi, 2 * psi)
            self.perturbations = [psi]
        elif family == "tensor":
            H = sp.Matrix(3, 3, lambda i, j: hf(i + 1, j + 1))
            H = H.subs(hf(3, 3), -hf(1, 1) - hf(2, 2))            # traceless
            X1[1:, 1:] = -H
            self.perturbations = [hf(i, j) for i in range(1, 4) for j in range(i, 4) if (i, j) != (3, 3)]
        else:
            raise ValueError("family must be 'scalar' or 'tensor'")
        self.gD = [ETA, X1] + [sp.zeros(4)] * (N - 1)               # exact: gD = eta + eps X1
        self._build_geometry()

    # ---------------------------------------------------------------- geometry
    def _build_geometry(self):
        N, gD = self.N, self.gD
        # inverse metric: B0 = eta, B_n = -eta sum_k X_k B_{n-k}
        inv = [ETA]
        for n in range(1, N + 1):
            acc = sp.zeros(4)
            for k in range(1, n + 1):
                acc += gD[k] * inv[n - k]
            inv.append(_x(-ETA * acc))
        self.gDinv = inv
        # tetrad e^a_mu and inverse e_a^mu (zero shift; spatial block is minus the spatial part of gD)
        g00 = [gD[n][0, 0] for n in range(N + 1)]
        Sp = [-gD[n][1:, 1:] for n in range(N + 1)]
        e00, ie00 = s_pow(g00, sp.Rational(1, 2)), s_pow(g00, -sp.Rational(1, 2))
        S, Si = m_pow(Sp, sp.Rational(1, 2)), m_pow(Sp, -sp.Rational(1, 2))
        e, ei = [], []
        for n in range(N + 1):
            M = sp.zeros(4); M[0, 0] = e00[n]; M[1:, 1:] = S[n]; e.append(M)
            Mi = sp.zeros(4); Mi[0, 0] = ie00[n]; Mi[1:, 1:] = Si[n]; ei.append(Mi)
        self.e_up, self.e_dn = e, ei
        # sqrt(-gamma) = e^0_0 det(S);  det(1+Y) = exp(tr log(1+Y))
        Y = [sp.zeros(3)] + S[1:]
        trlog, Yk = s_const(sp.S.Zero, N), s_const(sp.eye(3), N)
        for k in range(1, N + 1):
            Yk = s_mul(Yk, Y)
            trlog = s_add(trlog, s_scale([M.trace() for M in Yk], sp.Rational((-1) ** (k + 1), k)))
        detS, term = s_const(sp.S.One, N), s_const(sp.S.One, N)
        for k in range(1, N + 1):
            term = s_scale(s_mul(term, trlog), sp.Rational(1, k))
            detS = s_add(detS, term)
        self.sqrtg = s_mul(e00, detS)
        self.f = s_mul(self.sqrtg, ie00)
        # Christoffels Gamma^k_{mn} as series
        dg = [[[s_diff(s_entry(gD, m_, n_), X[l]) for n_ in range(4)] for m_ in range(4)] for l in range(4)]
        Gam = [[[None] * 4 for _ in range(4)] for _ in range(4)]
        for k in range(4):
            for m_ in range(4):
                for n_ in range(m_, 4):
                    acc = s_const(sp.S.Zero, N)
                    for l in range(4):
                        comb = s_add(s_add(dg[m_][l][n_], dg[n_][l][m_]), s_scale(dg[l][m_][n_], -1))
                        if all(c == 0 for c in comb):
                            continue
                        acc = s_add(acc, s_mul(s_entry(inv, k, l), comb))
                    Gam[k][m_][n_] = Gam[k][n_][m_] = s_scale(acc, sp.Rational(1, 2))
        self.Gamma = Gam
        # curved gammas gamma^mu = e_a^mu gamma^a
        self.gam_curved = []
        for mu in range(4):
            self.gam_curved.append([_x(sum((ei[n][a_, mu] * GAM[a_] for a_ in range(4)), sp.zeros(4))) for n in range(N + 1)])
        # spin connection
        Om = []
        for mu in range(4):
            wl = [[None] * 4 for _ in range(4)]
            for a_ in range(4):
                for b_ in range(4):
                    acc = s_const(sp.S.Zero, N)
                    for nu in range(4):
                        ea = s_entry(e, a_, nu)
                        if all(c == 0 for c in ea):
                            continue
                        inner = s_diff(s_entry(ei, b_, nu), X[mu])
                        for l in range(4):
                            eb = s_entry(ei, b_, l)
                            if all(c == 0 for c in eb):
                                continue
                            inner = s_add(inner, s_mul(Gam[nu][mu][l], eb))
                        acc = s_add(acc, s_mul(ea, inner))
                    wl[a_][b_] = s_scale(acc, ETA[a_, a_])               # lower the first index
            OmS = []
            for n in range(N + 1):
                M = sp.zeros(4)
                for a_ in range(4):
                    for b_ in range(4):
                        if wl[a_][b_][n] != 0:
                            M += wl[a_][b_][n] * GAM[a_] * GAM[b_]
                OmS.append(_x(M / 4))
            Om.append(OmS)
        self.Omega = Om

    def check_tetrad_postulate(self):
        """d_mu gamma^nu + Gamma^nu_{mu lam} gamma^lam + [Omega_mu, gamma^nu] = 0 (order by order)."""
        N = self.N
        for mu in range(4):
            for nu in range(4):
                tot = s_diff(self.gam_curved[nu], X[mu])
                for l in range(4):
                    tot = s_add(tot, s_mul([c * sp.eye(4) for c in self.Gamma[nu][mu][l]], self.gam_curved[l]))
                tot = s_add(tot, s_mul(self.Omega[mu], self.gam_curved[nu]))
                tot = s_add(tot, s_scale(s_mul(self.gam_curved[nu], self.Omega[mu]), -1))
                if any(not _is_zero(M) for M in tot):
                    return False
        return True

    # ---------------------------------------------------------------- action, Hamiltonian
    @cached_property
    def lagrangian_series(self):
        """Conformal-frame Lagrangian density in terms of the canonical field Psi, as an eps-series."""
        N = self.N
        fm = s_pow(self.f, -sp.Rational(1, 2))
        chi = [_x(c * sp.Matrix(Psi)) for c in fm]                      # column series
        bar = [_x(c * sp.Matrix([Psib]) * G0) for c in fm]             # row series  chi^dag gamma^0
        L = s_const(sp.S.Zero, N)
        for mu in range(4):
            Gm = self.gam_curved[mu]
            Dchi = s_add(s_diff(chi, X[mu]), s_mul(self.Omega[mu], chi))
            Dbar = s_add(s_diff(bar, X[mu]), s_scale(s_mul(bar, self.Omega[mu]), -1))
            t1 = s_mul(s_mul(bar, Gm), Dchi)
            t2 = s_mul(s_mul(Dbar, Gm), chi)
            L = s_add(L, s_scale([p[0, 0] - q[0, 0] for p, q in zip(t1, t2)], sp.I / 2))
        mass = s_mul(bar, chi)
        L = s_add(L, s_scale([p[0, 0] for p in mass], -a * self.m))
        return s_mul(self.sqrtg, L)

    @property
    def lagrangian(self):
        return sp.expand(s_expr(self.lagrangian_series))

    @cached_property
    def kinetic(self):
        return sp.expand(sum((sp.I / 2) * (q_ * sp.diff(p_, tau) - sp.diff(q_, tau) * p_) for p_, q_ in zip(Psi, Psib)))

    @cached_property
    def hamiltonian_series(self):
        L = self.lagrangian_series
        H = [sp.expand(-(c - (self.kinetic if n == 0 else 0))) for n, c in enumerate(L)]
        for c in H:
            bad = [D for D in c.atoms(sp.Derivative) if D.expr in Psi + Psib and any(v == tau for v in D.variables)]
            if bad:
                raise RuntimeError("time derivatives of Psi left in H: " + str(bad[:3]))
        return H

    @property
    def hamiltonian(self):
        return sp.expand(s_expr(self.hamiltonian_series))

    def H(self, n):
        return self.hamiltonian_series[n]

    def interaction(self, n):
        """Interaction Hamiltonian density at eps^n (n >= 1)."""
        return self.hamiltonian_series[n]

    @cached_property
    def eom(self):
        """(Psi', Psi'^dag) from Hamilton's equations: lists (component) of eps-series."""
        Hs = self.hamiltonian_series
        dP = [[sp.expand(-sp.I * euler(c, q_)) for c in Hs] for q_ in Psib]
        dPb = [[sp.expand(sp.I * euler(c, p_)) for c in Hs] for p_ in Psi]
        return dP, dPb

    @cached_property
    def rho_series(self):
        """Comoving energy density a^4 rho in canonical form (bilinear in Psi, Psib; no time derivatives)."""
        N = self.N
        dP, dPb = self.eom
        kin = s_const(sp.S.Zero, N)
        for p_, q_, dp, dq in zip(Psi, Psib, dP, dPb):
            kin = s_add(kin, s_scale([_x(q_ * c1 - c2 * p_) for c1, c2 in zip(dp, dq)], sp.I / 2))
        rowP, colP = sp.Matrix([Psib]), sp.Matrix(Psi)
        W = s_add(self.Omega[0], [_x(G0 * M * G0) for M in self.Omega[0]])
        rot = s_scale([(rowP * M * colP)[0, 0] for M in W], sp.I / 2)
        return s_mul(s_pow(self.sqrtg, -1), s_add(kin, rot))

    @property
    def rho(self):
        return sp.expand(s_expr(self.rho_series))

    def rho_n(self, n):
        return self.rho_series[n]

    # ---------------------------------------------------------------- checks
    def checks(self):
        out = [("tetrad postulate  d gamma^nu + Gamma gamma + [Omega, gamma^nu] = 0", self.check_tetrad_postulate())]
        ok = True
        for c in self.hamiltonian_series:
            dH = sp.expand(c - dagger(c))
            if dH != 0 and not all(euler(dH, f_) == 0 for f_ in Psi + Psib):
                ok = False
        out.append(("H is Hermitian up to a total derivative (every order)", ok))
        ok = True
        for c in self.rho_series:
            dr = sp.expand(c - dagger(c))
            if dr != 0 and not all(euler(dr, f_) == 0 for f_ in Psi + Psib):
                ok = False
        out.append(("a^4 rho is Hermitian up to a total derivative (every order)", ok))
        d0 = sp.expand(self.rho_n(0) - self.H(0))
        out.append(("a^4 rho_0 - H_0 is a total derivative", all(euler(d0, f_) == 0 for f_ in Psi + Psib)))
        return out

    # ---------------------------------------------------------------- display
    def clifford(self, expr):
        """LaTeX of a bilinear as  sum  (d^beta Psi)^dag [ sum_A c_A Gamma_A ] (d^alpha Psi)."""
        names, mats = clifford_basis()
        inv = [Mi.inv() for Mi in mats]
        groups = {}
        for term in sp.Add.make_args(sp.expand(expr)):
            lb = rb = None
            coef = sp.S.One
            for fac in sp.Mul.make_args(term):
                base = fac.expr if isinstance(fac, sp.Derivative) else fac
                if base in Psib:
                    lb = (Psib.index(base), _mi(fac))
                elif base in Psi:
                    rb = (Psi.index(base), _mi(fac))
                else:
                    coef *= fac
            if lb is None or rb is None:
                continue
            key = (lb[1], rb[1])
            K = groups.setdefault(key, sp.zeros(4))
            K[lb[0], rb[0]] += coef
        parts = []
        for (be, al), K in sorted(groups.items()):
            cs = []
            for nm, Mi in zip(names, inv):
                c = sp.factor(sp.expand((Mi * K).trace() / 4))
                if c != 0:
                    ct = _tex(c)
                    if isinstance(c, sp.Add):
                        ct = r"\left(" + ct + r"\right)"
                    cs.append(ct + ("" if nm == "1" else r"\," + nm))
            if not cs:
                continue
            parts.append(_dtex(be, r"\Psi^{\dagger}") + r"\left[" + " + ".join(cs).replace("+ -", "- ") + r"\right]"
                         + _dtex(al, r"\Psi"))
        return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def _mi(fac):
    cnt = [0, 0, 0, 0]
    if isinstance(fac, sp.Derivative):
        for v, n in fac.variable_count:
            cnt[X.index(v)] += n
    return tuple(cnt)


def _dtex(mi, name):
    labs = [r"\tau", "x", "y", "z"]
    s_ = "".join((r"\partial_{%s}" % labs[i]) + ("^{%d}" % n if n > 1 else "") for i, n in enumerate(mi) if n)
    return (s_ + name) if s_ else name


def _tex(e):
    """LaTeX with metric functions written without their arguments."""
    rep = {}
    labs = {tau: r"\tau", x: "x", y: "y", z: "z"}
    for D in e.atoms(sp.Derivative):
        if isinstance(D.expr, sp.core.function.AppliedUndef):
            nm = str(D.expr.func)
            nm = ("h_{%s}" % nm[1:]) if (nm.startswith("h") and len(nm) == 3) else sp.latex(sp.Symbol(nm))
            pre = "".join(r"\partial_{%s}" % labs[v] + ("^{%d}" % n if n > 1 else "") for v, n in D.variable_count)
            rep[D] = sp.Symbol(r"\left(" + pre + nm + r"\right)")
    e = e.xreplace(rep)
    rep2 = {}
    for f_ in e.atoms(sp.core.function.AppliedUndef):
        nm = str(f_.func)
        if nm == "a":
            rep2[f_] = sp.Symbol(r"a(\tau)") if f_.args == (tau,) else sp.Symbol(r"a(%s)" % sp.latex(f_.args[0]))
        elif nm.startswith("h") and len(nm) == 3:
            rep2[f_] = sp.Symbol("h_{%s}" % nm[1:])
        else:
            rep2[f_] = sp.Symbol(nm)
    return sp.latex(e.xreplace(rep2))
