# ======================================================================================================
# In-in engine: direct expectation value <0_in| T_00 |0_in> of the dark-matter energy density.
#
# Numerical back end of InInAuto (ininauto.scalar.numerics.engine).  Every kernel below is re-derived from the input
# action by ininauto.scalar.bridge before any run.  Mode hierarchy as in the perturbation engine; at every record time N,
#
#     rho(eta, x) = T_{mu nu} n^mu n^nu = -T^0_0           (normal observer of the constant-eta slices)
#
# evaluated as an operator expectation value in the in-vacuum, from the Heisenberg mode functions:
#     <rho> = sum_p  E[V_p, V_p^*]       (symmetrised bilinear of the full stress tensor, xi = 1/6),
# with V_p = v_p e^{ipx} + (first-order side bands) + (second-order corrections) exactly as in the number
# calculation.  In the conformal frame g = a^2 gamma, a^4 rho = T^(gamma)_{mu nu} n^mu n^nu, and for the scalar
# Newtonian-gauge metric gamma = diag(-(1+2psi), (1-2psi) delta_ij) one finds (derived symbolically, see the
# inin notebooks)
#     rho_0 = v'^2/2 + (grad v)^2/2 + a^2 m^2 v^2/2 + lambda v^4/24 - Lap(v^2)/6
#     rho_1 = -psi v'^2 - psi' v v' + psi (grad v)^2 - psi Lap(v^2)/3 + v^2 Lap(psi)/3 + grad psi . grad(v^2)/6
#     rho_2 = 2 psi^2 v'^2 + (2/3) psi^2 (grad v)^2 - (4/3) psi^2 v Lap v + (4/3) psi v^2 Lap psi
#             + (4/3) psi v grad psi . grad v + v^2 (psi'^2 + (grad psi)^2)/2
# and for the tensor metric gamma_ij = delta_ij + h_ij
#     rho_1 = -h_ij d_i v d_j v/2 + d_i d_j (h_ij v^2)/6
#     rho_2 = -(1/3) h_ik h_kj v d_i d_j v + (1/6) h_ik h_kj d_i v d_j v + (1/12) v^2 h_ij Lap h_ij - (1/6) h_ij h_ij' v v'
#             + (1/6) h_ij d_k h_ij v d_k v - (1/48) v^2 h_ij' h_ij' + (1/16) v^2 d_k h_ij d_k h_ij
#             - (1/3) (d_i C_ij) v d_j v - (1/24) v^2 (d_j h_ik)(d_k h_ij) ,       C_ij = h_ik h_kj
# (= the book's E_2^T, checked symbolically; the last line vanishes for a single plane wave, hence in every mean here).
# Renormalisation: per Fourier mode, 4th-order adiabatic subtraction of the FRW vacuum (as quartic_hartree_run) and,
# for the metric perturbation, the local (0th adiabatic order) vacuum of the locally constant metric,
#     e_loc(k; psi) = omega~/2 (1-2psi)^(-3/2),  omega~^2 = k^2/(1-2psi) + a^2 m^2      (scalar)
#     e_loc(k; h)   = omega~/2 det(delta+h)^(-1/2), omega~^2 = (delta+h)^(-1)_ij k_i k_j + a^2 m^2   (tensor),
# expanded to the order needed.
# local_order = 4 (the in-in notebooks' default; local_order = 0 reproduces the earlier results): in addition the full
# local (adiabatic) expansion of every metric kernel through 4th order is subtracted mode by mode -- the static gradient
# terms (q^2, q^4), the mixed q^2 x (time derivative) terms and the homogeneous time-derivative terms of the perturbed
# FRW vacuum (functions _loc_* at the end of this cell).  Without them each integrand keeps a non-covariant tail
# ~ q^2 T/(6 omega) per mode (a p_cut^2 cutoff artefact) and the curvature-squared logarithm.  The subtracted result is
# the covariant value of the book at mu_R = m (adiabatic subtraction = DeWitt-Schwinger subtraction at mu = m); for
# another scale set P["mu_R"] and add ct1 / ct2 = -B ln(mu_R/m)/(1440 pi^2), B = n.n.(3 H2 - H1) of the conformal
# metric: scalar B1 = 4 q^4 T, <B2> = 12 q^4 T^2; tensor <B2> = (3/8)(q^4 T^2 + 2 q^2 T'^2 + 2 T' T''' - T''^2).
# Quartic coupling: optional smooth switch-on f_sw(N) (P["fsw"] = [N0, N1]); f_sw = 1 at every record time.
# The mean at second order is organised per TARGET momentum k (all terms of a target
# are evaluated with omega_k), so that the vacuum (1/2) omega_k cancels mode by mode through the canonical commutator.
# Outputs are comoving, a^4 rho, in units H_e^4.
# ======================================================================================================
import json, sys, time
import numpy as np

S_SHARP = 8.0
C_H = 2 ** (2 / S_SHARP)          # h_inf


def bg(N):
    Hc = np.exp(N) * C_H * (1 + np.exp(S_SHARP * N)) ** (-2 / S_SHARP)
    return Hc, -np.tanh(S_SHARP * N / 2)            # aH and g = d ln(aH)/dN


def bg_full(N):
    Hc, g = bg(N)
    gN = -(S_SHARP / 2) / np.cosh(S_SHARP * N / 2) ** 2
    gNN = (S_SHARP**2 / 2) / np.cosh(S_SHARP * N / 2) ** 2 * np.tanh(S_SHARP * N / 2)
    return Hc, g, gN, gNN


def omega_derivs(k, m, N):
    """omega and its conformal-time derivatives omega', omega'', omega''', omega'''' (as quartic_hartree_run)."""
    Hc, g, gN, gNN = bg_full(N)
    u = m**2 * np.exp(2 * N); w = np.sqrt(k**2 + u)
    wN = u / w; wNN = 2 * u / w - u**2 / w**3
    wNNN = 4 * u / w - 6 * u**2 / w**3 + 3 * u**3 / w**5
    wNNNN = 8 * u / w - 28 * u**2 / w**3 + 36 * u**3 / w**5 - 15 * u**4 / w**7
    w1 = Hc * wN
    w2 = Hc**2 * (wNN + g * wN)
    w3 = Hc**3 * (wNNN + 3 * g * wNN + (2 * g**2 + gN) * wN)
    w4 = Hc**4 * (wNNNN + 6 * g * wNNN + (11 * g**2 + 4 * gN) * wNN + (6 * g**3 + 7 * g * gN + gNN) * wN)
    return w, w1, w2, w3, w4


def W_ad4(k, m, N):
    """Fourth-order WKB frequency W = omega + W^(2) + W^(4) (xi = 1/6)."""
    w, w1, w2, w3, w4 = omega_derivs(k, m, N)
    W2 = (-2 * w * w2 + 3 * w1**2) / (8 * w**3)
    W4 = (8 * w**3 * w4 - 80 * w**2 * w1 * w3 - 52 * w**2 * w2**2 + 396 * w * w1**2 * w2 - 297 * w1**4) / (128 * w**7)
    return w + W2 + W4


def adiabatic_init(k, m, N, order):
    """Adiabatic in-vacuum data (v, dv/dN) of order 1 (as perturbation_engine) or 4 (as quartic_hartree_run)."""
    Hc, _ = bg(N)
    if order == 1:
        w, w1 = omega_derivs(k, m, N)[:2]
        v = 1 / np.sqrt(2 * w) + 0j
        return v, (-1j * w - w1 / (2 * w)) * v / Hc
    d = 1e-4
    W = W_ad4(k, m, N)
    WN = (W_ad4(k, m, N - 2 * d) - 8 * W_ad4(k, m, N - d) + 8 * W_ad4(k, m, N + d) - W_ad4(k, m, N + 2 * d)) / (12 * d)
    v = 1 / np.sqrt(2 * W) + 0j
    return v, (-1j * W - Hc * WN / (2 * W)) * v / Hc


def E_sub(k, m, N):
    """Fourth-order adiabatic vacuum energy of one mode (FRW, xi = 1/6): omega/2 + E^(2) + E^(4)."""
    w, w1, w2, w3, _ = omega_derivs(k, m, N)
    E2 = w1**2 / (16 * w**3)
    E4 = -(45 * w1**4 - 40 * w * w1**2 * w2 - 4 * w**2 * w2**2 + 8 * w**2 * w1 * w3) / (256 * w**7)
    return w / 2 + E2 + E4


# explicit Runge-Kutta tableaux: classical RK4 (as perturbation_engine) and Butcher's 7-stage sixth-order method.
# The in-in energy contains (omega/2) x (Wronskian error) of every vacuum mode, so it needs a more accurate integrator
# than the particle number (where only the negative-frequency part enters); RK6 at the same phase step reduces the
# Wronskian drift by ~ dtheta^2 / 100 relative to RK4.
TABLEAU = {
    "rk4": ([0.0, 0.5, 0.5, 1.0], [[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    "rk6": ([0.0, 1 / 3, 2 / 3, 1 / 3, 1 / 2, 1 / 2, 1.0],
            [[], [1 / 3], [0.0, 2 / 3], [1 / 12, 1 / 3, -1 / 12], [-1 / 16, 9 / 8, -3 / 16, -3 / 8],
             [0.0, 9 / 8, -3 / 8, -3 / 4, 1 / 2], [9 / 44, -9 / 11, 63 / 44, 18 / 11, 0.0, -16 / 11]],
            [11 / 120, 0.0, 27 / 40, 27 / 40, -4 / 15, -4 / 15, 11 / 120]),
}


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u**2)


def profile(kind, q, nend, dN=1e-3, N_match=1.0):
    """T(N), T_N, T_NN of the perturbation mode (unit primordial amplitude) -- identical to perturbation_engine."""
    N = np.arange(-12.0, nend + 0.5, dN)
    Hc, g = bg(N)
    if kind == "scalar":
        hH = Hc * np.exp(-N)
        f = np.exp(N) / hH
        I = np.exp(N[0]) / C_H + np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * dN)])
        T = 1 - hH * np.exp(-N) * I
        TN = np.gradient(T, dN)
        i0 = min(int(np.searchsorted(N, N_match)), N.size - 1)
        y, yN = T[i0], TN[i0]
        acc = lambda n, y, yN: -(2 - bg(n)[1]) * yN - q**2 / (3 * bg(n)[0] ** 2) * y
        for i in range(i0, N.size - 1):
            n = N[i]
            k1 = (yN, acc(n, y, yN))
            k2 = (yN + dN / 2 * k1[1], acc(n + dN / 2, y + dN / 2 * k1[0], yN + dN / 2 * k1[1]))
            k3 = (yN + dN / 2 * k2[1], acc(n + dN / 2, y + dN / 2 * k2[0], yN + dN / 2 * k2[1]))
            k4 = (yN + dN * k3[1], acc(n + dN, y + dN * k3[0], yN + dN * k3[1]))
            y, yN = y + dN / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), yN + dN / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            T[i + 1] = y
    else:
        Nx = np.log(q / C_H)
        hv = np.ones_like(N)
        i0 = min(int(np.searchsorted(N, Nx + 1.0)), N.size - 1)
        y, yN = 1.0, 0.0
        acc = lambda n, y, yN: -(2 + bg(n)[1]) * yN - (q / bg(n)[0]) ** 2 * y
        for i in range(i0, N.size - 1):
            n = N[i]
            k1 = (yN, acc(n, y, yN))
            k2 = (yN + dN / 2 * k1[1], acc(n + dN / 2, y + dN / 2 * k1[0], yN + dN / 2 * k1[1]))
            k3 = (yN + dN / 2 * k2[1], acc(n + dN / 2, y + dN / 2 * k2[0], yN + dN / 2 * k2[1]))
            k4 = (yN + dN * k3[1], acc(n + dN, y + dN * k3[0], yN + dN * k3[1]))
            y, yN = y + dN / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), yN + dN / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            hv[i + 1] = y
        T = smoothstep(N - Nx) * hv
    TN = np.gradient(T, dN)
    TNN = np.gradient(TN, dN)
    return N, T, TN, TNN


def run(P):
    t0 = time.time()
    kind, q, m = P["kind"], float(P["q"]), float(P["m"])
    LIN, QUAD, LAM = bool(P["lin"]), bool(P["quad"]), bool(P["lam"])
    PERT = LIN or QUAD                                                  # False: homogeneous run (no metric perturbation)
    SIG1, SIG2 = float(P.get("sig1", True)), float(P.get("sig2", True))
    INIT = int(P.get("init_order", 4))
    LOCAL = int(P.get("local_order", 0)) >= 4                          # full 4th-order local (adiabatic) subtraction
    LNMU = np.log(float(P.get("mu_R") or m) / m)                        # mu_R: renormalisation scale (physical, H_e)
    FSW = P.get("fsw")                                                  # [N0, N1]: smooth switch-on of lambda, or None
    def fsw(N):
        if not FSW: return 1.0, 0.0
        u = min(max((N - FSW[0]) / (FSW[1] - FSW[0]), 0.0), 1.0)
        return float(smoothstep(u)), float(30 * u**2 * (1 - u)**2 / (FSW[1] - FSW[0]))     # f, df/dN
    RK_C, RK_A, RK_B = TABLEAU[P.get("integrator", "rk6")]
    nend, dth = float(P["nend"]), float(P["dtheta"])
    p = np.logspace(np.log10(P["pmin"]), np.log10(P["pmax"]), int(P["np"]))
    lnp = np.log(p); WP = p**3 / (2 * np.pi**2)
    xg, wg = np.polynomial.legendre.leggauss(int(P["nmu"]) if PERT else 1); WMU = wg / 2
    mu = xg[:, None]; pp = p[None, :]
    p2 = np.sqrt(pp**2 + q**2 + 2 * pp * q * mu)                       # |p+q|, shape (M,P)
    pdot = pp**2 + pp * q * mu                                          # p.(p+q)
    Wp = 0.5 if kind == "scalar" else 1.0 / 8.0
    gcp = pp**2 * (1 - mu**2)                                           # tensor coupling (e_ij p_i p_j -> g)
    if PERT:
        NG, TG, TNG, TNNG = profile(kind, q, nend)
        TN3G = np.gradient(TNNG, NG[1] - NG[0]); TN4G = np.gradient(TN3G, NG[1] - NG[0])
    else:
        NG = np.array([-50.0, 50.0]); TG = TNG = TNNG = TN3G = TN4G = np.zeros(2)

    def trapP(f):  return np.sum((f[..., 1:] + f[..., :-1]) / 2 * np.diff(lnp), axis=-1)
    def avg(f):    return np.sum(WMU[:, None] * f, axis=0)

    def om(k, N):
        u = m**2 * np.exp(2 * N); w = np.sqrt(k**2 + u)
        return w, u / w, 2 * u / w - u**2 / w**3                       # omega, omega_N, omega_NN

    def proj(x, xN, k, N):                                              # beta e^{i theta} on first-order basis
        Hc, g = bg(N); w, wN, _ = om(k, N); F = Hc * wN / (2 * w)
        return (w * x - 1j * F * x - 1j * Hc * xN) / np.sqrt(2 * w)

    def sub_v2(N):
        Hc, g = bg(N); w, wN, wNN = om(p, N); w1 = Hc * wN; w2 = Hc**2 * (wNN + g * wN)
        return 1 / (2 * w) + (2 * w * w2 - 3 * w1**2) / (16 * w**5)

    def local_c(N):
        """0th-adiabatic-order local response of |v_k|^2 to a constant perturbation: c1 (1st), c2 (2nd, averaged)."""
        w = om(p, N)[0]; M2 = m**2 * np.exp(2 * N)
        if kind == "scalar":
            def f(ps): return 0.5 / np.sqrt((1 - 4 * ps + 6 * ps**2) * ((1 - 2 * ps**2) * p**2 + (1 - 2 * ps - 2 * ps**2) * M2))
            h = 1e-3
            return (f(h) - f(-h)) / (2 * h), (f(h) - 2 * f(0.0) + f(-h)) / (2 * h**2)
        return 0.0 * w, (1 / (2 * w)) * (0.25 - p**2 / (6 * w**2) + p**4 / (20 * w**4))

    # ---------------- state ----------------
    names = ["v"]
    if PERT: names += ["vu", "r_up", "r_bk"]
    if LAM: names += ["l"]
    if LAM and PERT: names += ["lu", "w_up", "w_bk"]
    if QUAD: names += ["s", "s0"]
    if QUAD and LAM: names += ["s1", "s10"]
    shape = {n: ((1, p.size) if n in ("v", "l", "s0", "s10") else (mu.size, p.size)) for n in names}
    mom = {n: (pp if n in ("v", "l", "r_bk", "w_bk", "s", "s0", "s1", "s10") else p2) for n in names}
    X = {n: np.zeros(shape[n], complex) for n in names}
    XN = {n: np.zeros(shape[n], complex) for n in names}
    free = [n for n in ("v", "vu") if n in names]
    Nin = {n: np.minimum(np.minimum(-6.0, np.log(mom[n] / m) - 6.0), np.log(mom[n] / (C_H * 400.0))) * np.ones(shape[n])
           for n in free}
    act = {n: np.zeros(shape[n], bool) for n in free}

    def Tprof(N):
        Hc, g = bg(N)
        T = np.interp(N, NG, TG); TN_ = np.interp(N, NG, TNG); TNN_ = np.interp(N, NG, TNNG)
        return T, Hc * TN_, Hc**2 * (TNN_ + g * TN_)                    # T, T', T'' (conformal)

    def Tjet(N):
        """T and its conformal-time derivatives 0..4 at N"""
        return _conformal_derivs([np.interp(N, NG, x) for x in (TG, TNG, TNNG, TN3G, TN4G)], N)

    def variances(N, X):
        """Delta_0, Delta_1 (coefficient of c e^{iqx}), Delta_2 (mean, per unit power), as in perturbation_engine."""
        on = act["v"][0]
        D0 = trapP(WP * np.where(on, np.abs(X["v"][0])**2 - sub_v2(N), 0.0))
        T = Tprof(N)[0]
        D1q, D2a = 0.0, 0.0
        if PERT and kind == "scalar":
            c1, _ = local_c(N)
            D1q = trapP(WP * avg(np.conj(X["v"]) * X["r_up"] + X["v"] * np.conj(X["r_up"][::-1]))) - T * trapP(WP * np.where(on, c1, 0.0))
        if QUAD:
            _, c2 = local_c(N)
            s_tot = 2 * Wp * avg(X["s"])[None, :] + X["s0"]
            D2a = trapP(WP * (2 * Wp * avg(np.abs(X["r_up"])**2) + 2 * np.real(np.conj(X["v"][0]) * s_tot[0]))) - T**2 * trapP(WP * np.where(on, c2, 0.0))
        return D0, D1q, D2a

    def rhs(N, X, XN):
        Hc, g = bg(N); M2 = m**2 * np.exp(2 * N)
        T, Tp, Tpp = Tprof(N)
        w1 = om(pp, N)[0]; w2 = om(p2, N)[0]
        A, src = {}, {}

        def acc(name, S):                                               # x_NN from x'' + omega^2 x = S
            w = w1 if mom[name] is pp else w2
            src[name] = S
            return -g * XN[name] - (w / Hc) ** 2 * X[name] + S / Hc**2

        def fpp(name):                                                  # conformal second derivative from own eq.
            w = w1 if mom[name] is pp else w2
            return -w**2 * X[name] + src[name]

        def D1(name, tensor_coupling=gcp):
            f, fp = X[name], Hc * XN[name]
            if kind == "scalar":
                return -4 * T * fpp(name) - 4 * Tp * fp - (Tpp + 2 * M2 * T + q**2 * T / 3) * f
            return -T * tensor_coupling * f

        def D2(name):                                                   # zero mode, ensemble averaged
            f, fp = X[name], Hc * XN[name]
            if kind == "scalar":
                return (6 * T**2 * fpp(name) + 12 * T * Tp * fp
                        + (2 * T * Tpp + Tp**2 - q**2 * T**2 / 3 - 2 * M2 * T**2 - 2 * pp**2 * T**2) * f)
            s2, s2p = -T**2 / 4, -T * Tp / 2
            R2 = (-q**2 * T**2 / 4 - T * Tpp - 0.75 * Tp**2 - 3 * Hc * T * Tp) / 6.0
            return s2 * src[name] + s2p * fp + (pp**2 * T**2 / 3) * f + (R2 - Hc * s2p) * f

        for n in free: A[n] = acc(n, 0.0)
        diag = {}
        if PERT:
            A["r_up"] = acc("r_up", -D1("v"))
            A["r_bk"] = acc("r_bk", -D1("vu"))
        if LAM:
            on = act["v"][0]
            D0 = trapP(WP * np.where(on, np.abs(X["v"][0])**2 - sub_v2(N), 0.0)); fs = fsw(N)[0]; sig0 = fs * D0 / 2
            A["l"] = acc("l", -sig0 * X["v"])
            if PERT:
                if kind == "scalar":
                    c1, _ = local_c(N)
                    D1q = trapP(WP * avg(np.conj(X["v"]) * X["r_up"] + X["v"] * np.conj(X["r_up"][::-1]))) - T * trapP(WP * np.where(on, c1, 0.0))
                    sig1 = fs * SIG1 * 0.5 * (D1q - 2 * T * D0)
                else:
                    D1q, sig1 = 0.0, 0.0
                diag.update(D0=D0, D1=D1q)
                A["lu"] = acc("lu", -sig0 * X["vu"])
                A["w_up"] = acc("w_up", -(D1("l") + sig0 * X["r_up"] + sig1 * X["v"]))
                A["w_bk"] = acc("w_bk", -(D1("lu") + sig0 * X["r_bk"] + np.conj(sig1) * X["vu"]))
        if QUAD:
            A["s"] = acc("s", -D1("r_up"))
            A["s0"] = acc("s0", -D2("v"))
            s_tot = 2 * Wp * avg(X["s"])[None, :] + X["s0"]
        if QUAD and LAM:
            _, c2 = local_c(N)
            D2a = trapP(WP * (2 * Wp * avg(np.abs(X["r_up"])**2) + 2 * np.real(np.conj(X["v"][0]) * s_tot[0]))) - T**2 * trapP(WP * np.where(on, c2, 0.0))
            if kind == "scalar": sig2 = fs * SIG2 * 0.5 * (D2a - 4 * Wp * T * np.real(D1q) - 2 * T**2 * D0)
            else:                sig2 = fs * SIG2 * 0.5 * (D2a - (T**2 / 4) * D0)
            diag.update(D2=D2a)
            A["s1"] = acc("s1", -(D1("w_up") + np.conj(sig1) * X["r_up"]))
            A["s10"] = acc("s10", -(D2("l") + sig0 * s_tot + sig2 * X["v"]))
        return A, diag

    # ---------------- read-out: particle numbers (as perturbation_engine) and the in-in energy density ----------------
    def projA(x, xN, k, N):                                             # alpha e^{-i theta} on first-order basis
        Hc, g = bg(N); w, wN, _ = om(k, N); F = Hc * wN / (2 * w)
        return (w * x + 1j * F * x + 1j * Hc * xN) / np.sqrt(2 * w)

    def B0(x, xN, k, N):                                                # zeroth-order (instantaneous) basis
        Hc, _ = bg(N); w = om(k, N)[0]
        return (w * x - 1j * Hc * xN) / np.sqrt(2 * w)

    def A0(x, xN, k, N):
        Hc, _ = bg(N); w = om(k, N)[0]
        return (w * x + 1j * Hc * xN) / np.sqrt(2 * w)

    def record(N, full):
        Hc, _ = bg(N); M2 = m**2 * np.exp(2 * N)
        T, Tp, Tpp = Tprof(N)
        d = lambda n: Hc * XN[n]                                       # conformal derivative of a state
        w = om(p, N)[0]; wk = om(pp, N)[0]
        on = act["v"][0].astype(float)
        onu = act["vu"].astype(float) if "vu" in act else 1.0
        R = dict(T=np.array([T, Tp, Tpp]))
        v, vp = X["v"][0], d("v")[0]
        # ---- zeroth order (homogeneous): a^4 rho per mode, 4th-order adiabatic subtraction
        e0 = 0.5 * np.abs(vp)**2 + 0.5 * w**2 * np.abs(v)**2
        R["rho0_k"] = on * (e0 - E_sub(p, m, N))
        R["n_inst"] = on * np.abs(B0(X["v"], XN["v"], pp, N)[0])**2
        R["n0"] = on * np.abs(proj(X["v"], XN["v"], pp, N)[0])**2
        R["omega"] = w
        D0, D1q, D2a = variances(N, X)
        R["D"] = np.array([D0, np.real(D1q), np.imag(D1q), D2a])
        if LAM:
            l, lp = X["l"][0], d("l")[0]
            # homogeneous O(lambda): exchange (cross term with the Hartree response) and contact lambda <v^2>^2 / 8
            Hc_, g_ = bg(N); wv, w1, w2, w3, _ = omega_derivs(p, m, N)
            step = 1e-6
            dsub = Hc * (sub_v2(N + step) - sub_v2(N - step)) / (2 * step)
            fs, fsN = fsw(N)
            dM2 = fs * D0 / 2
            dM2p = fs * trapP(WP * on * (2 * np.real(np.conj(v) * vp) - dsub)) / 2 + Hc * fsN * D0 / 2
            Lk = -dM2 * (wv * w2 + w1**2) / (16 * wv**5) + dM2p * w1 / (16 * wv**4)
            R["ex0_k"] = on * (np.real(np.conj(vp) * lp) + w**2 * np.real(np.conj(v) * l) - Lk)
            R["contact0"] = np.array(fs * D0**2 / 8)
            R["dn_lam_inst"] = on * 2 * np.real(np.conj(B0(X["v"], XN["v"], pp, N)[0]) * B0(X["l"], XN["l"], pp, N)[0])
        if PERT:
            ru, rup, rb, rbp, vu, vup = X["r_up"], d("r_up"), X["r_bk"], d("r_bk"), X["vu"], d("vu")
            msk = on[None, :] * onu
            def Bpair(f, fp_, g_, gp_):                                 # e^{iqx} coefficient of the free energy bilinear
                return 0.5 * fp_ * np.conj(gp_) + 0.5 * (pdot + M2) * f * np.conj(g_) + (q**2 / 6) * f * np.conj(g_)
            KE = Bpair(ru, rup, v[None, :], vp[None, :]) + Bpair(vu, vup, rb, rbp)
            if kind == "scalar":
                Eloc1 = (3 * w**2 + p**2) / (2 * w)                     # d/dpsi of e_loc
                KEl = T * (-np.abs(vp)**2 + (p**2 - q**2 / 3) * np.abs(v)**2) - Tp * np.real(v * np.conj(vp)) - T * Eloc1
                tot = msk * (KE + KEl[None, :])
                if LOCAL:                                               # 4th-order local subtraction (mu-independent)
                    R["loc1"] = on * _loc_scalar_1(p, q, _M2jet(m, N), Tjet(N))
                    tot = tot - msk * R["loc1"][None, :]
                R["ct1"] = np.array(-4 * q**4 * T * LNMU / (1440 * np.pi**2))
            else:
                KEl = -0.5 * T * gcp * np.abs(v[None, :])**2 + T * gcp / (4 * wk)      # rho_1 kernel and local vacuum
                tot = msk * (KE + KEl)                                  # coefficient of (e_ij p_i p_j / g); integrates to 0
            # scalar: e^{iqx} coefficient of the local energy density (per p); tensor: the density vanishes after the
            # azimuthal integral, so record instead the quadrupole moment int (e_ij p^_i p^_j) delta rho_p, i.e. avg((1-mu^2)/2 x tot)
            R["dE1_p"] = avg(tot) if kind == "scalar" else avg(0.5 * (1 - mu**2) * tot)
            if LOCAL and kind != "scalar":                              # local terms of the quadrupole moment
                R["loc1"] = on * _loc_tensor_1(p, q, _M2jet(m, N), Tjet(N)); R["dE1_p"] = R["dE1_p"] - R["loc1"]
            R["dE1_var"] = 2 * Wp * avg(np.abs(tot)**2)
            R["dE1_raw_var"] = 2 * Wp * avg(np.abs(msk * KE)**2)
            # (first-order local occupation kernel, as perturbation_engine)
            b_up = proj(ru, XN["r_up"], p2, N); b_bk = proj(rb, XN["r_bk"], pp, N); b_vu = proj(vu, XN["vu"], p2, N)
            beta = proj(X["v"], XN["v"], pp, N)
            R["K"] = beta * np.conj(b_up) + b_bk * np.conj(b_vu)
            if LAM:
                lu_, lup = X["lu"], d("lu"); wu, wup, wb, wbp = X["w_up"], d("w_up"), X["w_bk"], d("w_bk")
                KEL = (Bpair(ru, rup, l[None, :], lp[None, :]) + Bpair(wu, wup, v[None, :], vp[None, :])
                       + Bpair(vu, vup, wb, wbp) + Bpair(lu_, lup, rb, rbp))
                if kind == "scalar":
                    KELl = (T * (-2 * np.real(np.conj(vp) * lp) + (p**2 - q**2 / 3) * 2 * np.real(np.conj(v) * l))
                            - Tp * np.real(v * np.conj(lp) + l * np.conj(vp)))
                    totL = msk * (KEL + KELl[None, :])
                    R["dE1L_p"] = avg(totL)
                    R["contact1"] = np.array(fsw(N)[0] * D0 * D1q / 4)  # (1/8) 2 Delta_0 Delta_1
                else:
                    totL = msk * (KEL - T * gcp * np.real(np.conj(v[None, :]) * l[None, :]))
                    R["dE1L_p"] = avg(0.5 * (1 - mu**2) * totL); R["contact1"] = np.array(0j)
                R["dE1L_var"] = 2 * Wp * avg(np.abs(totL)**2)
        if QUAD:
            s_tot = 2 * Wp * avg(X["s"]) + X["s0"][0]; s_totN = 2 * Wp * avg(XN["s"]) + XN["s0"][0]
            sp_ = Hc * s_totN
            rb, rbp, vu, vup = X["r_bk"], d("r_bk"), X["vu"], d("vu")
            msk = on[None, :] * onu
            Ea = 2 * Wp * avg(msk * (0.5 * np.abs(rbp)**2 + 0.5 * w**2 * np.abs(rb)**2))
            Eb = on * 2 * np.real(0.5 * np.conj(vp) * sp_ + 0.5 * w**2 * np.conj(v) * s_tot)
            if kind == "scalar":
                def Xc(sv, svp, rv, rvp):                               # rho_1 cross term: source sv, its response rv
                    return (-2 * T * np.real(svp * np.conj(rvp)) - Tp * np.real(sv * np.conj(rvp) + rv * np.conj(svp))
                            + (2 * T * pdot + q**2 * T / 3) * np.real(sv * np.conj(rv)))
                # rho_1 cross terms, organised per momentum PAIR (p, p+q): source p with its response at p+q, and source
                # p+q with its response at p (the two halves cancel the vacuum part pair by pair, as for the first order)
                Ec = Wp * avg(msk * (Xc(v[None, :], vp[None, :], X["r_up"], d("r_up")) + Xc(vu, vup, rb, rbp)))
                Ed = on * (2 * T**2 * np.abs(vp)**2 + (2 * p**2 * T**2 + 0.5 * Tp**2 - 5 / 6 * q**2 * T**2) * np.abs(v)**2)
                Eloc = on * T**2 * (15 * M2**2 + 40 * M2 * p**2 + 24 * p**4) / (4 * w**3)
            else:
                Ec = -Wp * avg(msk * gcp * T * np.real(v[None, :] * np.conj(X["r_up"]) + vu * np.conj(rb)))
                Ed = on * ((T**2 * p**2 / 6 - (q**2 * T**2 + Tp**2) / 48) * np.abs(v)**2 - T * Tp / 6 * np.real(v * np.conj(vp)))
                Eloc = on * T**2 * (M2**2 / 8 + M2 * p**2 / 3 + p**4 / 5) / w**3
            if LOCAL:                                                   # 4th-order local subtraction, per target
                Md, Td = _M2jet(m, N), Tjet(N)
                R["loc2"] = on * (_loc_scalar_2(p, q, Md, Td) if kind == "scalar" else _loc_tensor_2(p, q, Md, Td))
                Eloc = Eloc + R["loc2"]
            T0, T1_, T2_, T3_ = Tjet(N)[:4]
            B2 = 12 * q**4 * T0**2 if kind == "scalar" else 3 / 8 * (q**4 * T0**2 + 2 * q**2 * T1_**2 + 2 * T1_ * T3_ - T2_**2)
            R["ct2"] = np.array(-B2 * LNMU / (1440 * np.pi**2))
            R.update(E2a=Ea, E2b=Eb, E2c=Ec, E2d=Ed, E2loc=Eloc, dE2_k=Ea + Eb + Ec + Ed - Eloc)
            # identity check:  E2a + E2b = omega dn2_inst + (omega/2) tnorm2   (zeroth-order basis, target counting)
            a0, b0 = A0(X["v"], XN["v"], pp, N)[0], B0(X["v"], XN["v"], pp, N)[0]
            As, Bs = A0(s_tot, s_totN, p, N), B0(s_tot, s_totN, p, N)
            Abk, Bbk = A0(rb, XN["r_bk"], pp, N), B0(rb, XN["r_bk"], pp, N)
            R["dn2_inst"] = on * (2 * np.real(np.conj(b0) * Bs) + 2 * Wp * avg(msk * np.abs(Bbk)**2))
            R["tnorm2"] = on * (2 * np.real(np.conj(a0) * As - np.conj(b0) * Bs) + 2 * Wp * avg(msk * (np.abs(Abk)**2 - np.abs(Bbk)**2)))
            R["tnorm2_scale"] = on * 2 * Wp * avg(msk * (np.abs(Abk)**2 + np.abs(Bbk)**2))
            # the same particle energy counted per SOURCE momentum (free of the grid-edge leakage of the redistribution)
            Bup = B0(X["r_up"], XN["r_up"], p2, N); wu2 = om(p2, N)[0]
            R["wdn2_src_inst"] = on * (w * 2 * np.real(np.conj(b0) * Bs) + 2 * Wp * avg(msk * wu2 * np.abs(Bup)**2))
            R["dE2_vac_k"] = 0.5 * w * R["tnorm2"] + Ec + Ed - Eloc                 # vacuum polarisation + metric terms
            R["dE2_hyb_k"] = R["wdn2_src_inst"] + R["dE2_vac_k"]                    # in-in total, leakage-free organisation
            # (number bookkeeping, first-order basis, as perturbation_engine)
            beta = proj(X["v"], XN["v"], pp, N)[0]
            b_s = proj(s_tot, s_totN, p, N); b_bk = proj(rb, XN["r_bk"], pp, N); b_up = proj(X["r_up"], XN["r_up"], p2, N)
            R["gain"] = 2 * Wp * avg(np.abs(b_bk)**2); R["diag"] = 2 * np.real(np.conj(beta) * b_s)
            R["dn2"] = R["gain"] + R["diag"]
            R["dn2_src"] = R["diag"] + 2 * Wp * avg(np.abs(b_up)**2)
            if LAM:
                l, lp = X["l"][0], d("l")[0]
                s1_tot = 2 * Wp * avg(X["s1"]) + X["s10"][0]; s1p = Hc * (2 * Wp * avg(XN["s1"]) + XN["s10"][0])
                lu_, lup, wb, wbp = X["lu"], d("lu"), X["w_bk"], d("w_bk")
                E21a = 2 * Wp * avg(msk * (np.real(np.conj(rbp) * wbp) + w**2 * np.real(np.conj(rb) * wb)))
                E21b = on * 2 * np.real(0.5 * np.conj(vp) * s1p + 0.5 * w**2 * np.conj(v) * s1_tot
                                        + 0.5 * np.conj(lp) * sp_ + 0.5 * w**2 * np.conj(l) * s_tot)
                if kind == "scalar":
                    E21c = Wp * avg(msk * (Xc(v[None, :], vp[None, :], X["w_up"], d("w_up")) + Xc(l[None, :], lp[None, :], X["r_up"], d("r_up"))
                                           + Xc(vu, vup, wb, wbp) + Xc(lu_, lup, rb, rbp)))
                    E21d = on * (2 * T**2 * 2 * np.real(np.conj(vp) * lp)
                                 + (2 * p**2 * T**2 + 0.5 * Tp**2 - 5 / 6 * q**2 * T**2) * 2 * np.real(np.conj(v) * l))
                else:
                    E21c = -Wp * avg(msk * gcp * T * np.real(v[None, :] * np.conj(X["w_up"]) + l[None, :] * np.conj(X["r_up"])
                                                             + vu * np.conj(wb) + lu_ * np.conj(rb)))
                    E21d = on * ((T**2 * p**2 / 6 - (q**2 * T**2 + Tp**2) / 48) * 2 * np.real(np.conj(v) * l)
                                 - T * Tp / 6 * np.real(v * np.conj(lp) + l * np.conj(vp)))
                R.update(E21a=E21a, E21b=E21b, E21c=E21c, E21d=E21d, dE21_k=E21a + E21b + E21c + E21d)
                # identity E21a + E21b = omega dn21_inst + (omega/2) tnorm21, and the source-counted particle part
                Al, Bl = A0(X["l"], XN["l"], pp, N)[0], B0(X["l"], XN["l"], pp, N)[0]
                s1N = 2 * Wp * avg(XN["s1"]) + XN["s10"][0]
                As1, Bs1 = A0(s1_tot, s1N, p, N), B0(s1_tot, s1N, p, N)
                Awb, Bwb = A0(wb, XN["w_bk"], pp, N), B0(wb, XN["w_bk"], pp, N)
                Bwu = B0(X["w_up"], XN["w_up"], p2, N)
                diag1 = 2 * np.real(np.conj(b0) * Bs1 + np.conj(Bl) * Bs)
                R["dn21_inst"] = on * (diag1 + 2 * Wp * avg(msk * 2 * np.real(np.conj(Bbk) * Bwb)))
                R["tnorm21"] = on * (2 * np.real(np.conj(a0) * As1 + np.conj(Al) * As) - diag1
                                     + 2 * Wp * avg(msk * 2 * np.real(np.conj(Abk) * Awb - np.conj(Bbk) * Bwb)))
                R["wdn21_src_inst"] = on * (w * diag1 + 2 * Wp * avg(msk * wu2 * 2 * np.real(np.conj(Bup) * Bwu)))
                R["dE21_vac_k"] = 0.5 * w * R["tnorm21"] + E21c + E21d
                R["dE21_hyb_k"] = R["wdn21_src_inst"] + R["dE21_vac_k"]
                R["contact2"] = np.array(fsw(N)[0] * (2 * D0 * D2a + 2 * Wp * np.abs(D1q)**2) / 8)
        if not full:                                                    # time-series records keep only the light arrays
            R = {k: v_ for k, v_ in R.items() if k not in ("K",)}
        return R

    # ---------------- integration ----------------
    N = min(Nin[n].min() for n in free)
    recN = sorted(set(round(float(x), 6) for x in list(P["nrec"]) + list(P.get("nser", []))))
    full_set = set(round(float(x), 6) for x in P["nrec"])
    rec = {}
    steps = 0; diag_hist = []
    AVG_WIN, AVG_EVERY = float(P.get("avg_window", 0.05)), int(P.get("avg_every", 4))
    # averaging window of record r: centred, [r - W/2, r + W/2] (avg_center=True; bias O(W^2/24) for a quantity that
    # grows like e^N), or the original backward window (r - W, r] (bias (1 - e^{-W})/W - 1 = -2.5% at W = 0.05 for the
    # late-time energy, which grows like a)
    AVG_CENTER = bool(P.get("avg_center", False)) and AVG_WIN > 0
    lo = (lambda r: r - AVG_WIN / 2) if AVG_CENTER else (lambda r: r - AVG_WIN)
    hi = (lambda r: r + AVG_WIN / 2) if AVG_CENTER else (lambda r: r)
    stops = sorted(set(recN) | {round(hi(r), 9) for r in recN})       # step-size alignment points
    N_stop = max([nend] + [hi(r) for r in recN])
    INSTANT = {"T", "omega", "n0", "K", "gain", "diag", "dn2", "dn2_src", "D"}     # kept at the record time
    acc_sum = {r: {} for r in recN}; acc_n = {r: 0 for r in recN}; acc_w = {r: 0.0 for r in recN}
    last_t, pending = {}, {}
    while N < N_stop - 1e-12:
        for n in free:
            new = (~act[n]) & (Nin[n] <= N + 1e-12)
            if new.any():
                v0, vN0 = adiabatic_init(mom[n], m, N, INIT)
                X[n] = np.where(new, v0, X[n]); XN[n] = np.where(new, vN0, XN[n]); act[n] |= new
        Hc, g = bg(N)
        wmax = max(max(np.max(np.where(act[n], om(mom[n], N)[0], 0)) for n in free), 1e-30)
        dN = min(dth * Hc / wmax, 0.01, N_stop - N)
        pend = [Nin[n][~act[n]] for n in free]
        pend = np.concatenate([x.ravel() for x in pend]) if any(x.size for x in pend) else np.array([])
        if pend.size and pend.min() > N: dN = min(dN, pend.min() - N)
        nxt = [r for r in stops if r > N + 1e-12]
        if nxt: dN = min(dN, nxt[0] - N)
        kX, kV, dg = [], [], None                                      # explicit Runge-Kutta step (Butcher tableau)
        for i, ci in enumerate(RK_C):
            if i == 0:
                Yi, YNi = X, XN
            else:
                Yi = {n: X[n] + dN * sum(a * kX[j][n] for j, a in enumerate(RK_A[i]) if a) for n in names}
                YNi = {n: XN[n] + dN * sum(a * kV[j][n] for j, a in enumerate(RK_A[i]) if a) for n in names}
            Ai, di = rhs(N + ci * dN, Yi, YNi)
            if i == 0: dg = di
            kX.append(YNi); kV.append(Ai)
        for n in names:
            X[n], XN[n] = (X[n] + dN * sum(b * kX[i][n] for i, b in enumerate(RK_B) if b),
                           XN[n] + dN * sum(b * kV[i][n] for i, b in enumerate(RK_B) if b))
        N += dN; steps += 1
        if dg and steps % 200 == 0: diag_hist.append([N] + [complex(dg.get(k, 0)).real for k in ("D0", "D1", "D2")])
        # time average of the in-in energy over the window of every record time r (see lo, hi above): the
        # instantaneous <T_00> contains pair-coherence terms oscillating at ~2 omega (as in quartic_hartree_run)
        open_r = [r for r in recN if r not in rec and lo(r) < N + 1e-12 and N <= hi(r) + 1e-9]
        at_stop = any(abs(N - s) < 1e-9 for s in stops)
        if open_r and (steps % AVG_EVERY == 0 or at_stop):
            Ri = record(N, False)
            for r in open_r:
                # centred mode: each sample weighted by the e-fold interval since the previous one (a true time
                # average; the step dN ~ e^{-2N} shrinks across the window, so equal sample weights would bias it)
                wt = (N - last_t.get(r, lo(r))) if AVG_CENTER else 1.0
                last_t[r] = N
                acc_n[r] += 1; acc_w[r] += wt
                for key, val in Ri.items():
                    acc_sum[r][key] = acc_sum[r].get(key, 0) + wt * np.asarray(val)
        for r in recN:
            if abs(N - r) < 1e-9 and r not in rec and r not in pending:
                pending[r] = record(N, r in full_set)                  # instantaneous values at the record time
            if r in pending and N >= hi(r) - 1e-9:                      # window closed: replace by the averages
                R = pending.pop(r)
                for key in list(R):
                    if key in acc_sum[r] and key not in INSTANT:
                        R[key + "_inst"] = R[key]; R[key] = acc_sum[r][key] / acc_w[r]
                R["n_avg"] = np.array(acc_n[r])
                rec[r] = R; acc_sum[r] = {}
        if not all(np.isfinite(X[n]).all() for n in names):
            raise FloatingPointError(f"non-finite values at N={N}")
    out = dict(params=json.dumps(P), p=p, mu=xg, wmu=WMU, p2=p2, Wp=Wp, steps=steps, seconds=time.time() - t0,
               diag=np.array(diag_hist), profile=np.array([NG[::50], TG[::50]]), Nrec=np.array(sorted(rec)))
    for i, Nr in enumerate(sorted(rec)):
        for key, val in rec[Nr].items():
            out[f"{key}__{i}"] = val
    np.savez(P["out"], **out)
    return out


# ------------------------------------------------------------------------------------------------------
# Local (adiabatic) subtraction terms of the metric kernels, through fourth adiabatic order (local_order = 4).
# Generated symbolically (frequency-domain expansion of the engine's own pair/target kernels, whose resonant poles
# cancel identically, plus the perturbed-FRW / homogeneous-oscillator adiabatic energy for the time derivatives);
# checked against the engine's high-p tails.  Arguments: k (comoving), q, M2 jet (M2, M2', ...), T jet (T, T', ...),
# derivatives with respect to conformal time.
# ------------------------------------------------------------------------------------------------------
def _conformal_derivs(fN, N):
    """conformal-time derivatives 0..4 of f(N) from its N-derivatives fN[0..4]"""
    Hc, g, gN, gNN = bg_full(N)
    f0, f1, f2, f3, f4 = fN
    return [f0, Hc * f1, Hc**2 * (f2 + g * f1), Hc**3 * (f3 + 3 * g * f2 + (2 * g**2 + gN) * f1),
            Hc**4 * (f4 + 6 * g * f3 + (11 * g**2 + 4 * gN) * f2 + (6 * g**3 + 7 * g * gN + gNN) * f1)]


def _M2jet(m, N):
    M2 = m**2 * np.exp(2 * N)
    return _conformal_derivs([M2, 2 * M2, 4 * M2, 8 * M2, 16 * M2], N)


def _temporal_scalar_1(k, Md, Td):
    """[psi^1] of the 2nd+4th-order FRW vacuum energy in the metric a^2[-(1+2psi)deta^2+(1-2psi)dx^2], homogeneous psi
    (= (1-2psi)^-2 E_sub of the conformally related FRW); complex-step derivative, exact to rounding"""
    h = 1e-30; e = 1j * h
    M0, M1, M2_, M3 = Md[:4]; T0, T1, T2, T3 = Td[:4]
    Q0 = M0 - 2 * e * M0 * T0; Q1 = M1 - 2 * e * (M1 * T0 + M0 * T1)
    Q2 = M2_ - 2 * e * (M2_ * T0 + 2 * M1 * T1 + M0 * T2); Q3 = M3 - 2 * e * (M3 * T0 + 3 * M2_ * T1 + 3 * M1 * T2 + M0 * T3)
    w = np.sqrt(k**2 + Q0)
    d1 = Q1 / (2 * w); d2 = Q2 / (2 * w) - Q1**2 / (4 * w**3); d3 = Q3 / (2 * w) - 3 * Q1 * Q2 / (4 * w**3) + 3 * Q1**3 / (8 * w**5)
    l0, l1, l2 = 1 - 2 * e * T0, -2 * e * T1, -2 * e * T2
    w1 = l0 * d1; w2 = l0**2 * d2 + l0 * l1 * d1; w3 = l0**3 * d3 + 3 * l0**2 * l1 * d2 + (l0 * l1**2 + l0**2 * l2) * d1
    E = w1**2 / (16 * w**3) - (45 * w1**4 - 40 * w * w1**2 * w2 - 4 * w**2 * w2**2 + 8 * w**2 * w1 * w3) / (256 * w**7)
    return np.imag((1 - 2 * e * T0)**-2 * E) / h


def _loc_scalar_2(k, q, Md, Td):
    M2_0, M2_1, M2_2, M2_3 = Md[:4]; T_0, T_1, T_2, T_3 = Td[:4]
    """scalar O(psi^2) mean"""
    x0 = k**2
    x1 = M2_0**7
    x2 = k**14
    x3 = k**12
    x4 = M2_0*x3
    x5 = M2_0**2
    x6 = k**10
    x7 = x5*x6
    x8 = M2_0**3
    x9 = k**8
    x10 = x8*x9
    x11 = M2_0**4
    x12 = k**6
    x13 = x11*x12
    x14 = M2_0**5
    x15 = k**4
    x16 = x14*x15
    x17 = M2_0**6
    x18 = x0*x17
    x19 = T_1*x17
    x20 = 3840*T_3
    x21 = T_2**2
    x22 = T_1**2
    x23 = T_0*T_2
    x24 = M2_1*T_0
    x25 = M2_2*x22
    x26 = M2_1*x9
    x27 = M2_0*x26
    x28 = T_1*T_2
    x29 = T_0*T_1
    x30 = M2_3*x29
    x31 = M2_0*x9
    x32 = T_1*x24
    x33 = M2_2*x32
    x34 = x5*x9
    x35 = M2_1**3
    x36 = x12*x8
    x37 = T_1*T_3
    x38 = 15360*x37
    x39 = T_0**2
    x40 = M2_3*x39
    x41 = M2_1*x40
    x42 = M2_1**2
    x43 = x23*x42
    x44 = x11*x15
    x45 = x0*x14
    x46 = q**2
    x47 = M2_0*x12
    x48 = M2_1**4*x39
    x49 = M2_2**2
    x50 = q**4
    x51 = x22*x46
    x52 = x39*x46
    x53 = x39*x50
    x54 = M2_0*x6
    x55 = x23*x46
    x56 = T_3*x24
    x57 = x12*x5
    x58 = 32640*x57
    x59 = M2_1*x28
    x60 = x15*x8
    x61 = 28800*x60
    x62 = 76800*x32
    x63 = x0*x11
    x64 = 5760*x63
    x65 = x15*x5
    return (1/122880)*(15360*M2_0*M2_1*T_0*T_1*x6 + 11520*M2_0*M2_2*T_0*T_2*x9 + 181440*M2_0*M2_2*x15*x39*x42 + 90720*M2_0*T_0*T_1*x15*x35 - 12600*M2_0*x0*x48 + 40320*M2_0*x12*x22*x42 + 34560*M2_0*x12*x39*x49 + 5120*M2_0*x22*x46*x6 + 38400*M2_0*x39*x42*x9 + 32640*M2_1*M2_2*T_0*T_1*x0*x8 + 69120*M2_1*M2_2*T_0*T_1*x15*x5 + 38400*M2_1*T_0*T_1*x5*x9 + 1920*M2_1*T_0*T_3*x14 + 3840*M2_1*T_1*T_2*x0*x11 + 7680*M2_1*T_1*T_2*x14 + 5760*M2_2*T_0*T_2*x0*x11 + 32640*M2_2*T_0*T_2*x12*x5 + 28800*M2_2*T_0*T_2*x15*x8 + 45360*M2_2*x0*x39*x42*x5 + 161280*M2_2*x12*x39*x42 - 1920*M2_2*x14*x23 - 42240*M2_2*x26*x29 + 25200*M2_2*x39*x42*x8 + 1920*M2_3*T_0*T_1*x14 + 174720*T_0*T_1*x12*x35 + 18480*T_0*T_1*x35*x8 + 42240*T_0*T_2*x0*x14*x46 + 1920*T_0*T_2*x0*x42*x8 + 37120*T_0*T_2*x11*x15*x46 + 14080*T_0*T_2*x17*x46 - 11520*T_0*T_3*x27 - T_1*x20*x34 + 24000*x0*x11*x39*x42 + 7680*x0*x14*x21 + 78080*x0*x14*x39*x50 + 38400*x0*x17*x22 + 77280*x0*x22*x42*x8 - 65520*x0*x29*x35*x5 + 11040*x0*x39*x49*x8 - 22080*x0*x41*x8 + 7680*x1*x22 - 35840*x1*x52 - 947200*x10*x52 + 76800*x11*x12*x22 + 11520*x11*x15*x21 + 33024*x11*x15*x39*x50 + 5040*x11*x22*x42 - 15360*x11*x33 + 3600*x11*x39*x49 - 7200*x11*x41 - 2880*x11*x43 + 7680*x12*x21*x8 + 17920*x12*x22*x46*x8 + 33600*x12*x39*x42*x5 - 1049600*x13*x52 + 76800*x14*x15*x22 - 5760*x14*x25 + 10560*x14*x39*x42 + 126000*x15*x22*x42*x5 + 24000*x15*x39*x42*x8 + 26640*x15*x39*x49*x5 - 214200*x15*x48 - 686080*x16*x52 + 1920*x17*x21 + 28800*x17*x39*x50 - 16640*x17*x51 - 243200*x18*x52 - x19*x20 - 23040*x19*x24 - 20480*x2*x52 + 1920*x21*x5*x9 - 13440*x22*x42*x9 + 21760*x22*x46*x5*x9 + 7680*x22*x5*x6 + 38400*x22*x8*x9 - 19200*x25*x31 - 63360*x25*x57 - 74880*x25*x60 - 36480*x25*x63 - 30720*x26*x40 - 19200*x27*x28 - 2048*x3*x53 - 11520*x30*x31 - x30*x58 - x30*x61 - x30*x64 - 21120*x33*x47 - 67456*x34*x53 - 15360*x34*x55 - x36*x38 - 67328*x36*x53 - 1280*x36*x55 - 23040*x37*x44 - x38*x45 + 15360*x39*x42*x6 + 15360*x39*x49*x9 - 153600*x4*x52 - 69120*x41*x47 - 53280*x41*x65 - 92160*x43*x47 - 37440*x43*x65 - 49920*x43*x9 - 25600*x44*x51 - x44*x62 - 43520*x45*x51 - x45*x62 - 23625*x48*x5 - 509440*x52*x7 - 18432*x53*x54 - 5120*x54*x55 - x56*x58 - x56*x61 - x56*x64 - 49920*x57*x59 - 34560*x59*x60)/(np.sqrt(M2_0 + x0)*(x1 + 35*x10 + 35*x13 + 21*x16 + 7*x18 + x2 + 7*x4 + 21*x7))


def _loc_scalar_1_flat(k, q, M2_0, M2_1, M2_2, M2_3, T_0, T_1, T_2, T_3):
    """scalar O(psi): static q^2, q^4 and mixed q^2 T'' local terms"""
    x0 = q**2
    x1 = k**2
    x2 = M2_0**2
    x3 = k**4
    x4 = M2_0*x1
    x5 = T_0/(x2 + x3 + 2*x4)
    x6 = M2_0**3
    x7 = k**6
    x8 = M2_0*x3
    x9 = x1*x2
    x10 = T_0*x0/(M2_0**4 + 4*M2_0*x7 + k**8 + 4*x1*x6 + 6*x2*x3)
    return (1/1440)*x0*(15*T_2*x2/(x6 + x7 + 3*x8 + 3*x9) + 20*x10*x6 - 8*x10*x7 - 64*x10*x8 - 225*x10*x9 + 300*x2*x5 + 240*x3*x5 + 600*x4*x5)/np.sqrt(M2_0 + x1)

def _loc_tensor_1(k, q, Md, Td):
    M2_0, M2_1, M2_2, M2_3, M2_4 = Md[:5]; T_0, T_1, T_2, T_3, T_4 = Td[:5]
    """tensor O(h) quadrupole: local terms (flat q-expansion + homogeneous background)"""
    x0 = k**2
    x1 = M2_0 + x0
    x2 = np.sqrt(x1)
    x3 = 1/x2
    x4 = q**2
    x5 = T_0*x4
    x6 = x3*x5
    x7 = x1**(-3/2)
    x8 = T_0*q**4
    x9 = 2048*x8
    x10 = T_2*x7
    x11 = x0*x2
    x12 = M2_0**2
    x13 = k**4
    x14 = M2_0*x0
    x15 = 1/(x12 + x13 + 2*x14)
    x16 = T_2*x15
    x17 = 10752*x16
    x18 = x13*x3
    x19 = x3*x8
    x20 = x14*x3
    x21 = x3*x4
    x22 = x0*x3
    x23 = T_1*x22
    x24 = M2_1*x23
    x25 = M2_0**3
    x26 = k**6
    x27 = M2_0*x13
    x28 = x0*x12
    x29 = 1/(x25 + x26 + 3*x27 + 3*x28)
    x30 = x18*x29
    x31 = 5376*T_4
    x32 = x20*x29
    x33 = M2_1*T_1
    x34 = x11*x33
    x35 = 5376*x33
    x36 = M2_1*T_3
    x37 = T_0*x11
    x38 = 16128*M2_2
    x39 = T_2*x22
    x40 = 2688*x29
    x41 = 8064*M2_4
    x42 = T_0*x3
    x43 = x29*x42
    x44 = x0*x43
    x45 = x12*x29
    x46 = M2_1**2
    x47 = x13*x2
    x48 = M2_0**4
    x49 = k**8
    x50 = M2_0*x26
    x51 = x12*x13
    x52 = x0*x25
    x53 = 1/(x48 + x49 + 4*x50 + 6*x51 + 4*x52)
    x54 = 2688*x53
    x55 = T_4*x54
    x56 = x26*x3
    x57 = x14*x2
    x58 = x27*x3
    x59 = 13440*x36*x53
    x60 = M2_2*T_2
    x61 = 32256*x18
    x62 = x53*x60
    x63 = M2_3*T_1
    x64 = 26880*x53*x63
    x65 = M2_4*x53
    x66 = T_0*x18
    x67 = 6720*x65
    x68 = x14*x42
    x69 = x0*x53
    x70 = M2_1*M2_3
    x71 = x42*x70
    x72 = x28*x3
    x73 = x46*x53
    x74 = 20160*x73
    x75 = M2_2**2
    x76 = x42*x75
    x77 = M2_0**5
    x78 = k**10
    x79 = M2_0*x49
    x80 = x12*x26
    x81 = x13*x25
    x82 = x0*x48
    x83 = 1/(x77 + x78 + 5*x79 + 10*x80 + 10*x81 + 5*x82)
    x84 = x47*x83
    x85 = 5376*x36
    x86 = x56*x83
    x87 = 25536*x60
    x88 = 9408*x60
    x89 = 18816*x63
    x90 = 8064*x63
    x91 = T_0*x83
    x92 = x47*x91
    x93 = 1344*M2_4
    x94 = x56*x91
    x95 = x57*x83
    x96 = x58*x83
    x97 = x57*x91
    x98 = x42*x83
    x99 = x27*x98
    x100 = x18*x83
    x101 = M2_2*x33
    x102 = 26208*x101
    x103 = x70*x91
    x104 = 8064*x46
    x105 = T_2*x46
    x106 = 3696*x105
    x107 = M2_1**3*T_1
    x108 = x75*x91
    x109 = x20*x83
    x110 = x14*x83
    x111 = x72*x83
    x112 = x28*x98
    x113 = M2_2*x46
    x114 = M2_0**6
    x115 = k**12
    x116 = M2_0*x78
    x117 = x12*x49
    x118 = x25*x26
    x119 = x13*x48
    x120 = x0*x77
    x121 = 1/(x114 + x115 + 6*x116 + 15*x117 + 20*x118 + 15*x119 + 6*x120)
    x122 = x101*x121
    x123 = 61824*x122
    x124 = 13440*x121
    x125 = x101*x124
    x126 = T_0*x47
    x127 = x121*x70
    x128 = 17472*x127
    x129 = T_0*x56
    x130 = 1344*x129
    x131 = x105*x121
    x132 = 20832*x131
    x133 = 3360*x131
    x134 = x11*x121
    x135 = x121*x18
    x136 = 34776*x107
    x137 = M2_1**4
    x138 = x137*x42
    x139 = x121*x75
    x140 = 16800*x139
    x141 = T_0*x2
    x142 = x14*x141
    x143 = x121*x27
    x144 = T_0*x113
    x145 = 1344*x28
    x146 = M2_0**7
    x147 = k**14
    x148 = 1/(7*M2_0*x115 + 7*x0*x114 + 21*x12*x78 + 21*x13*x77 + x146 + x147 + 35*x25*x49 + 35*x26*x48)
    x149 = x148*x49
    x150 = x141*x41
    x151 = x42*x78
    x152 = M2_4*x148
    x153 = 10752*x152
    x154 = x141*x50
    x155 = 24192*x152
    x156 = x152*x42
    x157 = 43008*x156
    x158 = x141*x26
    x159 = x148*x70
    x160 = 40320*x159
    x161 = 53760*x71
    x162 = x107*x148
    x163 = 38976*x162
    x164 = 16464*x162
    x165 = x137*x148
    x166 = x141*x27
    x167 = 161280*x148*x71
    x168 = x14*x148
    x169 = x141*x51
    x170 = x148*x52
    x171 = x42*x82
    x172 = x113*x148
    x173 = x113*x141
    x174 = x172*x42
    x175 = x141*x28
    x176 = M2_0**8
    x177 = k**16
    x178 = 1/(8*M2_0*x147 + 8*x0*x146 + 28*x114*x13 + 28*x115*x12 + x176 + x177 + 56*x25*x78 + 56*x26*x77 + 70*x48*x49)
    x179 = x178*x78
    x180 = x141*x93
    x181 = x115*x42
    x182 = x178*x93
    x183 = x141*x79
    x184 = M2_4*x178
    x185 = 5376*x184
    x186 = x116*x42
    x187 = 6720*x184
    x188 = x141*x49
    x189 = x178*x188
    x190 = 6720*x70
    x191 = 25536*x71
    x192 = 34272*x137*x178
    x193 = 13944*x178
    x194 = 26208*x75
    x195 = 59136*x76
    x196 = x154*x178
    x197 = 20160*x70
    x198 = 102144*x178*x71
    x199 = x178*x27
    x200 = 78624*x75
    x201 = x178*x76
    x202 = 236544*x201
    x203 = x178*x80
    x204 = x117*x42
    x205 = 13440*x184
    x206 = x141*x81
    x207 = x118*x42
    x208 = x178*x82
    x209 = x119*x42
    x210 = x120*x42
    x211 = x113*x178
    x212 = x211*x42
    x213 = 254016*x212
    x214 = 762048*x212
    x215 = x169*x178
    x216 = x141*x52
    x217 = x178*x216
    x218 = M2_0**9
    x219 = k**18
    x220 = 1/(9*M2_0*x177 + 9*x0*x176 + 84*x114*x26 + 84*x115*x25 + 36*x12*x147 + 36*x13*x146 + x218 + x219 + 126*x48*x78 + 126*x49*x77)
    x221 = x141*x78
    x222 = x220*x221
    x223 = 18816*x70
    x224 = 13440*x220
    x225 = x137*x220
    x226 = 87318*x225
    x227 = x220*x49
    x228 = 197736*x138
    x229 = 18480*x75
    x230 = x220*x76
    x231 = 14448*x230
    x232 = x220*x70
    x233 = 75264*x232
    x234 = x220*x71
    x235 = 67200*x234
    x236 = x220*x50
    x237 = 593208*x138
    x238 = x220*x75
    x239 = 73920*x238
    x240 = 72240*x230
    x241 = 113400*x173
    x242 = x113*x220
    x243 = 40488*x242
    x244 = 340200*x173
    x245 = x242*x42
    x246 = 161952*x245
    x247 = x141*x80
    x248 = 134400*x234
    x249 = x141*x82
    x250 = x220*x249
    x251 = x220*x51
    x252 = 144480*x230
    x253 = x220*x52
    x254 = M2_0**10
    x255 = k**20
    x256 = 1/(10*M2_0*x219 + 10*x0*x218 + 210*x114*x49 + 210*x115*x48 + 45*x12*x177 + 45*x13*x176 + 120*x146*x26 + 120*x147*x25 + x254 + x255 + 252*x77*x78)
    x257 = x137*x256
    x258 = 56385*x257
    x259 = x256*x78
    x260 = 7287*x138
    x261 = 169155*x257
    x262 = x138*x256
    x263 = 29148*x262
    x264 = 46704*x173
    x265 = x113*x256
    x266 = 34608*x265
    x267 = x173*x256
    x268 = 186816*x267
    x269 = 173040*x265
    x270 = x256*x82
    x271 = 346080*x265
    x272 = 1/(M2_0**11 + 11*M2_0*x255 + k**22 + 11*x0*x254 + 462*x114*x78 + 462*x115*x77 + 55*x12*x219 + 55*x13*x218 + 330*x146*x49 + 330*x147*x48 + 165*x176*x26 + 165*x177*x25)
    x273 = x137*x272
    x274 = 18732*x273
    x275 = x138*x272
    x276 = 12684*x275
    x277 = 74928*x273
    x278 = 63420*x275
    x279 = 126840*x275
    return (1/1008)*M2_0*x15*x19 - 1/5040*M2_0*x16*x21 + (1/1260)*M2_0*x5*x7 - 1/8*M2_2*x24*x53 + (139/1920)*M2_2*x34*x83 + (1/1290240)*M2_2*x39*x40 + (1/1290240)*M2_3*x23*x40 - 1/480*M2_4*x99 - 1/1290240*T_0*x30*x38 - 1/1680*T_2*x21*x45 + (1/60)*x0*x10 + (31/384)*x0*x113*x98 - 163/1536*x0*x121*x138 + (1/1260)*x10*x4 - 1/1290240*x100*x102 - 1/1290240*x100*x106 - 1/1290240*x102*x109 - 11/960*x103*x11 - 13/960*x103*x18 - 1/1290240*x104*x112 - 1/1290240*x104*x92 - 1/1290240*x104*x94 - 1/1290240*x104*x97 + (169/3840)*x105*x11*x83 - 1/1290240*x106*x109 - 923/7680*x107*x134 + (737/3840)*x107*x22*x83 - 13/1920*x108*x11 - 1/1290240*x108*x61 - 1/1290240*x11*x17 - 1/1290240*x11*x54*x60 - 1/1290240*x11*x59 - 13/960*x110*x71 - 1/40*x110*x76 + (1/1290240)*x111*x85 - 1/1290240*x111*x88 - 1/1290240*x111*x90 - 1/1290240*x112*x93 + (67/1920)*x113*x121*x68 + (1/1290240)*x115*x231 + (1/1290240)*x115*x276 + (1/1290240)*x116*x235 + (1/1290240)*x116*x240 + (1/1290240)*x116*x278 + (1/1290240)*x117*x248 + (1/1290240)*x117*x252 + (1/1290240)*x117*x279 + (1/1290240)*x118*x248 + (1/1290240)*x118*x252 + (1/1290240)*x118*x279 + (1/1290240)*x119*x235 + (1/1290240)*x119*x240 + (1/1290240)*x119*x278 + (1/420)*x12*x15*x6 + (1/1290240)*x120*x224*x71 + (1/1290240)*x120*x231 + (1/1290240)*x120*x276 - 1/1290240*x121*x136*x20 + (1/1290240)*x121*x145*x71 + (1/48)*x122*x58 + (1/1290240)*x123*x47 + (1/1290240)*x123*x57 + (1/1290240)*x124*x28*x76 + (1/1290240)*x125*x56 + (1/1290240)*x125*x72 + (1/1290240)*x126*x128 + (1/1290240)*x126*x140 - 49/960*x126*x172 + (1/1290240)*x126*x192 + (1/1290240)*x127*x130 + (1/1290240)*x128*x142 - 1/1290240*x129*x137*x193 + (1/96)*x129*x139 + (1/1290240)*x130*x172 - 1/192*x131*x58 + (1/1290240)*x132*x47 + (1/1290240)*x132*x57 - 1/1290240*x133*x56 - 1/1290240*x133*x72 - 21/640*x134*x144 - 1/1290240*x135*x136 + (67/1920)*x135*x144 + (15/1024)*x138*x168 - 1/1290240*x138*x193*x28 - 83/3840*x138*x199 - 1/1290240*x14*x38*x43 + (1/1290240)*x140*x142 + (1/1290240)*x142*x192 + (1/480)*x143*x71 + (1/48)*x143*x76 + (1/1290240)*x145*x174 - 1/1290240*x149*x150 - 1/1290240*x149*x161 - 1/30*x15*x24 - 1/1290240*x150*x170 - 1/1290240*x150*x203 + (1/1290240)*x151*x153 + (1/1290240)*x151*x243 + (1/1290240)*x153*x171 - 1/1290240*x154*x155 + (1/1290240)*x154*x261 - 1/1290240*x155*x169 + (1/20)*x156*x80 + (1/1290240)*x157*x79 + (1/1290240)*x157*x81 + (1/1290240)*x158*x160 - 99/1280*x158*x211 + (1/1290240)*x158*x226 + (1/16)*x159*x166 + (1/1290240)*x160*x175 - 1/1290240*x161*x170 + (49/1920)*x162*x58 - 1/1290240*x163*x47 - 1/1290240*x163*x57 + (1/1290240)*x164*x56 + (1/1290240)*x164*x72 + (1883/30720)*x165*x37 + (15/1024)*x165*x66 + (693/5120)*x166*x225 - 1/1290240*x167*x50 - 1/1290240*x167*x51 - 49/960*x168*x173 + (1/1290240)*x169*x261 - 1/1290240*x17*x18 - 1/1290240*x17*x20 + (1/1290240)*x171*x243 - 99/1280*x173*x178*x28 - 99/640*x173*x199 + (1/480)*x174*x27 + (1/1290240)*x175*x226 - 1/1290240*x179*x180 - 1/1290240*x179*x191 - 1/1290240*x179*x195 - 1/1290240*x18*x59 + (1/1290240)*x18*x64 - 1/1290240*x180*x208 - 1/1290240*x181*x182 + (1/1290240)*x181*x224*x70 - 1/1290240*x181*x266 - 1/1290240*x182*x210 - 1/1290240*x183*x185 + (1/1290240)*x183*x233 + (1/1290240)*x183*x239 + (1/1290240)*x183*x277 - 1/1290240*x185*x206 - 1/1290240*x186*x187 - 1/1290240*x186*x269 - 1/1290240*x187*x209 + (1/1290240)*x188*x258 + (1/1290240)*x189*x190 + (1/1290240)*x189*x194 - 1/240*x19*x25*x53 + (1/1290240)*x190*x217 - 1/1290240*x191*x208 + (1/1290240)*x194*x217 - 1/1290240*x195*x208 + (1/1290240)*x196*x197 + (1/1290240)*x196*x200 + (1/1290240)*x197*x215 - 1/1290240*x198*x79 - 1/1290240*x198*x81 - 1/1290240*x20*x59 + (1/40)*x20*x62 + (1/1290240)*x20*x64 + (1/1290240)*x200*x215 - 11/40*x201*x80 - 1/1290240*x202*x79 - 1/1290240*x202*x81 - 19/160*x203*x71 - 1/1290240*x204*x205 - 1/1290240*x204*x271 - 1/1290240*x205*x207 + (1/1290240)*x206*x233 + (1/1290240)*x206*x239 + (1/1290240)*x206*x277 - 1/1290240*x207*x271 - 1/1290240*x209*x269 - 1/1290240*x210*x266 + (1/1290240)*x213*x49 + (1/1290240)*x213*x52 + (1/1290240)*x214*x50 + (1/1290240)*x214*x51 + (1/1290240)*x216*x258 + (11/480)*x22*x29*x36 + (1/1290240)*x221*x274 + (1/1290240)*x222*x223 + (1/1290240)*x222*x229 + (1/1290240)*x223*x250 - 1/1290240*x227*x228 - 1/1290240*x227*x241 - 1/1290240*x228*x253 + (1/1290240)*x229*x250 + (7/80)*x232*x247 - 1/1290240*x236*x237 - 1/1290240*x236*x244 - 1/1290240*x237*x251 + (11/128)*x238*x247 - 1/1290240*x241*x253 - 1/1290240*x244*x251 + (241/1280)*x245*x80 + (1/1290240)*x246*x79 + (1/1290240)*x246*x81 + (223/2560)*x247*x273 + (1/1290240)*x249*x274 - 1/1290240*x259*x260 - 1/1290240*x259*x264 - 1/1290240*x260*x270 - 347/10240*x262*x80 - 1/1290240*x263*x79 - 1/1290240*x263*x81 - 1/1290240*x264*x270 - 139/640*x267*x80 - 1/1290240*x268*x79 - 1/1290240*x268*x81 + (1/48)*x29*x34 + (1/1290240)*x29*x37*x38 + (1/1290240)*x3*x45*x9 - 1/1290240*x30*x31 + (1/1290240)*x30*x35 - 1/1290240*x31*x32 + (1/1290240)*x31*x53*x58 + (1/1290240)*x32*x35 + (1/120)*x36*x96 + (1/320)*x37*x65 - 7/320*x37*x73 - 119/1920*x39*x73 - 1/1290240*x41*x44 + (7/240)*x44*x46 - 1/80*x46*x99 + (1/1290240)*x47*x55 + (1/1290240)*x55*x56 + (1/1290240)*x55*x57 + (1/1290240)*x55*x72 - 1/315*x6 - 7/480*x60*x96 + (1/1290240)*x61*x62 - 1/80*x63*x96 + (1/1290240)*x66*x67 + (1/1290240)*x66*x74 + (1/1290240)*x67*x68 + (1/1290240)*x68*x74 + (1/320)*x69*x71 + (23/1920)*x69*x76 + (1/1290240)*x7*x9 - 1/1290240*x84*x85 - 1/1290240*x84*x87 - 1/1290240*x84*x89 + (1/1290240)*x85*x86 - 1/1290240*x85*x95 - 1/1290240*x86*x88 - 1/1290240*x86*x90 - 1/1290240*x87*x95 - 1/1290240*x89*x95 - 1/1290240*x92*x93 - 1/1290240*x93*x94 - 1/1290240*x93*x97


def _loc_tensor_2(k, q, Md, Td):
    M2_0, M2_1, M2_2, M2_3, M2_4 = Md[:5]; T_0, T_1, T_2, T_3, T_4 = Td[:5]
    """tensor O(h^2) mean: local terms (flat q, Omega expansion + homogeneous background)"""
    x0 = k**2
    x1 = M2_0 + x0
    x2 = x1**(-3/2)
    x3 = M2_1*T_1
    x4 = (1/96)*T_0
    x5 = x3*x4
    x6 = T_0**2
    x7 = (1/64)*x6
    x8 = np.sqrt(x1)
    x9 = 1/x8
    x10 = M2_0**2
    x11 = k**4
    x12 = M2_0*x0
    x13 = 1/(x10 + x11 + 2*x12)
    x14 = x13*x9
    x15 = M2_1*T_3
    x16 = x15*x4
    x17 = T_2*x3
    x18 = (1/24)*x14
    x19 = (1/192)*T_2
    x20 = M2_2*x14
    x21 = T_0*T_1
    x22 = M2_3*x21
    x23 = x6*x8
    x24 = (1/128)*M2_2
    x25 = T_1**2
    x26 = M2_2*x25
    x27 = M2_0*x24
    x28 = M2_1**2
    x29 = T_0*x3
    x30 = (1/384)*x0
    x31 = M2_0**3
    x32 = k**6
    x33 = M2_0*x11
    x34 = x0*x10
    x35 = 1/(x31 + x32 + 3*x33 + 3*x34)
    x36 = x35*x8
    x37 = (1/768)*T_0
    x38 = (1/256)*M2_0
    x39 = T_0*x9
    x40 = x15*x39
    x41 = x17*x9
    x42 = T_0*T_2
    x43 = x42*x9
    x44 = x3*x39
    x45 = x35*x44
    x46 = x26*x9
    x47 = x35*x46
    x48 = x35*x6
    x49 = x48*x9
    x50 = M2_4*x49
    x51 = M2_1*M2_3
    x52 = (1/1536)*x35
    x53 = x28*x9
    x54 = x42*x53
    x55 = x28*x8
    x56 = x25*x53
    x57 = M2_2**2
    x58 = (1/1024)*x57
    x59 = x0*x35
    x60 = M2_2*x42
    x61 = x60*x9
    x62 = x22*x9
    x63 = x28*x49
    x64 = (5/192)*x11
    x65 = M2_2*x64
    x66 = M2_2*x12
    x67 = M2_0**4
    x68 = k**8
    x69 = M2_0*x32
    x70 = x10*x11
    x71 = x0*x31
    x72 = 1/(x67 + x68 + 4*x69 + 6*x70 + 4*x71)
    x73 = M2_0*x72
    x74 = x73*x8
    x75 = T_0*x15
    x76 = (1/192)*x75
    x77 = x17*x8
    x78 = x29*x8
    x79 = x26*x8
    x80 = (5/512)*x73
    x81 = (1/512)*x72
    x82 = M2_4*x23
    x83 = x81*x82
    x84 = x23*x72
    x85 = (5/512)*x51
    x86 = T_2*x37
    x87 = x72*x9
    x88 = M2_1**3
    x89 = x21*x88
    x90 = x25*x55
    x91 = x6*x9
    x92 = x51*x91
    x93 = x10*x87
    x94 = M2_2*x86
    x95 = x11*x72
    x96 = x32*x72
    x97 = x0*x72
    x98 = x75*x8
    x99 = (1/48)*x98
    x100 = x11*x87
    x101 = x57*x91
    x102 = x10*x81
    x103 = M2_4*x91
    x104 = x53*x6
    x105 = x104*x72
    x106 = M2_2*x91
    x107 = (7/1536)*x79
    x108 = x12*x72
    x109 = x29*x87
    x110 = x12*x87
    x111 = M2_2*x44
    x112 = x106*x72
    x113 = x55*x6
    x114 = M2_0**5
    x115 = k**10
    x116 = M2_0*x68
    x117 = x10*x32
    x118 = x11*x31
    x119 = x0*x67
    x120 = 1/(x114 + x115 + 5*x116 + 10*x117 + 10*x118 + 5*x119)
    x121 = x8*x89
    x122 = x120*x9
    x123 = x122*x68
    x124 = T_1*T_3
    x125 = (1/960)*x124
    x126 = x122*x25
    x127 = (1/96)*x114
    x128 = M2_1**4*x6
    x129 = (1/120)*x115
    x130 = T_2**2
    x131 = (1/1920)*x130
    x132 = x23*x51
    x133 = x120*x132
    x134 = x42*x55
    x135 = x120*x134
    x136 = x122*x69
    x137 = (1/160)*x120*x78
    x138 = x11*x120
    x139 = x122*x32
    x140 = M2_2*x120
    x141 = x140*x42*x8
    x142 = x22*x8
    x143 = (1/120)*x22
    x144 = x120*x90
    x145 = x23*x57
    x146 = x120*x145
    x147 = (1/960)*x136
    x148 = (9/640)*x140*x23
    x149 = x123*x6
    x150 = (13/1920)*M2_2
    x151 = (1/384)*x82
    x152 = x139*x6
    x153 = (7/768)*M2_4
    x154 = x12*x120
    x155 = x122*x33
    x156 = (1/64)*x142
    x157 = x122*x29
    x158 = x122*x6
    x159 = x158*x28
    x160 = M2_2*x159
    x161 = x122*x26
    x162 = M2_4*x158
    x163 = (1/256)*x10
    x164 = x158*x51
    x165 = x122*x28*x42
    x166 = x122*x70
    x167 = x126*x28
    x168 = (5/2048)*x10
    x169 = x158*x57
    x170 = (5/96)*x119
    x171 = q**2
    x172 = x158*x171
    x173 = q**4
    x174 = x126*x171
    x175 = x171*x42
    x176 = x122*x34
    x177 = (17/384)*x34
    x178 = x158*x173
    x179 = (1/240)*x69
    x180 = M2_0**6
    x181 = k**12
    x182 = M2_0*x115
    x183 = x10*x68
    x184 = x31*x32
    x185 = x11*x67
    x186 = x0*x114
    x187 = 1/(x180 + x181 + 6*x182 + 15*x183 + 20*x184 + 15*x185 + 6*x186)
    x188 = x128*x8
    x189 = x121*x187
    x190 = x187*x68
    x191 = (1/120)*x78
    x192 = x187*x44
    x193 = x187*x32
    x194 = (1/64)*x98
    x195 = x60*x8
    x196 = M2_2*x68
    x197 = x128*x9
    x198 = x187*x197
    x199 = (1/480)*x187
    x200 = x199*x23
    x201 = x106*x199
    x202 = (7/3840)*x82
    x203 = x187*x69
    x204 = (1/40)*x192
    x205 = x187*x33
    x206 = (7/192)*x205
    x207 = (5/96)*x40
    x208 = x187*x41
    x209 = x11*x187
    x210 = M2_2*x78
    x211 = M2_2*x113
    x212 = x187*x211
    x213 = (1/160)*x106*x187
    x214 = x187*x46
    x215 = x103*x187
    x216 = (1/512)*x187
    x217 = x89*x9
    x218 = x12*x187
    x219 = M2_2*x192
    x220 = (3/1024)*x187
    x221 = x187*x70
    x222 = x187*x34
    x223 = (1/32)*x187
    x224 = x223*x34
    x225 = (1/120)*x118
    x226 = x187*x71
    x227 = (1/48)*x226
    x228 = (1/16)*x71
    x229 = x223*x71
    x230 = x113*x187
    x231 = x104*x187
    x232 = M2_2*x104
    x233 = M2_0**7
    x234 = k**14
    x235 = M2_0*x181
    x236 = x10*x115
    x237 = x31*x68
    x238 = x32*x67
    x239 = x11*x114
    x240 = x0*x180
    x241 = 1/(x233 + x234 + 7*x235 + 21*x236 + 35*x237 + 35*x238 + 21*x239 + 7*x240)
    x242 = x241*x68
    x243 = x115*x241
    x244 = (7/240)*x40
    x245 = (1/40)*x41
    x246 = (1/80)*x243
    x247 = x241*x9
    x248 = x188*x241
    x249 = (31/3840)*x82
    x250 = (11/1280)*x103
    x251 = x241*x69
    x252 = x116*x241
    x253 = (7/80)*x40
    x254 = (3/40)*x41
    x255 = (3/80)*x61
    x256 = (1/40)*x62
    x257 = x241*x32
    x258 = (3/80)*x46
    x259 = (33/1280)*x103
    x260 = x121*x241
    x261 = x241*x33
    x262 = x197*x241
    x263 = (1/1280)*x113
    x264 = (61/3840)*x104
    x265 = x241*x70
    x266 = x117*x241
    x267 = x118*x241
    x268 = (1/80)*x267
    x269 = (61/1280)*x104
    x270 = x241*x56
    x271 = x101*x241
    x272 = x211*x241
    x273 = x241*x34
    x274 = x241*x71
    x275 = (1/16)*x273
    x276 = (1/48)*x92
    x277 = x57*x7
    x278 = M2_0**8
    x279 = k**16
    x280 = M2_0*x234
    x281 = x10*x181
    x282 = x115*x31
    x283 = x67*x68
    x284 = x114*x32
    x285 = x11*x180
    x286 = x0*x233
    x287 = 1/(x278 + x279 + 8*x280 + 28*x281 + 56*x282 + 70*x283 + 56*x284 + 28*x285 + 8*x286)
    x288 = x115*x287
    x289 = x288*x8
    x290 = (13/1920)*x75
    x291 = x181*x287
    x292 = (11/1920)*x291
    x293 = (1/80)*x77
    x294 = (1/80)*x41
    x295 = (17/640)*x60
    x296 = (19/480)*x61
    x297 = (19/960)*x142
    x298 = (7/240)*x62
    x299 = (1/160)*x79
    x300 = (1/160)*x46
    x301 = x116*x287
    x302 = (13/640)*x98
    x303 = x182*x287
    x304 = (11/480)*x40
    x305 = (3/80)*x77
    x306 = (1/20)*x41
    x307 = (51/640)*x195
    x308 = (19/120)*x61
    x309 = (19/320)*x142
    x310 = (7/60)*x62
    x311 = x287*x68
    x312 = (107/1280)*x210
    x313 = (23/160)*x111
    x314 = (3/160)*x79
    x315 = (1/40)*x46
    x316 = x103*x287
    x317 = (107/2560)*x134
    x318 = (31/320)*x54
    x319 = x121*x287
    x320 = x287*x69
    x321 = (69/160)*x111
    x322 = x113*x287
    x323 = (1/120)*x104
    x324 = (1/320)*x90
    x325 = (1/640)*x56
    x326 = x188*x287
    x327 = x197*x287
    x328 = (313/15360)*x145
    x329 = (761/15360)*x101
    x330 = (93/320)*x54
    x331 = x117*x287
    x332 = x183*x287
    x333 = x118*x287
    x334 = x333*x8
    x335 = x184*x287
    x336 = x185*x287
    x337 = (1/40)*x322
    x338 = (1/30)*x104
    x339 = (3/640)*x56
    x340 = (761/5120)*x101
    x341 = x153*x23
    x342 = x119*x287
    x343 = M2_2*x322
    x344 = x287*x70
    x345 = x287*x71
    x346 = M2_0**9
    x347 = k**18
    x348 = M2_0*x279
    x349 = x10*x234
    x350 = x181*x31
    x351 = x115*x67
    x352 = x114*x68
    x353 = x180*x32
    x354 = x11*x233
    x355 = 1/(9*x0*x278 + x346 + x347 + 9*x348 + 36*x349 + 84*x350 + 126*x351 + 126*x352 + 84*x353 + 36*x354)
    x356 = x181*x355
    x357 = x356*x8
    x358 = (7/960)*x75
    x359 = x234*x355
    x360 = x359*x9
    x361 = (11/768)*x60
    x362 = (43/3840)*x60
    x363 = (19/1920)*x142
    x364 = (1/128)*x22
    x365 = (7/2560)*x82
    x366 = (7/7680)*x103
    x367 = x182*x355
    x368 = (7/240)*x98
    x369 = x235*x355
    x370 = (5/192)*x40
    x371 = (11/192)*x195
    x372 = x355*x61
    x373 = (43/768)*x372
    x374 = (19/480)*x142
    x375 = (5/128)*x62
    x376 = x115*x355
    x377 = (43/512)*x210
    x378 = (1399/7680)*x111
    x379 = x355*x82
    x380 = (7/640)*x379
    x381 = x103*x355
    x382 = (7/1536)*x381
    x383 = (1/512)*x132
    x384 = (45/1024)*x134
    x385 = (1013/15360)*x54
    x386 = x121*x355
    x387 = (145/1024)*x386
    x388 = x116*x355
    x389 = (129/512)*x210
    x390 = (1399/1920)*x111
    x391 = (193/3840)*x56
    x392 = x188*x355
    x393 = x197*x355
    x394 = (3/512)*x132
    x395 = (135/1024)*x134
    x396 = x355*x54
    x397 = (1013/3840)*x396
    x398 = x183*x355
    x399 = x236*x355
    x400 = (43/384)*x372
    x401 = (5/64)*x62
    x402 = x184*x355
    x403 = x237*x355
    x404 = x185*x355
    x405 = x404*x8
    x406 = x238*x355
    x407 = x239*x355
    x408 = x407*x9
    x409 = x355*x56
    x410 = (193/960)*x409
    x411 = x145*x355
    x412 = x101*x355
    x413 = (7/768)*x381
    x414 = x117*x355
    x415 = x118*x355
    x416 = x211*x355
    x417 = x232*x355
    x418 = x217*x355
    x419 = M2_0**10
    x420 = k**20
    x421 = M2_0*x347
    x422 = x10*x279
    x423 = x234*x31
    x424 = x181*x67
    x425 = x114*x115
    x426 = x180*x68
    x427 = x233*x32
    x428 = x11*x278
    x429 = 1/(10*x0*x346 + x419 + x420 + 10*x421 + 45*x422 + 120*x423 + 210*x424 + 252*x425 + 210*x426 + 120*x427 + 45*x428)
    x430 = x103*x429
    x431 = (7/320)*x430
    x432 = x181*x429
    x433 = (173/1920)*x210
    x434 = x234*x429
    x435 = (31/384)*x111
    x436 = (21/160)*x430
    x437 = (331/7680)*x134
    x438 = (59/1536)*x54
    x439 = x115*x429
    x440 = (747/10240)*x121
    x441 = (283/6144)*x217
    x442 = x210*x429
    x443 = (173/480)*x442
    x444 = x111*x429
    x445 = (155/384)*x444
    x446 = (1/40)*x429
    x447 = x446*x90
    x448 = x446*x56
    x449 = x188*x429
    x450 = x145*x429
    x451 = x101*x429
    x452 = x132*x429
    x453 = x429*x92
    x454 = x134*x429
    x455 = (331/1920)*x454
    x456 = x429*x54
    x457 = (295/1536)*x456
    x458 = x116*x429
    x459 = (2241/10240)*x121
    x460 = x217*x429
    x461 = (283/1536)*x460
    x462 = x429*x90
    x463 = (1/10)*x462
    x464 = x429*x56
    x465 = (1/8)*x464
    x466 = x197*x429
    x467 = (21/64)*x430
    x468 = (155/192)*x444
    x469 = x185*x429
    x470 = x239*x429
    x471 = x232*x429
    x472 = (295/768)*x456
    x473 = x117*x429
    x474 = x118*x429
    x475 = (1/4)*x464
    x476 = M2_0**11
    x477 = k**22
    x478 = M2_0*x420
    x479 = x10*x347
    x480 = x279*x31
    x481 = x234*x67
    x482 = x114*x181
    x483 = x115*x180
    x484 = x233*x68
    x485 = x278*x32
    x486 = x11*x346
    x487 = 1/(11*x0*x419 + x476 + x477 + 11*x478 + 55*x479 + 165*x480 + 330*x481 + 462*x482 + 462*x483 + 330*x484 + 165*x485 + 55*x486)
    x488 = x279*x487
    x489 = (1/80)*x82
    x490 = x103*x487
    x491 = (17/1920)*x490
    x492 = x487*x82
    x493 = (3/40)*x492
    x494 = (119/1920)*x490
    x495 = x234*x487
    x496 = (247/3840)*x132
    x497 = (59/1280)*x92
    x498 = x181*x487
    x499 = (119/3840)*x121
    x500 = (1367/7680)*x217
    x501 = x188*x487
    x502 = x145*x487
    x503 = (7/5120)*x502
    x504 = (357/5120)*x101
    x505 = x132*x487
    x506 = (247/768)*x505
    x507 = x487*x92
    x508 = (177/640)*x507
    x509 = x121*x487
    x510 = (119/960)*x509
    x511 = x217*x487
    x512 = (1367/1536)*x511
    x513 = x197*x487
    x514 = (7/1024)*x502
    x515 = x101*x487
    x516 = (1071/2560)*x515
    x517 = (3/16)*x492
    x518 = (119/640)*x490
    x519 = (119/384)*x490
    x520 = x285*x487
    x521 = x211*x487
    x522 = x232*x487
    x523 = (247/384)*x505
    x524 = (177/256)*x507
    x525 = (1367/768)*x511
    x526 = x185*x487
    x527 = x239*x487
    x528 = (7/512)*x502
    x529 = (1071/1024)*x515
    x530 = M2_0**12
    x531 = k**24
    x532 = M2_0*x477
    x533 = x10*x420
    x534 = x31*x347
    x535 = x279*x67
    x536 = x114*x234
    x537 = x180*x181
    x538 = x115*x233
    x539 = x278*x68
    x540 = x32*x346
    x541 = x11*x419
    x542 = 1/(12*x0*x476 + x530 + x531 + 12*x532 + 66*x533 + 220*x534 + 495*x535 + 792*x536 + 924*x537 + 792*x538 + 495*x539 + 220*x540 + 66*x541)
    x543 = (1/3840)*x542
    x544 = x543*x82
    x545 = x103*x543
    x546 = x202*x542
    x547 = x103*x542
    x548 = (1/480)*x547
    x549 = x279*x542
    x550 = (89/3840)*x132
    x551 = x347*x542
    x552 = (103/3840)*x92
    x553 = x234*x542
    x554 = (1/10)*x121
    x555 = (43/480)*x217
    x556 = x188*x542
    x557 = (73/1280)*x145
    x558 = (59/7680)*x101
    x559 = x132*x542
    x560 = (89/640)*x559
    x561 = x348*x542
    x562 = (721/3840)*x92
    x563 = x121*x542
    x564 = (1/2)*x563
    x565 = x217*x542
    x566 = (43/80)*x565
    x567 = x197*x542
    x568 = x145*x542
    x569 = (219/640)*x568
    x570 = (413/7680)*x101
    x571 = x349*x542
    x572 = (7/1280)*x82
    x573 = (7/960)*x547
    x574 = x350*x542
    x575 = (7/480)*x547
    x576 = x351*x542
    x577 = x352*x542
    x578 = (1617/5120)*x211
    x579 = (73/7680)*x232
    x580 = x211*x542
    x581 = (1617/1024)*x580
    x582 = x232*x542
    x583 = (73/1280)*x582
    x584 = (89/256)*x559
    x585 = (721/1280)*x92
    x586 = (43/32)*x565
    x587 = (721/768)*x92
    x588 = x239*x542
    x589 = x285*x542
    x590 = x353*x542
    x591 = x354*x542
    x592 = (219/256)*x568
    x593 = (413/2560)*x101
    x594 = (413/1536)*x101
    x595 = (1617/512)*x580
    x596 = (73/512)*x582
    x597 = M2_0**13
    x598 = k**26
    x599 = 1/(13*M2_0*x531 + 13*x0*x530 + 78*x10*x477 + 78*x11*x476 + 1287*x114*x279 + 1287*x115*x278 + 1716*x180*x234 + 1716*x181*x233 + 286*x31*x420 + 286*x32*x419 + 715*x346*x68 + 715*x347*x67 + x597 + x598)
    x600 = x347*x599
    x601 = (7/1920)*x132
    x602 = x420*x599
    x603 = (1/384)*x92
    x604 = (1/960)*x217
    x605 = x188*x599
    x606 = (3647/20480)*x605
    x607 = x197*x599
    x608 = (11/3072)*x145
    x609 = (43/15360)*x101
    x610 = x132*x599
    x611 = (49/1920)*x610
    x612 = x276*x599
    x613 = x217*x599
    x614 = (7/960)*x613
    x615 = (3647/4096)*x605
    x616 = x145*x599
    x617 = (77/3072)*x616
    x618 = x101*x599
    x619 = (43/1920)*x618
    x620 = x211*x599
    x621 = (1093/10240)*x620
    x622 = (223/10240)*x232
    x623 = (3279/5120)*x620
    x624 = x232*x599
    x625 = (1561/10240)*x624
    x626 = (49/640)*x610
    x627 = x599*x92
    x628 = (7/96)*x627
    x629 = (7/320)*x613
    x630 = (49/384)*x610
    x631 = (7/48)*x627
    x632 = (7/192)*x613
    x633 = x354*x599
    x634 = x428*x599
    x635 = (3647/2048)*x605
    x636 = (77/1024)*x616
    x637 = (301/3840)*x618
    x638 = (385/3072)*x616
    x639 = (301/1920)*x618
    x640 = (3279/2048)*x620
    x641 = (4683/10240)*x624
    x642 = (1561/2048)*x624
    x643 = M2_0**14
    x644 = k**28
    x645 = 1/(14*M2_0*x598 + 14*x0*x597 + 91*x10*x531 + 91*x11*x530 + 2002*x114*x347 + 2002*x115*x346 + 3003*x180*x279 + 3003*x181*x278 + 3432*x233*x234 + 364*x31*x477 + 364*x32*x476 + 1001*x419*x68 + 1001*x420*x67 + x643 + x644)
    x646 = x188*x645
    x647 = (2299/16384)*x646
    x648 = x347*x645
    x649 = (21469/245760)*x197
    x650 = (6897/8192)*x646
    x651 = x197*x645
    x652 = (150283/245760)*x651
    x653 = (139/15360)*x211
    x654 = x232*x645
    x655 = (103/15360)*x654
    x656 = x211*x645
    x657 = (973/15360)*x656
    x658 = (103/1920)*x654
    x659 = (34485/16384)*x646
    x660 = (150283/81920)*x651
    x661 = (150283/49152)*x651
    x662 = x354*x645
    x663 = (973/5120)*x656
    x664 = (721/3840)*x654
    x665 = (973/3072)*x656
    x666 = (721/1920)*x654
    x667 = M2_0**15
    x668 = k**30
    x669 = 1/(15*M2_0*x644 + 15*x0*x643 + 105*x10*x598 + 105*x11*x597 + 3003*x114*x420 + 3003*x115*x419 + 5005*x180*x347 + 5005*x181*x346 + 6435*x233*x279 + 6435*x234*x278 + 455*x31*x531 + 455*x32*x530 + 1365*x476*x68 + 1365*x477*x67 + x667 + x668)
    x670 = x188*x669
    x671 = (223/61440)*x670
    x672 = x197*x669
    x673 = (157/61440)*x672
    x674 = (1561/61440)*x670
    x675 = (157/7680)*x672
    x676 = (1561/20480)*x670
    x677 = (1099/15360)*x672
    x678 = (1561/12288)*x670
    x679 = (1099/7680)*x672
    x680 = M2_0**16
    x681 = k**32
    x682 = x197/(16*M2_0*x668 + 16*x0*x667 + 120*x10*x644 + 120*x11*x643 + 4368*x114*x477 + 4368*x115*x476 + 8008*x180*x420 + 8008*x181*x419 + 11440*x233*x347 + 11440*x234*x346 + 12870*x278*x279 + 560*x31*x598 + 560*x32*x597 + 1820*x530*x68 + 1820*x531*x67 + x680 + x681)
    x683 = (7/480)*x682
    x684 = (21/160)*x682
    x685 = (21/40)*x682
    x686 = (49/40)*x682
    x687 = (147/80)*x682
    x688 = x197/(M2_0**17 + 17*M2_0*x681 + k**34 + 17*x0*x680 + 136*x10*x668 + 136*x11*x667 + 6188*x114*x531 + 6188*x115*x530 + 12376*x180*x477 + 12376*x181*x476 + 19448*x233*x420 + 19448*x234*x419 + 24310*x278*x347 + 24310*x279*x346 + 680*x31*x644 + 680*x32*x643 + 2380*x597*x68 + 2380*x598*x67)
    x689 = (1/960)*x688
    x690 = (1/96)*x688
    x691 = (3/64)*x688
    x692 = (1/8)*x688
    x693 = (7/32)*x688
    return M2_0*T_1*x122*x37*x88 + (25/3072)*M2_0*x135 + (25/2048)*M2_0*x144 + (19/2048)*M2_0*x146 + (11/4096)*M2_0*x160 + (255/32768)*M2_0*x198 - 31/2048*M2_0*x212 + (43/8192)*M2_0*x248 + (1/64)*M2_0*x35*x41 + (3/256)*M2_0*x47 - 1/512*M2_0*x63 - M2_0*x83 + (119/2048)*M2_2*x105 + (111/2560)*M2_2*x11*x157 + (13/960)*M2_2*x158*x69 + M2_2*x179*x187*x23 - M2_2*x2*x7 + M2_2*x200*x70 - M2_2*x3*x37*x73*x9 - 1/256*M2_2*x45 + M2_3*T_1*x190*x4*x9 - T_0*x19*x20 - 1/4*x0*x122*x89 - 73/3072*x0*x135 - 71/768*x0*x140*x78 - 121/6144*x0*x144 + (61/6144)*x0*x146 + (409/12288)*x0*x160 + x0*x18*x29 + (415/3072)*x0*x189 - 1243/98304*x0*x198 - 77/6144*x0*x212 - 163/8192*x0*x248 - x0*x36*x5 + (5/768)*x0*x47 + (1/256)*x0*x50 - 9/512*x0*x63 - x0*x83 + x10*x134*x216 - 5/3072*x10*x165 + (7/2048)*x10*x169 + (1/1024)*x10*x187*x217 + x10*x220*x90 - 47/8192*x10*x262 + (347/1920)*x100*x17 - 73/1920*x100*x22 - 161/3840*x100*x60 + (19/480)*x100*x75 + (163/15360)*x101*x193 + (201/5120)*x101*x205 + (11/384)*x101*x222 - 41/7680*x101*x242 + (41/15360)*x101*x356 + (3541/15360)*x101*x404 - 3/256*x101*x73 - 19/1536*x101*x97 - x102*x103 + x102*x46 + (1/256)*x103*x108 + x103*x199*x68 + (1/384)*x103*x226 - x103*x292 + (3/512)*x103*x95 + (1/512)*x104*x190 + (1/128)*x104*x226 - 1/20*x104*x332 - 17/384*x105*x12 - x105*x64 - 7/480*x106*x96 + x107*x138 - x107*x97 - 7/256*x108*x46 - 1/24*x108*x78 - 91/960*x109*x33 - 1/24*x109*x34 + (49/1920)*x11*x141 + (3/320)*x11*x164 + (983/15360)*x11*x165 - 1367/30720*x11*x167 + (117/10240)*x11*x169 + (155/2048)*x11*x260 + (2783/30720)*x11*x262 + (2389/30720)*x11*x272 - 4469/81920*x11*x326 + (23/480)*x11*x45 + (19/192)*x110*x17 - 1/32*x110*x22 - 11/384*x110*x60 + (7/192)*x110*x75 + (1891/15360)*x111*x193 - 1597/15360*x111*x242 - 2557/7680*x111*x251 - 5437/15360*x111*x265 - 1/8*x111*x274 - 1399/1280*x111*x398 + (157/768)*x111*x97 - 3/320*x112*x33 + (1/192)*x112*x34 + (103/3840)*x113*x138 - 169/4096*x113*x140 + (25/768)*x113*x154 - 21/2560*x113*x193 + (953/6144)*x113*x196*x355 - 1/640*x113*x251 + (1/192)*x113*x97 - 13/1680*x115*x172 + x115*x201 + (333/32768)*x115*x501 - 11/240*x116*x126 - 143/3360*x116*x172 + x116*x204 + x116*x213 - x116*x337 - 557/3840*x116*x411 - 3229/3072*x116*x417 + (21743/61440)*x116*x466 - 1091/98304*x116*x501 - 47/480*x117*x126 - 313/3360*x117*x172 + x117*x204 + x117*x213 - x117*x337 - 523/2560*x117*x411 - 811/512*x117*x417 + (29887/40960)*x117*x466 - 3089/32768*x117*x501 - 49/480*x118*x126 - 1/10*x118*x172 + x118*x201 - 163/1280*x118*x411 - 3259/3072*x118*x417 + (147/640)*x118*x418 + (12677/20480)*x118*x466 - 3755/32768*x118*x501 + (301/3072)*x119*x211*x429 - 91/3072*x119*x411 - 1637/6144*x119*x417 - 1/192*x119*x418 + (9235/49152)*x119*x466 - 511/12288*x119*x501 + (7/384)*x12*x141 + (1/192)*x12*x164 + (71/768)*x12*x165 + (93/1024)*x12*x167 + (25/3072)*x12*x169 + (49/384)*x12*x260 + (2771/24576)*x12*x262 + (65/1024)*x12*x272 - 713/12288*x12*x326 + (7/96)*x12*x45 + (1/512)*x120*x121 - 179/4096*x122*x128 - x122*x177*x60 - x123*x125 + x123*x131 - x123*x5 - 1/480*x124*x136 - x125*x166 - x126*x127 - x126*x129 - x126*x170 - x127*x172 + x129*x192 + x129*x22*x247 - x129*x322 + x13*x23*x24 + x130*x147 + x131*x166 + (49/7680)*x132*x209 + (7/768)*x132*x218 - 283/7680*x132*x257 - 443/7680*x132*x261 - 1/48*x132*x273 - 241/3840*x132*x311 - 219/1280*x132*x320 - 197/1280*x132*x344 - 35/768*x132*x345 + (7/3840)*x132*x432 - x133*x30 + x133*x38 - 69/1280*x134*x209 - 5/96*x134*x218 - 25/512*x134*x257 - 57/512*x134*x261 - x134*x275 + (107/1280)*x134*x320 - 1/48*x136*x29 + x137*x32 + x137*x33 + (23/960)*x138*x142 - x138*x151 - 229/3840*x138*x77 - 1/480*x138*x98 - x139*x143 - 183/1280*x139*x17 - 293/7680*x139*x26 - 7/320*x139*x60 - 17/384*x139*x75 + x14*x16 + (1/384)*x14*x22 - 7/768*x14*x26 + x14*x27*x6 + x14*x28*x7 + x142*x206 + (1/48)*x142*x222 + x143*x267*x9 - 53/7680*x145*x209 - 5/1536*x145*x218 - 43/5120*x145*x257 - 123/5120*x145*x261 - 313/7680*x145*x320 - 197/5120*x145*x376 + x147*x175 + x148*x32 + x148*x33 + x149*x150 - 13/5760*x149*x173 - x15*x36*x37 + x150*x158*x70 - x151*x154 + x151*x222 - x152*x153 + (13/960)*x152*x28 + x154*x156 - x154*x16*x8 - 1/32*x154*x77 + (7/384)*x154*x79 - 383/1280*x155*x17 - 11/320*x155*x22 - 127/1920*x155*x60 - 37/384*x155*x75 + x156*x193 + (23/256)*x157*x66 + (9/1280)*x159*x33 - 5/768*x159*x34 - x161*x177 - 211/2560*x161*x33 - 5/256*x162*x33 - 1/96*x162*x34 - x163*x164 + x163*x260 + (1/960)*x166*x175 - x166*x5 - x167*x168 + x168*x187*x232 - 5/32*x17*x176 + x17*x18 - 1/48*x17*x93 - x170*x172 - x174*x179 - 1/960*x174*x68 - 1/320*x174*x70 - 5/192*x176*x22 - 5/96*x176*x75 - 1/480*x178*x69 - 1/384*x178*x70 + x181*x447 + (31/6144)*x181*x450 - 3041/122880*x181*x556 - 41/1280*x182*x316 - x182*x380 - x182*x397 - x182*x410 + (1039/15360)*x182*x412 + x182*x443 + (5/768)*x182*x450 - 7/640*x182*x452 + x182*x455 + x182*x461 + x182*x463 + (109/320)*x182*x471 - x182*x510 + (26791/122880)*x182*x513 - 19/1280*x182*x521 - 1783/15360*x182*x556 - 17/240*x183*x316 - 21/1280*x183*x379 - 1013/2560*x183*x396 - 193/640*x183*x409 + (1873/7680)*x183*x412 + (173/320)*x183*x442 - 25/1024*x183*x450 - 119/1920*x183*x452 + (331/1280)*x183*x454 + (283/1024)*x183*x460 + (3/20)*x183*x462 + (12371/30720)*x183*x471 - 119/640*x183*x509 + (31841/81920)*x183*x513 + (411/5120)*x183*x521 - 4441/20480*x183*x556 - 149/1920*x184*x316 - x184*x380 - x184*x397 - x184*x410 + (2707/7680)*x184*x412 + x184*x443 - 95/1536*x184*x450 - 49/480*x184*x452 + x184*x455 + x184*x461 + x184*x463 + (1907/15360)*x184*x471 - x184*x510 + (41941/122880)*x184*x513 + (487/2560)*x184*x521 - 6191/30720*x184*x556 - 27/640*x185*x316 + x185*x447 - 305/6144*x185*x450 - 91/1280*x185*x452 - 1581/20480*x185*x471 + (72241/491520)*x185*x513 - 11441/122880*x185*x556 - 7/768*x186*x316 + (7/192)*x186*x355*x92 + (175/3072)*x186*x412 - 7/512*x186*x450 - 7/384*x186*x452 - 133/3072*x186*x471 + (101/4096)*x186*x513 + (21/512)*x186*x521 - 35/2048*x186*x556 + (983/32768)*x187*x188 + (11/640)*x187*x196*x43 + x189*x38 - x19*x3*x36 + x190*x191 + (1/64)*x190*x40 + (83/1920)*x190*x41 + (27/1280)*x190*x46 + x191*x221 + x192*x225 + x193*x194 + (23/960)*x193*x195 + x193*x202 + (493/7680)*x193*x54 + (201/2048)*x193*x56 + (83/1920)*x193*x77 + (27/1280)*x193*x79 + (11/480)*x193*x92 - x194*x242 - x194*x265 + (53/960)*x195*x205 + x195*x224 - 11/128*x195*x398 + x196*x200 + (4533/81920)*x197*x439 + (23761/491520)*x197*x498 - 11091/40960*x197*x553 - x2*x5 - x20*x30*x6 + x203*x207 + (21/320)*x203*x61 + (1/24)*x203*x62 + (1/60)*x203*x78 + (351/2560)*x205*x54 + (1063/6144)*x205*x56 + (203/1920)*x205*x77 + (67/1280)*x205*x79 + (17/3840)*x205*x82 + (223/3840)*x205*x92 + x206*x98 - x207*x399 - x207*x403 + x208*x228 + (143/960)*x208*x69 + (323/1920)*x208*x70 - 1387/15360*x209*x210 + (3131/61440)*x209*x217 - 4391/30720*x209*x232 + (153/10240)*x209*x90 - 1/12*x210*x218 - 483/5120*x210*x257 - 1123/5120*x210*x261 - 1/8*x210*x273 + (107/640)*x210*x320 + (1489/20480)*x211*x439 + (19421/61440)*x211*x458 + (10487/20480)*x211*x473 + (7509/20480)*x211*x474 - 143/10240*x211*x498 + (1537/10240)*x211*x526 + (47/640)*x214*x69 + (107/1280)*x214*x70 + (13/1920)*x215*x69 + (7/960)*x215*x70 + x216*x31*x54 - 95/1024*x217*x218 - 12277/61440*x217*x257 - 7919/20480*x217*x261 - 287/1536*x217*x273 + (2357/30720)*x217*x311 + (3797/15360)*x217*x320 + (8117/30720)*x217*x344 + (3/32)*x217*x345 + (157/640)*x217*x376 + (1403/1920)*x217*x388 + (461/640)*x217*x414 - 469/3072*x218*x232 - 77/1536*x218*x90 + (4451/15360)*x219*x33 + (1/6)*x219*x34 + x220*x31*x56 + (11/192)*x221*x40 + (51/640)*x221*x61 + (5/96)*x221*x62 + (115/1536)*x222*x54 + (239/3072)*x222*x56 + (1/16)*x222*x77 + (9/256)*x222*x92 + x222*x99 + x224*x79 - x225*x322 + x227*x40 + x227*x62 - x228*x241*x54 - x228*x270 + x229*x46 + x229*x61 - 1/2560*x230*x33 + (1/128)*x230*x34 + (3/256)*x231*x69 + (9/512)*x231*x70 - 3059/30720*x232*x257 - 2523/10240*x232*x261 - 451/3072*x232*x273 - 607/30720*x232*x311 + (743/15360)*x232*x320 + (4793/30720)*x232*x344 + (45/512)*x232*x345 - 1607/6144*x232*x376 + (5897/61440)*x232*x432 + (12913/30720)*x232*x495 + x234*x448 + (13/30720)*x234*x451 + x234*x503 + x234*x606 - x235*x373 - x235*x382 + x235*x445 - 71/6144*x235*x451 - 143/256*x235*x453 + x235*x457 + x235*x465 + x235*x506 + x235*x512 + x235*x514 + (13165/6144)*x235*x522 - x235*x564 - 11151/8192*x235*x567 - x235*x581 + x235*x615 - x236*x400 - x236*x413 - 197/3072*x236*x451 - 75/64*x236*x453 + x236*x468 + x236*x472 + x236*x475 + (13543/3072)*x236*x522 + x236*x523 + x236*x525 + x236*x528 - x236*x563 - 11241/4096*x236*x567 - x236*x595 + x236*x635 - x237*x400 - x237*x413 - 407/3072*x237*x451 - 485/384*x237*x453 + x237*x468 + x237*x472 + x237*x475 + (14173/3072)*x237*x522 + x237*x523 + x237*x525 + x237*x528 - x237*x563 - 11391/4096*x237*x567 - x237*x595 + x237*x635 - x238*x373 - x238*x382 + x238*x445 - 827/6144*x238*x451 - 185/256*x238*x453 + x238*x457 + x238*x465 + x238*x506 + x238*x512 + x238*x514 + (15433/6144)*x238*x522 - x238*x564 - 11691/8192*x238*x567 - x238*x581 + x238*x615 + x239*x448 - 2087/30720*x239*x451 - 51/256*x239*x453 + x239*x503 + (19213/30720)*x239*x522 - 12591/40960*x239*x567 + x239*x606 - x24*x72*x78 - 7/512*x240*x451 - 7/384*x240*x453 + (21/512)*x240*x522 - 15/2048*x240*x567 + x242*x249 - x242*x263 - 463/7680*x242*x54 - 31/512*x242*x56 - 1/2560*x242*x92 + x243*x244 + x243*x245 - x243*x250 + x243*x264 + x244*x267 + x245*x267 + x246*x46 + x246*x61 - x247*x277*x71 + x249*x265 - x250*x267 - 703/3840*x251*x54 + (31/1920)*x251*x82 - 83/3840*x251*x92 - 1/32*x251*x98 + x252*x253 + x252*x254 + x252*x255 + x252*x256 + x252*x258 - x252*x259 + x252*x269 + x253*x266 + x254*x266 - x254*x332 + x255*x266 + x256*x266 - 11/240*x257*x90 + x258*x266 - x258*x332 - x259*x266 - 13/120*x261*x90 - x263*x265 + x264*x267 - 1423/7680*x265*x54 - 323/7680*x265*x92 + x266*x269 + x268*x46 + x268*x61 + x27*x35*x43 - 47/256*x270*x69 - 95/512*x270*x70 - 101/3840*x271*x69 - 281/7680*x271*x70 - x273*x277*x8 - x274*x276 - x275*x90 + x279*x431 - 1491/8192*x279*x607 - x279*x621 + x279*x647 + x280*x436 - x280*x493 + x280*x508 - x280*x516 + x280*x560 - x280*x566 + x280*x569 - x280*x583 - 4469/4096*x280*x607 - x280*x623 + x280*x650 + x281*x467 - x281*x517 + x281*x524 - x281*x529 + x281*x584 - x281*x586 + x281*x592 - x281*x596 - 22317/8192*x281*x607 - x281*x640 + x281*x659 + (7/16)*x282*x430 - 1/4*x282*x492 + (59/64)*x282*x507 - 357/256*x282*x515 + (89/192)*x282*x559 - 43/24*x282*x565 + (73/64)*x282*x568 - 73/384*x282*x582 - 7425/2048*x282*x607 - 1093/512*x282*x620 + (11495/4096)*x282*x646 + x283*x467 - x283*x517 + x283*x524 - x283*x529 + x283*x584 - x283*x586 + x283*x592 - x283*x596 - 22205/8192*x283*x607 - x283*x640 + x283*x659 + x284*x436 - x284*x493 + x284*x508 - x284*x516 + x284*x560 - x284*x566 + x284*x569 - x284*x583 - 4413/4096*x284*x607 - x284*x623 + x284*x650 + x285*x431 - 1443/8192*x285*x607 - x285*x621 + x285*x647 + (1/1024)*x286*x607 - x288*x293 - x288*x297 - x288*x299 - x288*x313 - x288*x318 + x288*x325 + x288*x329 + (19/1920)*x288*x82 + (323/3840)*x288*x92 - x289*x290 - x289*x295 - x290*x334 - x291*x294 + x291*x296 + x291*x298 - x291*x300 - x291*x323 + x292*x40 - x293*x333 - x294*x336 - x295*x334 + x296*x336 - x297*x333 + x298*x336 - x299*x333 + x30*x35*x61 - x300*x336 - x301*x302 - x301*x305 - x301*x307 - x301*x309 - x301*x314 - x301*x321 - x301*x330 + x301*x339 + x301*x340 + (149/3840)*x301*x82 + (143/480)*x301*x92 - x302*x331 + x303*x304 - x303*x306 + x303*x308 + x303*x310 - x303*x315 - x303*x338 + x304*x335 - x305*x331 - x306*x335 - x307*x331 + x308*x335 - x309*x331 - x309*x398 + x310*x335 + x311*x312 + x311*x317 + x311*x324 - x311*x328 + x312*x344 - x313*x333 - x314*x331 - x315*x335 + x317*x344 - x318*x333 + (2857/30720)*x319*x32 + (6697/30720)*x319*x33 + (1/8)*x319*x34 + (11383/122880)*x32*x327 + (3541/30720)*x32*x343 - 1329/16384*x32*x392 + (1/160)*x320*x90 - x321*x331 - x323*x336 + x324*x344 + x325*x333 + (8021/40960)*x327*x33 + (317/3072)*x327*x34 - x328*x344 + x329*x333 + (6481/30720)*x33*x343 - 2393/16384*x33*x392 - x330*x331 + x331*x339 + x331*x340 + (73/1280)*x331*x82 + (249/640)*x331*x92 + (11/320)*x332*x40 + (19/80)*x332*x61 + (7/40)*x332*x62 + (143/3840)*x333*x82 + (53/240)*x333*x92 - x335*x338 + (11/1920)*x336*x40 + (49/512)*x34*x343 - 133/2048*x34*x392 + x341*x342 - x341*x574 - x341*x576 + (35/768)*x342*x92 - x347*x491 - x347*x544 + x347*x671 - x348*x494 - x348*x546 + x348*x611 + x348*x614 + x348*x617 + x348*x625 + x348*x652 - x348*x657 + x348*x674 - x349*x518 + x349*x626 + x349*x629 + x349*x636 + x349*x641 + x349*x660 - x349*x663 + x349*x676 + x35*x38*x40 - x350*x519 + x350*x630 + x350*x632 + x350*x638 + x350*x642 + x350*x661 - x350*x665 + x350*x678 - x351*x519 + x351*x630 + x351*x632 + x351*x638 + x351*x642 + x351*x661 - x351*x665 + x351*x678 - x352*x518 + x352*x626 + x352*x629 + x352*x636 + x352*x641 + x352*x660 - x352*x663 + x352*x676 - x353*x494 - x353*x546 + x353*x611 + x353*x614 + x353*x617 + x353*x625 + x353*x652 - x353*x657 + x353*x674 - x354*x491 - x354*x544 + x354*x671 - x356*x363 - x356*x365 - x356*x378 - x356*x385 - x356*x391 - 37/1536*x356*x92 - x357*x358 - x357*x361 - x358*x405 - x359*x366 - x360*x362 - x360*x364 - x360*x76 - x361*x405 - x362*x408 - x363*x404 - x364*x408 - x365*x404 - x366*x407 - x367*x368 - x367*x371 - x367*x374 - x367*x390 - 23/384*x367*x92 - x368*x402 - x369*x370 - x369*x375 - x370*x406 - x371*x402 - x374*x402 - x375*x406 + x376*x377 - x376*x383 + x376*x384 + x377*x415 - x378*x404 + x38*x50 - x383*x415 + x384*x415 - x385*x404 - 145/512*x386*x69 - x387*x68 - x387*x70 + x388*x389 - x388*x394 + x388*x395 + x389*x414 - x390*x402 - x391*x404 + (283/16384)*x393*x68 - 643/24576*x393*x69 - 5119/49152*x393*x70 - 373/6144*x393*x71 - x394*x414 + x395*x414 + (1/768)*x398*x92 - 7/160*x398*x98 - x399*x401 - 1/192*x40*x407 - 41/768*x40*x59 - x401*x403 + (47/384)*x402*x92 + (187/1536)*x404*x92 - 5/32*x41*x59 + (2599/6144)*x416*x69 + (2339/6144)*x416*x70 + (231/2048)*x416*x71 - x420*x545 - x420*x655 + x420*x673 - x421*x548 + x421*x612 + x421*x619 - x421*x658 + x421*x675 - x422*x573 + x422*x628 + x422*x637 - x422*x664 + x422*x677 - x423*x575 + x423*x631 + x423*x639 - x423*x666 + x423*x679 - 7/384*x424*x547 + (301/1536)*x424*x618 + (35/192)*x424*x627 - 721/1536*x424*x654 + (1099/6144)*x424*x672 - x425*x575 + x425*x631 + x425*x639 - x425*x666 + x425*x679 - x426*x573 + x426*x628 + x426*x637 - x426*x664 + x426*x677 - x427*x548 + x427*x612 + x427*x619 - x427*x658 + x427*x675 - x428*x545 - x428*x655 + x428*x673 + x432*x433 + x432*x437 + x432*x441 + x433*x469 + x434*x435 + x434*x438 - 83/768*x434*x92 + x435*x470 + x437*x469 + x438*x470 - x439*x440 - 17/320*x44*x96 - x440*x474 + x441*x469 - 1175/16384*x449*x68 - 3967/16384*x449*x69 - 4409/16384*x449*x70 - 1617/16384*x449*x71 - x458*x459 - x459*x473 + (17/7680)*x46*x95 - x477*x683 - x478*x684 - x479*x685 - 5/512*x48*x55 - x480*x686 - x481*x687 - x482*x687 - x483*x686 - x484*x685 - x485*x684 - x486*x683 - x488*x489 + x488*x497 - x488*x504 - x489*x520 - 11/512*x49*x51 - x49*x58 + x49*x65 + (1/32)*x49*x66 + x495*x496 + x495*x500 + x496*x527 + x497*x520 - x498*x499 - x499*x526 + x500*x527 - x504*x520 - x52*x54 + x52*x56 - x531*x689 - x532*x690 - x533*x691 - x534*x692 - x535*x693 - 21/80*x536*x688 - x537*x693 - x538*x692 - x539*x691 - 17/1536*x54*x73 + (85/1536)*x54*x97 - x540*x690 - x541*x689 + x549*x550 - x549*x555 + x549*x557 - x549*x579 + x55*x72*x86 + x550*x589 - x551*x552 + x551*x558 - x552*x591 - x553*x554 - x553*x578 - x554*x588 - x555*x589 + x557*x589 + x558*x591 - 17/1024*x56*x73 + (51/1024)*x56*x97 - x561*x562 + x561*x570 - x562*x590 + x570*x590 - x571*x572 - x571*x585 + x571*x593 - x572*x577 - x574*x587 + x574*x594 - x576*x587 + x576*x594 - x577*x585 + x577*x593 - x578*x588 - x579*x589 + x58*x84 - 1/192*x59*x62 - 5/768*x60*x74 + x600*x601 + x600*x604 + x600*x608 + x600*x622 + x601*x633 + x602*x603 + x602*x609 + x603*x634 + x604*x633 + x608*x633 + x609*x634 + x622*x633 + x648*x649 - x648*x653 + x649*x662 - x65*x84 - x653*x662 - 5/192*x66*x84 - 1/48*x73*x77 - x74*x76 - x76*x93 + (11/192)*x77*x97 - 9/320*x78*x95 - x79*x80 - x8*x94*x97 + x80*x92 + x81*x90 + x84*x85 + x85*x91*x97 - 1/1536*x87*x89 + x93*x94 + x97*x99


def _loc_scalar_1(k, q, Md, Td):
    """scalar O(psi), coefficient of c e^{iqx}: static + mixed + homogeneous time-derivative local terms"""
    return _loc_scalar_1_flat(k, q, *Md[:4], *Td[:4]) + _temporal_scalar_1(k, Md, Td)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].lstrip().startswith("{"):
    P = json.loads(sys.argv[1])
    o = run(P)
    print(f"{P['kind']} q={P['q']:g} m={P['m']:g}: done in {o['seconds']:.0f} s ({o['steps']} steps)")
