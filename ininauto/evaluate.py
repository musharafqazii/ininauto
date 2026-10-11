"""End-to-end solver: covariant action -> renormalised <rho> through O(lambda h^2), as numbers.

    import sympy as sp
    from ininauto import KIN, PHI2, RPHI2, PHI4, evaluate
    m, lam = sp.symbols('m lam', real=True)
    res = evaluate(-KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24, values={'m': 1.0})
    print(res.summary())

Pipeline
  1. parse the action (xi, m^2, lambda);
  2. derive geometry, reduced Lagrangian, Hamiltonian, normal-observer energy density (symbolic);
  3. re-derive every kernel the numerical back end uses -- mode operators D_0, P_1, P_2 by Euler-Lagrange,
     the velocity energy density, the quartic vertices and determinant factors -- and stop if any differs
     (ininauto.bridge);
  4. integrate the retarded mode hierarchy in the in-vacuum for each Fourier mode q of the metric perturbation,
     evaluate the in-in energy density, renormalise (4th-order adiabatic + full local subtraction = covariant
     result at mu_R = m, curvature-squared term for other mu_R; finite Hartree tadpoles and contacts);
  5. assemble rho_00, rho_01, rho_02, rho_10, rho_11, rho_12 (time series and late-time values) and write them out.
"""
from __future__ import annotations

from pathlib import Path
import json
import sympy as sp

from .action import ScalarAction
from .bridge import verify_all
from .numerics import runner


def _number(expr, values, what):
    v = sp.sympify(expr).subs({sp.Symbol(k, real=True): val for k, val in values.items()})
    v = v.subs({sp.Symbol(k): val for k, val in values.items()})
    v = sp.sympify(v)
    if v.free_symbols:
        raise ValueError(f"{what} = {expr} needs numerical values for {sorted(map(str, v.free_symbols))} "
                         f"(pass values={{...}}, in units of H_e)")
    return float(v)


def evaluate(action, values=None, *, families=("scalar", "tensor"), q_grid=(0.1, 0.1316, 0.1732, 0.228, 0.3),
             output_dir="inin_results", workers=None, plots=True, log=print, hom_settings=None, **settings):
    """Run the whole calculation.  ``values`` gives numbers for the symbols of the action (m in units of H_e;
    the quartic coupling may stay symbolic -- results are returned per unit lambda and, if a value is given, summed).
    ``settings`` override the numerical defaults (runner.DEFAULTS), e.g. pmax, np, nmu, nend, dtheta, mu_R."""
    values = dict(values or {})
    act = action if isinstance(action, ScalarAction) else ScalarAction.parse(action)
    m2 = _number(act.mass_squared, values, "m^2")
    if m2 <= 0:
        raise ValueError("the numerical back end needs m^2 > 0")
    lam_sym = act.coupling
    lam_val = None
    if lam_sym != 0:
        try:
            lam_val = _number(lam_sym, values, "lambda")
        except ValueError:
            lam_val = None                                         # stays symbolic: per-unit-lambda coefficients
    log("step 1-3: symbolic derivation and action -> engine kernel verification")
    reports = verify_all(act, families)
    for r in reports.values():
        log(str(r))
    if not all(r.ok for r in reports.values()):
        raise RuntimeError("The numerical kernels do not follow from this action; refusing to run.")
    log(f"step 4: numerical evaluation (m = {m2**0.5:g} H_e, families {list(families)}, q grid {list(q_grid)})")
    res = runner.evaluate_numeric(m2**0.5, families=families, q_grid=q_grid, lam=(lam_sym != 0), output_dir=Path(output_dir) / "runs",
                                  workers=workers, log=log, hom_settings=hom_settings, **settings)
    res.bridge = {f: {k: str(v) for k, v in r.checks.items()} for f, r in reports.items()}
    res.settings.update(action=str(act.expression), values={k: float(v) for k, v in values.items()},
                        lambda_value=lam_val)
    if lam_val is not None:
        h = res.homogeneous
        if "rho_10_per_lambda" in h:
            h["rho_total_homogeneous"] = h["rho_00"] + lam_val*h["rho_10_per_lambda"]
        for fam, integ in res.integrated.items():
            if "rho_12_per_lambda_int" in integ:
                integ["rho_2_total_int"] = integ["rho_02_int"] + lam_val*integ["rho_12_per_lambda_int"]
    log("step 5: writing results")
    path = runner.save(res, output_dir)
    log(f"   {path}")
    if plots:
        try:
            _plot(res, Path(output_dir))
        except Exception as exc:                                  # plotting is optional
            log(f"   (plots skipped: {exc})")
    return res


def _plot(res, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(res.families)
    fig, ax = plt.subplots(1, 1 + len(fams), figsize=(5.5*(1 + len(fams)), 4.2))
    ax = np.atleast_1d(ax)
    h = res.homogeneous
    ax[0].plot(h["N"], np.abs(h["rho_00"]), label=r"$|\rho_{00}|$")
    if "rho_10_per_lambda" in h:
        ax[0].plot(h["N"], np.abs(h["rho_10_per_lambda"]), label=r"$|\rho_{10}|/\lambda$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("N = ln a"); ax[0].set_title(r"homogeneous, $a^4\langle\rho\rangle$"); ax[0].legend()
    for a_, fam in zip(ax[1:], fams):
        for key, val in res.integrated[fam].items():
            if key == "N":
                continue
            a_.plot(res.integrated[fam]["N"], np.abs(val), label=key)
        a_.set_yscale("log"); a_.set_xlabel("N = ln a"); a_.set_title(f"{fam}: second order, per unit power, $\\int d\\ln q$")
        a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rho_overview.png", dpi=130); plt.close(fig)
