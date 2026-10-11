"""End-to-end numerical evaluation: action -> renormalised <rho> coefficients rho_00 ... rho_12.

The action fixes the field content, xi = 1/6, m^2 and lambda.  What the action cannot fix is supplied here
explicitly (and recorded in every output): the background, the perturbation profiles, the state and the
renormalisation scale.

* background: smooth inflation -> radiation, H(N) = H_e 2^{1/4} (1 + e^{8N})^{-1/4} (units H_e = a_e = 1);
* perturbation profiles: T(N) of a single Fourier mode q -- scalar: Newtonian potential (constant on super-horizon
  scales during inflation, radiation-era oscillation inside); tensor: free tensor mode with a C^2 switch-on at horizon entry;
* state: adiabatic (4th order) in-vacuum of every mode, quartic coupling switched on smoothly (f_sw) before N = -3;
* renormalisation: 4th-order adiabatic subtraction of the FRW vacuum and full 4th-order local subtraction of every
  metric kernel = the covariant (proper-time) result at mu_R = m; other mu_R through the curvature-squared term.

Normalisation of the outputs (all comoving, a^4 <rho>, units H_e^4):
  rho_00(N), rho_10(N)/lambda            homogeneous, renormalised;
  rho_01(N; q), rho_11(N; q)/lambda      coefficient of c e^{iqx} for psi = T(eta) (c e^{iqx} + c.c.)  [scalar];
                                         tensor: quadrupole moment (the density itself averages to zero);
  rho_02(N; q), rho_12(N; q)/lambda      ensemble mean per unit power <psi^2> = T^2 (scalar) / <h_ij h_ij> = T^2;
  *_int                                   the q-grid integral over ln q (flat spectrum per ln q).
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import engine as E

DEFAULTS = dict(nend=5.5, dtheta=0.05, pmin=0.03, pmax=5.0, np=40, nmu=8, init_order=4, integrator="rk6",
                avg_window=0.05, avg_every=4, avg_center=True, local_order=4, mu_R=None, fsw=[-6.0, -4.5])
HOM_DEFAULTS = dict(DEFAULTS, pmin=0.05, pmax=12.0, np=80, dtheta=0.025)
SERIES_STEP = 0.25


def half_cores():
    return max(1, (os.cpu_count() or 2) // 2)


def _series(nend, start):
    return [round(float(v), 6) for v in np.arange(start, nend - 1e-9, SERIES_STEP)]


def _finish(P, outdir, tag):
    P["nrec"] = [P["nend"] - 1.0, P["nend"] - 0.5, P["nend"]]
    key = hashlib.sha1(json.dumps({k: v for k, v in P.items() if k != "out"}, sort_keys=True).encode()).hexdigest()[:12]
    P["out"] = str(Path(outdir) / f"{tag}_{key}.npz")
    return P


def perturbation_job(kind, q, m, outdir, lam=True, **over):
    P = dict(DEFAULTS, **over); P.update(kind=kind, q=float(q), m=float(m), lin=True, quad=True, lam=bool(lam))
    P["nser"] = _series(P["nend"], -2.0)
    return _finish(P, outdir, f"{kind}_q{q:.4g}")


def homogeneous_job(m, outdir, lam=True, **over):
    P = dict(HOM_DEFAULTS, **over); P.update(kind="scalar", q=0.1, m=float(m), lin=False, quad=False, lam=bool(lam))
    P["nser"] = _series(P["nend"], -3.0)
    return _finish(P, outdir, "homogeneous")


def _cached(P):
    f = Path(P["out"])
    if not f.exists():
        return False
    try:
        return json.loads(str(np.load(f)["params"])) == P
    except Exception:
        return False


def _run_one(P):
    E.run(P)
    return P["out"]


def run_jobs(jobs, workers=None, log=print):
    todo = [P for P in jobs if not _cached(P)]
    log(f"{len(jobs) - len(todo)} cached, {len(todo)} to run on {workers or half_cores()} worker(s)")
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers or half_cores()) as ex:
        futs = {ex.submit(_run_one, P): P for P in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            P = futs[fu]
            fu.result()                                      # re-raise engine errors
            log(f"  [{i}/{len(todo)}] done: {Path(P['out']).name}")


def load(P):
    d = np.load(P["out"])
    Nrec = [float(v) for v in d["Nrec"]]
    rec = [{k.split("__")[0]: d[k] for k in d.files if k.endswith(f"__{i}")} for i in range(len(Nrec))]
    return dict(p=d["p"], wmu=d["wmu"], Wp=float(d["Wp"]), Nrec=Nrec, rec=rec, P=P)


def trapP(R, f, pcut=None):
    """int d^3p/(2pi)^3 f on the run's logarithmic grid."""
    p = R["p"]; lnp = np.log(p); f = p**3 / (2 * np.pi**2) * np.asarray(f)
    if pcut is not None:
        keep = p <= pcut * (1 + 1e-9); f, lnp = f[..., keep], lnp[keep]
    return np.sum((f[..., 1:] + f[..., :-1]) / 2 * np.diff(lnp), axis=-1)


def int_lnq(qs, y):
    qs, y = np.asarray(qs, float), np.asarray(y)
    if len(qs) == 1:
        return y[0]
    dl = np.diff(np.log(qs)).reshape((-1,) + (1,) * (y.ndim - 1))
    return np.sum((y[1:] + y[:-1]) / 2 * dl, axis=0)


def _g(r, key, default=0.0):
    return r[key] if key in r else default


def homogeneous_series(R):
    out = dict(N=np.array(R["Nrec"]))
    out["rho_00"] = np.array([trapP(R, r["rho0_k"]) for r in R["rec"]])
    if "ex0_k" in R["rec"][0]:
        out["rho_10_per_lambda"] = np.array([trapP(R, r["ex0_k"]) + float(r["contact0"]) for r in R["rec"]])
    return out


def perturbation_series(R):
    """All metric orders of one run (one q), as time series."""
    out = dict(N=np.array(R["Nrec"]), T=np.array([r["T"][0] for r in R["rec"]]))
    out["rho_01"] = np.array([complex(trapP(R, r["dE1_p"]) + complex(_g(r, "ct1", 0))) for r in R["rec"]])
    out["rho_02"] = np.array([float(trapP(R, r["dE2_k"]) + float(_g(r, "ct2", 0))) for r in R["rec"]])
    out["rho_02_hybrid"] = np.array([float(trapP(R, r["dE2_hyb_k"]) + float(_g(r, "ct2", 0))) for r in R["rec"]])
    out["particles_02"] = np.array([float(trapP(R, r["wdn2_src_inst"])) for r in R["rec"]])
    if "dE1L_p" in R["rec"][0]:
        out["rho_11_per_lambda"] = np.array([complex(trapP(R, r["dE1L_p"]) + complex(_g(r, "contact1", 0))) for r in R["rec"]])
    if "dE21_k" in R["rec"][0]:
        out["rho_12_per_lambda"] = np.array([float(trapP(R, r["dE21_k"]) + float(r["contact2"])) for r in R["rec"]])
    return out


@dataclass
class NumericResult:
    settings: dict
    homogeneous: dict
    families: dict = field(default_factory=dict)        # family -> {q -> series}
    integrated: dict = field(default_factory=dict)      # family -> {name -> series over N}
    bridge: dict = field(default_factory=dict)

    def summary(self, lam=None):
        lines = []
        h = self.homogeneous
        i = -1
        lines.append(f"N_end = {h['N'][i]:g}:  rho_00 = {h['rho_00'][i]:.6e}"
                     + (f",  rho_10/lambda = {h['rho_10_per_lambda'][i]:.6e}" if "rho_10_per_lambda" in h else ""))
        for fam, d in self.integrated.items():
            parts = [f"{k} = {np.asarray(v)[i]:.6e}" for k, v in d.items() if k != "N"]
            lines.append(f"[{fam}] integrated over ln q at N_end: " + ", ".join(parts))
        return "\n".join(lines)


SHARED = ("nend", "local_order", "mu_R", "fsw", "integrator", "init_order", "avg_center")


def evaluate_numeric(m, families=("scalar", "tensor"), q_grid=(0.1, 0.1316, 0.1732, 0.228, 0.3), lam=True,
                     output_dir="inin_numeric", workers=None, log=print, hom_settings=None, **settings):
    """settings override DEFAULTS for the perturbation runs; hom_settings override HOM_DEFAULTS for the homogeneous
    run; the shared keys (nend, local_order, mu_R, fsw, ...) given in settings apply to both."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    hs = {k: v for k, v in settings.items() if k in SHARED}; hs.update(hom_settings or {})
    hom = homogeneous_job(m, out, lam=lam, **hs)
    pjobs = {(fam, q): perturbation_job(fam, q, m, out, lam=lam, **settings) for fam in families for q in q_grid}
    run_jobs([hom] + list(pjobs.values()), workers=workers, log=log)
    res = NumericResult(settings=dict(perturbation=dict(DEFAULTS, **settings), homogeneous=dict(HOM_DEFAULTS, **hs),
                                      m=m, q_grid=list(q_grid), families=list(families)),
                        homogeneous=homogeneous_series(load(hom)))
    for fam in families:
        res.families[fam] = {q: perturbation_series(load(pjobs[(fam, q)])) for q in q_grid}
        series = res.families[fam]
        qs = np.array(list(q_grid))
        integ = dict(N=series[q_grid[0]]["N"])
        for key in ("rho_02", "rho_02_hybrid", "rho_12_per_lambda", "particles_02"):
            if key in series[q_grid[0]]:
                integ[key + "_int"] = int_lnq(qs, np.array([series[q][key] for q in q_grid]))
        res.integrated[fam] = integ
    return res


def save(res: NumericResult, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    def conv(v):
        v = np.asarray(v)
        if np.iscomplexobj(v):
            return {"re": v.real.tolist(), "im": v.imag.tolist()}
        return v.tolist()
    blob = dict(settings=res.settings, bridge=res.bridge,
                homogeneous={k: conv(v) for k, v in res.homogeneous.items()},
                families={f: {str(q): {k: conv(v) for k, v in s.items()} for q, s in d.items()} for f, d in res.families.items()},
                integrated={f: {k: conv(v) for k, v in d.items()} for f, d in res.integrated.items()})
    (out / "rho_results.json").write_text(json.dumps(blob, indent=1, default=str))
    (out / "summary.txt").write_text(res.summary() + "\n")
    return out / "rho_results.json"
