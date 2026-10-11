"""End-to-end layer: the action -> engine bridge, its ability to detect a wrong kernel, and a tiny numeric run."""
import numpy as np
import sympy as sp
import ininauto.bridge as bridge
from ininauto import KIN, PHI2, RPHI2, PHI4, verify_all
from ininauto.action import ScalarAction
from ininauto.numerics import runner

m, lam = sp.symbols('m lam', real=True)
ACTION = -KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24


def test_bridge_all_kernels_follow_from_the_action():
    for fam, rep in verify_all(ACTION).items():
        assert rep.ok, str(rep)


def test_bridge_detects_a_wrong_kernel(monkeypatch):
    good = bridge._engine_scalar
    def bad(q, p, f, T):
        D1, D2 = good(q, p, f, T)
        return D1 + q**2*T*f/12, D2                      # q^2/3 -> q^2/4 in P_1
    monkeypatch.setattr(bridge, "_engine_scalar", bad)
    rep = bridge.verify(ScalarAction.parse(ACTION), "scalar")
    assert not rep.ok and rep.checks['mode operator P_1 (side band)'] != 0


def test_tiny_homogeneous_numeric_run(tmp_path):
    res = runner.evaluate_numeric(1.0, families=(), output_dir=tmp_path, workers=1, log=lambda *a: None,
                                  hom_settings=dict(np=6, pmax=2.0, nend=-2.0, dtheta=0.1, avg_window=0.0), nend=-2.0)
    h = res.homogeneous
    assert np.all(np.isfinite(h["rho_00"])) and np.all(np.isfinite(h["rho_10_per_lambda"]))
    assert len(h["N"]) >= 3
