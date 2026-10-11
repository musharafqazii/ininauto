"""ininauto: in-in perturbation theory on perturbed FLRW, starting from the action.

Three sectors on one shared core:

    ininauto.core          jet symbols, metric families (Newtonian gauge / TT), bosonic Wick engine
    ininauto.scalar        conformal scalar Phi with lambda Phi^4  (action -> H -> rho -> Wick -> numbers)
    ininauto.fermionic     free Dirac and Majorana fields (explicit spinor algebra, fermion loops)
    ininauto.higgs_portal  dark scalar Phi + Higgs-like h with (lambda_p/4) h^2 Phi^2

Basic usage (everything is also re-exported here):
    from ininauto import *
    import sympy as sp
    m, lam = sp.symbols('m lambda', real=True)
    action = -KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24
    result = run(action)                       # symbolic preview
    res = evaluate(action, values={'m': 1.0})  # end-to-end numbers

Old module paths (ininauto.wick, ininauto.action, ininauto.dirac, ininauto.portal, ...) still import, as aliases
of their new locations.
"""
__version__ = "0.7.0"

import importlib as _importlib
import importlib.abc as _abc
import importlib.util as _util
import sys as _sys

_LEGACY = {f"ininauto.{m}": f"ininauto.core.{m}" for m in ("symbols", "geometry", "wick")}
_LEGACY.update({f"ininauto.{m}": f"ininauto.scalar.{m}" for m in (
    "action", "expansion", "hamiltonian", "stress", "operators", "inin", "pipeline", "full", "adiabatic",
    "evaluate", "bridge", "counterterm_operators", "renormalization", "free_modes", "report", "numerics",
    "numerics.engine", "numerics.runner")})
_LEGACY.update({"ininauto.dirac": "ininauto.fermionic", "ininauto.dirac.core": "ininauto.fermionic.dirac",
                "ininauto.dirac.wick": "ininauto.fermionic.wick", "ininauto.dirac.majorana": "ininauto.fermionic.majorana",
                "ininauto.portal": "ininauto.higgs_portal"})


class _LegacyAlias(_abc.MetaPathFinder, _abc.Loader):
    """Resolve pre-0.7 module paths to the reorganised modules (same module objects, nothing duplicated)."""

    def find_spec(self, name, path=None, target=None):
        return _util.spec_from_loader(name, self) if name in _LEGACY else None

    def create_module(self, spec):
        return _importlib.import_module(_LEGACY[spec.name])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _LegacyAlias) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _LegacyAlias())

from . import core, scalar, higgs_portal                          # noqa: E402
from .core.wick import (F, S, anti, GaussianState, local, vertex, wick_vev, nested_commutator_vev,  # noqa: E402
                        latex_result)
from .scalar import (KIN, PHI2, RPHI2, PHI4, ScalarAction, conformal_scalar_action, run, Result, solve,  # noqa: E402
                     export_completed, FullResult, homogeneous_subtraction, AdiabaticSubtraction, evaluate,
                     verify_all)
from .higgs_portal import (KINH, H2, RH2, H4, H2PHI2, PortalAction, PortalSystem, portal_action,  # noqa: E402
                           portal_system, portal_inin, portal_diagrams)


def __getattr__(name):
    # the fermionic sector (sympy spinor algebra) is imported on first use
    if name == "fermionic":
        return _importlib.import_module("ininauto.fermionic")
    if name in ("DiracField", "MajoranaField", "inin_term", "majorana_inin_term"):
        return getattr(_importlib.import_module("ininauto.fermionic"), name)
    raise AttributeError(f"module 'ininauto' has no attribute {name!r}")


__all__ = ["__version__", "core", "scalar", "higgs_portal", "evaluate", "verify_all", "KIN", "PHI2", "RPHI2",
           "PHI4", "ScalarAction", "conformal_scalar_action", "run", "Result", "solve", "export_completed",
           "FullResult", "homogeneous_subtraction", "AdiabaticSubtraction", "F", "S", "anti", "GaussianState",
           "local", "vertex", "wick_vev", "nested_commutator_vev", "latex_result", "KINH", "H2", "RH2", "H4",
           "H2PHI2", "PortalAction", "PortalSystem", "portal_action", "portal_system", "portal_inin", "portal_diagrams"]
