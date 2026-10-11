"""Package layout: three sectors on a shared core, top-level re-exports, and pre-0.7 import paths as aliases."""
import importlib
import ininauto


def test_version_and_sectors():
    assert ininauto.__version__ == "0.7.0"
    for sub in ("core", "scalar", "fermionic", "higgs_portal"):
        assert importlib.import_module(f"ininauto.{sub}").__name__ == f"ininauto.{sub}"


def test_legacy_paths_are_aliases():
    pairs = {"ininauto.wick": "ininauto.core.wick", "ininauto.symbols": "ininauto.core.symbols",
             "ininauto.action": "ininauto.scalar.action", "ininauto.numerics.runner": "ininauto.scalar.numerics.runner",
             "ininauto.dirac": "ininauto.fermionic", "ininauto.dirac.majorana": "ininauto.fermionic.majorana",
             "ininauto.portal": "ininauto.higgs_portal"}
    for old, new in pairs.items():
        assert importlib.import_module(old) is importlib.import_module(new), old
    from ininauto.portal import checks            # 0.6 notebook style
    from ininauto.dirac import DiracField         # 0.4/0.5 notebook style
    assert callable(checks) and DiracField is ininauto.DiracField


def test_portal_package_is_callable():
    from ininauto import higgs_portal, portal_action, portal_system
    S = portal_action()
    assert higgs_portal(S, "scalar").H == portal_system(S, "scalar").H
