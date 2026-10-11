import sympy as sp
import pytest
from ininauto.portal import higgs_portal, portal_action, portal_diagrams, checks, PortalAction, H2PHI2
from ininauto.wick import U, Ub


@pytest.mark.parametrize("family", ["scalar", "tensor"])
def test_portal_checks(family):
    S = higgs_portal(portal_action(), family)
    for name, ok in checks(S):
        assert ok, name


def test_parse_rejects_bad_portal():
    with pytest.raises(ValueError):
        PortalAction.parse(portal_action() + sp.Symbol("x") * H2PHI2 * sp.Symbol("y"))


def test_flat_order_lambda_p():
    S = higgs_portal(portal_action(), "scalar")
    out = dict(portal_diagrams(S, (0, 1, 0)))
    contact = out["rho010"]
    assert len(contact) == 1
    t = contact.terms[0]
    assert len(t.loops) == 2
    # Hartree factorisation: (1/4a^4) <phi^2><chi^2>, each loop |u_k|^2
    integ = sp.expand(t.integrand)
    for m in sp.Add.make_args(integ):
        species = sorted(str(f.args[0]) for f in m.atoms(U))
        assert species == ["chi", "phi"]
    ham = out["rho000 H010(t1)"]
    assert len(ham) == 1
    # every term carries one equal-time tadpole at the vertex, U(s,k,0,t1) Ub(s,k,0,t1), whose loop momentum k
    # appears nowhere else (Sigma_Phi from the chi loop, Sigma_chi from the phi loop)
    t1 = sp.Symbol("t1", real=True)
    for m in sp.Add.make_args(sp.expand(ham.terms[0].integrand)):
        ok = False
        for f in m.atoms(U):
            s_, k, n, t = f.args
            if t == t1 and n == 0 and Ub(s_, k, 0, t1) in m.atoms(Ub):
                rest = m / (f * Ub(s_, k, 0, t1))
                ok = ok or not rest.has(k)
        assert ok
