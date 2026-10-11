"""Symbols render in physics notation (LaTeX and Unicode) while keeping their code names."""
import pickle
import sympy as sp
from ininauto import KIN, PHI2, RPHI2, PHI4
from ininauto.symbols import E, V, G, PST, M, MIX


def test_latex_of_action_and_jets():
    m, lam = sp.symbols('m lambda', real=True)
    tex = sp.latex(-KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24)
    assert r'\left(\nabla\Phi\right)^{2}' in tex and r'R\,\Phi^{2}' in tex and r'\Phi^{4}' in tex
    assert 'PHI' not in tex and 'KIN' not in tex
    e = sp.latex(E*G/2 + V**2 - PST*V + M + MIX)
    assert e.count('eps') == e.count(r'\epsilon')
    for bad in ('grad', 'psi_t', 'M_{2}'):
        assert bad not in e
    assert r"\phi'" in e and r'\epsilon' in e and r'\nabla\psi\cdot\nabla\phi' in e


def test_code_names_and_pickling_unchanged():
    assert str(G) == 'grad_phi_squared' and str(V) == 'v'
    assert pickle.loads(pickle.dumps(G * V)) == G * V


def test_wick_kernels_render_as_mode_functions():
    from ininauto.wick import U, Ub, SP, SQ, latex_result, WickResult, WickTerm
    k1, q1, t = sp.symbols('k1 q1 tau')
    phi_, psi_ = sp.Symbol('phi'), sp.Symbol('psi')
    e = sp.latex(U(phi_, k1, 1, t)*Ub(phi_, -k1 - q1, 0, t)*SP(k1, k1)*SP(k1, q1)*SQ(psi_, q1, 0, t))
    assert "u'_{k_{1}}" in e and r"u^{*}_{\left|k_{1} + q_{1}\right|}" in e
    assert r"k_{1}^{2}" in e and r"k_{1}\cdot q_{1}" in e and r"\tilde{\psi}" in e
    for bad in ('Ub', 'SP', 'SQ', 'operatorname'):
        assert bad not in e
    assert str(U(phi_, k1, 1, t)) == 'U(phi, k1, 1, tau)'
