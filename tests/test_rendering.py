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
