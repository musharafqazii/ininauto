"""Commutative jet symbols; explicit contracted derivatives become operators in operators.py.

Every jet symbol keeps a short code name (used by str(), by the code and in stored results) and carries its own
LaTeX and Unicode form, so that sympy.latex(...), Jupyter display and pretty printing show the physics notation,
e.g. grad_phi_squared -> (nabla phi)^2, v -> phi', psi_t -> psi', eps -> epsilon.
"""
import sympy as s
from sympy.printing.pretty.stringpict import prettyForm

_TEX = {}
_UNI = {}


class TexSymbol(s.Symbol):
    """A Symbol with its own LaTeX/Unicode rendering (code name unchanged; pickles and compares like a Symbol of
    this class)."""

    def __new__(cls, name, tex=None, uni=None, **assumptions):
        if tex is not None:
            _TEX[name] = tex
        if uni is not None:
            _UNI[name] = uni
        return s.Symbol.__new__(cls, name, **assumptions)

    def _latex(self, printer):
        return _TEX.get(self.name, self.name)

    def _pretty(self, printer):
        return prettyForm(_UNI.get(self.name, self.name))


class M2(s.Function):
    """M^2(tau) = a(tau)^2 m^2 (an undetermined function of time, printed as M^2(tau))."""
    is_real = True

    def _latex(self, printer, exp=None):
        arg = printer._print(self.args[0])
        base = r"M^{2}\!\left(" + arg + r"\right)"
        return base if exp is None else r"\left(" + base + r"\right)^{" + str(exp) + "}"

    def _pretty(self, printer):
        return prettyForm("M²(" + str(printer._print(self.args[0])) + ")")


def _ts(name, tex, uni, real=True):
    return TexSymbol(name, tex=tex, uni=uni, real=real) if real else TexSymbol(name, tex=tex, uni=uni)


E=_ts('eps', r'\epsilon', 'ε', real=False)
lam=s.Symbol('lambda')
T=s.Symbol('tau',real=True)
a=s.Function('a')(T)
M=M2(T)          # M^2 = a(tau)^2 m^2: time dependent; relabelled at each vertex time
P=_ts('phi', r'\phi', 'φ')
V=_ts('v', r"\phi'", "φ'")
PI=_ts('pi', r'\pi', 'π')
G=_ts('grad_phi_squared', r'\left(\nabla\phi\right)^{2}', '(∇φ)²')
LAP=_ts('phi_lap_phi', r'\phi\,\nabla^{2}\phi', 'φ∇²φ')
PS=_ts('psi', r'\psi', 'ψ')
PST=_ts('psi_t', r"\psi'", "ψ'")
PSTT=_ts('psi_tt', r"\psi''", "ψ''")
PSL=_ts('lap_psi', r'\nabla^{2}\psi', '∇²ψ')
PSG=_ts('grad_psi_squared', r'\left(\nabla\psi\right)^{2}', '(∇ψ)²')
MIX=_ts('grad_psi_grad_phi', r'\left(\nabla\psi\cdot\nabla\phi\right)', '(∇ψ·∇φ)')
SH=_ts('h_ij_h_ij', r'\left(h_{ij}h_{ij}\right)', '(hᵢⱼhᵢⱼ)')
SHP=_ts('h_ij_hprime_ij', r"\left(h_{ij}h'_{ij}\right)", "(hᵢⱼh'ᵢⱼ)")
SHPP=_ts('h_ij_hdoubleprime_ij', r"\left(h_{ij}h''_{ij}\right)", "(hᵢⱼh''ᵢⱼ)")
HP2=_ts('hprime_ij_hprime_ij', r"\left(h'_{ij}h'_{ij}\right)", "(h'ᵢⱼh'ᵢⱼ)")
R3=_ts('R3_tensor_quadratic', r'{}^{(3)}\!R^{(2)}', '³R⁽²⁾')
HG=_ts('h_ij_phi_i_phi_j', r'\left(h_{ij}\partial_i\phi\,\partial_j\phi\right)', '(hᵢⱼ∂ᵢφ∂ⱼφ)')
CG=_ts('h_ik_h_kj_phi_i_phi_j', r'\left(h_{ik}h_{kj}\partial_i\phi\,\partial_j\phi\right)', '(hᵢₖhₖⱼ∂ᵢφ∂ⱼφ)')
HESS=_ts('h_ij_phi_phi_ij', r'\left(h_{ij}\,\phi\,\partial_i\partial_j\phi\right)', '(hᵢⱼφ∂ᵢ∂ⱼφ)')
CHESS=_ts('h_ik_h_kj_phi_phi_ij', r'\left(h_{ik}h_{kj}\,\phi\,\partial_i\partial_j\phi\right)', '(hᵢₖhₖⱼφ∂ᵢ∂ⱼφ)')
DIVC=_ts('phi_h_ik_d_i_h_kj_phi_j', r'\left(\phi\,h_{ik}\,\partial_i h_{kj}\,\partial_j\phi\right)', '(φhᵢₖ∂ᵢhₖⱼ∂ⱼφ)')
HDER=_ts('phi_h_jk_d_i_h_jk_phi_i', r'\left(\phi\,h_{jk}\,\partial_i h_{jk}\,\partial_i\phi\right)', '(φhⱼₖ∂ᵢhⱼₖ∂ᵢφ)')


def trunc(expr,n=2):
    """Expand only the metric-counting epsilon; never expand non-polynomial time functions."""
    return s.expand(s.series(expr,E,0,n+1).removeO())


def coeffs(expr,n=2):
    expr=trunc(expr,n)
    return {i:s.expand(expr.coeff(E,i)) for i in range(n+1)}
