"""Higgs-sector fields: chi = a h, its jets (the dark-sector jets with phi -> chi), M_h^2 = a^2 m_h^2,
and their lowering to Wick operators (registered in the scalar operator REGISTRY)."""
from __future__ import annotations

import sympy as s
from sympy.printing.pretty.stringpict import prettyForm

from ..core.symbols import *                 # noqa: F401,F403  (jets, E, M, T, a, trunc)
from ..core.symbols import TexSymbol, _TEX, _UNI
from ..core.wick import Poly, Monomial, FieldAtom
from ..scalar.operators import REGISTRY


# ------------------------------------------------------------------------------------------- Higgs-sector symbols
class MH2(s.Function):
    """M_h^2(tau) = a(tau)^2 m_h^2, printed M_h^2(tau)."""
    is_real = True

    def _latex(self, printer, exp=None):
        base = r"M_{h}^{2}\!\left(" + printer._print(self.args[0]) + r"\right)"
        return base if exp is None else r"\left(" + base + r"\right)^{" + str(exp) + "}"

    def _pretty(self, printer):
        return prettyForm("M_h²(" + str(printer._print(self.args[0])) + ")")


MH = MH2(T)

_PHI_JETS = [P, V, PI, G, LAP, MIX, HG, CG, HESS, CHESS, DIVC, HDER]
_SPECIAL = {"phi": ("chi", r"\chi", "χ"), "v": ("v_chi", r"\chi'", "χ'"), "pi": ("pi_chi", r"\pi_{\chi}", "π_χ")}


def _chi_jet(sym):
    if sym.name in _SPECIAL:
        name, tex, uni = _SPECIAL[sym.name]
    else:
        name = sym.name.replace("phi", "chi")
        tex = _TEX[sym.name].replace(r"\phi", r"\chi")
        uni = _UNI[sym.name].replace("φ", "χ")
    return TexSymbol(name, tex=tex, uni=uni, real=True)


TO_CHI = {j: _chi_jet(j) for j in _PHI_JETS}
PC, VC, PIC, GC, LAPC = (TO_CHI[x] for x in (P, V, PI, G, LAP))


def _rename_poly(poly, old="phi", new="chi"):
    return Poly(tuple(Monomial(m.coefficient,
                               tuple(FieldAtom(new, at.dt, at.dx, at.indices)
                                     if isinstance(at, FieldAtom) and at.name == old else at for at in m.atoms))
                      for m in poly.monos))


REGISTRY.update({TO_CHI[j]: _rename_poly(REGISTRY[j]) for j in _PHI_JETS})


def to_chi(expr):
    """Rename a single-field (dark-sector) jet expression to the Higgs sector: phi -> chi, M^2 -> M_h^2."""
    return s.sympify(expr).xreplace({**TO_CHI, M: MH})
