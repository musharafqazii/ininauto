"""Action language of the Higgs portal: dark-sector blocks KIN, PHI2, RPHI2, PHI4 plus Higgs blocks
KINH, H2, RH2, H4 and the portal block H2PHI2."""
from __future__ import annotations

from dataclasses import dataclass

import sympy as s

from ..core.symbols import TexSymbol
from ..scalar.action import KIN, PHI2, RPHI2, PHI4, ScalarAction


# ------------------------------------------------------------------------------------------- action language
KINH = TexSymbol("KINH", tex=r"\left(\nabla h\right)^{2}", uni="(∇h)²")
H2 = TexSymbol("H2", tex=r"h^{2}", uni="h²")
RH2 = TexSymbol("RH2", tex=r"R\,h^{2}", uni="Rh²")
H4 = TexSymbol("H4", tex=r"h^{4}", uni="h⁴")
H2PHI2 = TexSymbol("H2PHI2", tex=r"h^{2}\Phi^{2}", uni="h²Φ²")

_BLOCKS = {"KIN": KIN, "PHI2": PHI2, "RPHI2": RPHI2, "PHI4": PHI4,
           "KINH": KINH, "H2": H2, "RH2": RH2, "H4": H4, "H2PHI2": H2PHI2}


@dataclass(frozen=True)
class PortalAction:
    expression: s.Expr
    dark: ScalarAction          # Phi sector (mass_squared, xi, coupling = lambda)
    higgs: ScalarAction         # h sector, written with the dark building blocks (coupling = lambda_h)
    portal: s.Expr              # lambda_p

    @classmethod
    def parse(cls, expression):
        e = s.sympify(expression, locals=dict(_BLOCKS))
        e = s.expand(e.xreplace({x: _BLOCKS[x.name] for x in e.free_symbols
                                 if x.name in _BLOCKS and not isinstance(x, TexSymbol)}))
        c = {k: e.coeff(b) for k, b in _BLOCKS.items()}
        rest = s.simplify(e - sum(c[k] * b for k, b in _BLOCKS.items()))
        if rest != 0:
            raise ValueError(f"Action contains unsupported operators: {rest}")
        dark = ScalarAction.parse(c["KIN"] * KIN + c["PHI2"] * PHI2 + c["RPHI2"] * RPHI2 + c["PHI4"] * PHI4)
        higgs = ScalarAction.parse(c["KINH"] * KIN + c["H2"] * PHI2 + c["RH2"] * RPHI2 + c["H4"] * PHI4)
        lam_p = s.expand(-4 * c["H2PHI2"])
        for name, g in (("lambda", dark.coupling), ("lambda_h", higgs.coupling), ("lambda_p", lam_p)):
            if g != 0 and not g.is_Symbol:
                raise ValueError(f"{name} must be zero or a single symbol")
        return cls(e, dark, higgs, lam_p)


def portal_action(m=None, m_h=None, lam=None, lam_h=0, lam_p=None):
    """-KIN/2 - m^2 PHI2/2 - RPHI2/12 - lam PHI4/24 - KINH/2 - m_h^2 H2/2 - RH2/12 - lam_h H4/24 - lam_p H2PHI2/4."""
    m = s.Symbol("m", positive=True) if m is None else s.sympify(m)
    m_h = s.Symbol("m_h", positive=True) if m_h is None else s.sympify(m_h)
    lam = s.Symbol("lambda", real=True) if lam is None else s.sympify(lam)
    lam_p = s.Symbol("lambda_p", real=True) if lam_p is None else s.sympify(lam_p)
    half = s.Rational(1, 2)
    return (-half * KIN - m**2 * PHI2 / 2 - RPHI2 / 12 - lam * PHI4 / 24
            - half * KINH - m_h**2 * H2 / 2 - RH2 / 12 - s.sympify(lam_h) * H4 / 24 - lam_p * H2PHI2 / 4)
