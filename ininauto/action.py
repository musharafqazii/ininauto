"""A deliberately restricted but explicit covariant action language.

KIN = integral sqrt(-g) g^{mu nu} d_mu Phi d_nu Phi
PHI2 = integral sqrt(-g) Phi^2
RPHI2 = integral sqrt(-g) R Phi^2
PHI4 = integral sqrt(-g) Phi^4

Only a single real scalar with canonical kinetic term and quartic potential is
supported.  All signs refer to (-+++).  Do not silently parse arbitrary actions.
"""
from dataclasses import dataclass
import sympy as s

KIN, PHI2, RPHI2, PHI4 = s.symbols('KIN PHI2 RPHI2 PHI4')

@dataclass(frozen=True)
class ScalarAction:
    expression: s.Expr
    mass_squared: s.Expr
    xi: s.Expr
    coupling: s.Expr

    @classmethod
    def parse(cls, expression):
        expression=s.expand(s.sympify(expression))
        k=expression.coeff(KIN)
        m=expression.coeff(PHI2)
        r=expression.coeff(RPHI2)
        c=expression.coeff(PHI4)
        unknown=s.simplify(expression-k*KIN-m*PHI2-r*RPHI2-c*PHI4)
        if unknown!=0:
            raise ValueError(f'Action contains unsupported operators: {unknown}')
        if s.simplify(k+s.Rational(1,2))!=0:
            raise ValueError('Only canonical kinetic normalization -KIN/2 is supported')
        if s.simplify(r+s.Rational(1,12))!=0:
            raise ValueError('The physical-density and renormalization modules currently require xi=1/6')
        return cls(expression, -2*m,-2*r,-24*c)


def conformal_scalar_action(mass_squared=None, coupling=None):
    """Convenience constructor returning the input *action* expression."""
    m2=s.Symbol('m',positive=True)**2 if mass_squared is None else s.sympify(mass_squared)
    lam=s.Symbol('lambda',real=True) if coupling is None else s.sympify(coupling)
    return -s.Rational(1,2)*KIN -m2*PHI2/2-RPHI2/12-lam*PHI4/24
