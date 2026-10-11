"""End-to-end: the action in, renormalised <rho> coefficients out (rho_00 ... rho_12)."""
import sympy as sp
from ininauto import KIN, PHI2, RPHI2, PHI4, evaluate

m, lam = sp.symbols('m lam', real=True)
action = -KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24

if __name__ == '__main__':
    res = evaluate(action, values={'m': 1.0}, output_dir='results')   # lambda symbolic: O(lambda) per unit lambda
    print(res.summary())
