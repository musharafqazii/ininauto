"""Full calculation; may require extensive computation and memory."""
import sympy as s
from ininauto import KIN, PHI2, RPHI2, PHI4, solve
m, lam=s.symbols('m lam',real=True)
action=-KIN/2-m**2*PHI2/2-RPHI2/12-lam*PHI4/24
result=solve(action,output_dir='full_results',workers=1)
print(result.status)
