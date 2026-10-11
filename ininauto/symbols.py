"""Commutative jet symbols; explicit contracted derivatives become operators in operators.py."""
import sympy as s
E=s.Symbol('eps')
lam=s.Symbol('lambda')
T=s.Symbol('tau',real=True)
a=s.Function('a')(T)
M=s.Function('M2')(T)          # M^2 = a(tau)^2 m^2: time dependent; relabelled at each vertex time
P=s.Symbol('phi',real=True)
V=s.Symbol('v',real=True)
PI=s.Symbol('pi',real=True)
G=s.Symbol('grad_phi_squared',real=True)
LAP=s.Symbol('phi_lap_phi',real=True)
PS=s.Symbol('psi',real=True)
PST=s.Symbol('psi_t',real=True)
PSTT=s.Symbol('psi_tt',real=True)
PSL=s.Symbol('lap_psi',real=True)
PSG=s.Symbol('grad_psi_squared',real=True)
MIX=s.Symbol('grad_psi_grad_phi',real=True)
SH=s.Symbol('h_ij_h_ij',real=True)
SHP=s.Symbol('h_ij_hprime_ij',real=True)
SHPP=s.Symbol('h_ij_hdoubleprime_ij',real=True)
HP2=s.Symbol('hprime_ij_hprime_ij',real=True)
R3=s.Symbol('R3_tensor_quadratic',real=True)
HG=s.Symbol('h_ij_phi_i_phi_j',real=True)
CG=s.Symbol('h_ik_h_kj_phi_i_phi_j',real=True)
HESS=s.Symbol('h_ij_phi_phi_ij',real=True)
CHESS=s.Symbol('h_ik_h_kj_phi_phi_ij',real=True)
DIVC=s.Symbol('phi_h_ik_d_i_h_kj_phi_j',real=True)
HDER=s.Symbol('phi_h_jk_d_i_h_jk_phi_i',real=True)


def trunc(expr,n=2):
    """Expand only the metric-counting epsilon; never expand non-polynomial time functions."""
    return s.expand(s.series(expr,E,0,n+1).removeO())


def coeffs(expr,n=2):
    expr=trunc(expr,n)
    return {i:s.expand(expr.coeff(E,i)) for i in range(n+1)}
