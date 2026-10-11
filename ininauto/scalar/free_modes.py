"""Free Gaussian mode equations and Wightman/retarded objects."""
import sympy as s

def scalar_modes(action):
    tau,tau0,k=s.symbols('tau tau0 k',real=True)
    a=s.Function('a')(tau)
    u=s.Function('u')(k,tau)
    us=s.Function('u_bar')(k,tau)
    m2=action.mass_squared
    eq=s.Eq(s.diff(u,tau,2)+(k**2+a**2*m2)*u,0)
    W=s.Eq(u*s.diff(us,tau)-s.diff(u,tau)*us,s.I)
    # Independent symbolic time variables ensure order of Wightman slots.
    t,s0=s.symbols('t s',real=True)
    Gplus=s.Function('u')(k,t)*s.Function('u_bar')(k,s0)
    GR=s.Heaviside(t-s0)*s.I*(Gplus-s.Function('u_bar')(k,t)*s.Function('u')(k,s0))
    return {'mode_equation':eq,'wronskian':W,'Wightman_k':Gplus,'retarded_k':GR}
