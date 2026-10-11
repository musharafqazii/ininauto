"""Automated conformally coupled scalar QFT: action -> canonical ops -> CTP.

Basic usage:
    from ininauto import *
    import sympy as sp
    m, lam = sp.symbols('m lambda', real=True)
    action = -KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24
    result = run(action)                       # symbolic preview
    res = evaluate(action, values={'m': 1.0})  # end-to-end numbers (rho_00 ... rho_12)
"""
from .action import (KIN, PHI2, RPHI2, PHI4, ScalarAction,conformal_scalar_action)
from .pipeline import run,Result
from .full import solve,export_completed,FullResult
from .adiabatic import homogeneous_subtraction,AdiabaticSubtraction
from .evaluate import evaluate
from .bridge import verify_all
from .wick import (F,S,anti,GaussianState,local,vertex,wick_vev,
                   nested_commutator_vev,latex_result)
from .portal import (KINH,H2,RH2,H4,H2PHI2,PortalAction,PortalSystem,portal_action,higgs_portal,
                     portal_inin,portal_diagrams)
__all__=['evaluate','verify_all','KIN','PHI2','RPHI2','PHI4','ScalarAction','conformal_scalar_action',
         'run','Result','solve','export_completed','FullResult','homogeneous_subtraction','AdiabaticSubtraction','F','S','anti','GaussianState','local','vertex',
         'wick_vev','nested_commutator_vev','latex_result',
         'KINH','H2','RH2','H4','H2PHI2','PortalAction','PortalSystem','portal_action','higgs_portal',
         'portal_inin','portal_diagrams']
