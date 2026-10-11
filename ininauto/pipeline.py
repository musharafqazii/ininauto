"""Orchestrates every stage, with the covariant action as the sole required input."""
from dataclasses import dataclass
from pathlib import Path
import sympy as s
from .action import ScalarAction
from .geometry import scalar_geometry,tensor_geometry
from .expansion import reduced_lagrangian
from .hamiltonian import legendre
from .stress import hilbert_density
from .inin import enumerate_diagrams,evaluate_diagrams
from .renormalization import prescription
from .free_modes import scalar_modes
from .report import write_reports

@dataclass
class Result:
    action:ScalarAction
    families:dict
    files:Path


def run(action,output_dir='results',*,families=('scalar','tensor'),
        wick_max_legs=4,wick_max_insertions=1, max_insertions=3):
    """Run covariant scalar-field in--in derivation; only ``action`` is required.

    For full Gaussian Wick expansion (possibly very expensive), set
    wick_max_legs=None, wick_max_insertions=None.
    Renormalized *finite* kernels remain symbolic until supplied by a
    state/background-dependent CTP/Hadamard computation.
    """
    action=action if isinstance(action,ScalarAction) else ScalarAction.parse(action)
    if action.coupling!=0 and not action.coupling.is_Symbol:
        raise ValueError('Quartic interaction must be zero or a single symbol, e.g. lambda')
    data={}
    for family in families:
        g=scalar_geometry() if family=='scalar' else tensor_geometry()
        lag=reduced_lagrangian(action,family)
        h=legendre(lag,action.coupling)
        rho=hilbert_density(family,h,action)
        # The free H_00 is the reference Hamiltonian, not an interaction vertex.
        V={k:v for k,v in h.interaction_coeffs.items() if k!=(0,0)}
        O=rho.coefficients
        jobs=enumerate_diagrams(O,V,max_insertions=max_insertions)
        diags=evaluate_diagrams(jobs,O,V,max_wick_legs=wick_max_legs,
                       max_wick_insertions=wick_max_insertions)
        counterterms=prescription(action,family)
        data[family]=(h,rho,g,lag,diags,counterterms)
    files=write_reports(output_dir,action,data,scalar_modes(action))
    return Result(action,data,files)
