"""Conformally coupled real scalar Phi with lambda Phi^4/24 on perturbed FLRW.

action -> expansion (reduced Lagrangian) -> hamiltonian (Legendre) -> stress (Hilbert energy density)
-> operators (jets -> Wick operators) -> inin (Dyson bookkeeping) -> pipeline.run / full.solve / evaluate
(numerical back end in scalar.numerics, kernels verified by scalar.bridge).
"""
from .action import KIN, PHI2, RPHI2, PHI4, ScalarAction, conformal_scalar_action
from .expansion import reduced_lagrangian
from .hamiltonian import legendre
from .stress import hilbert_density
from .operators import to_local, to_vertex, to_polynomial
from .inin import enumerate_diagrams, evaluate_diagrams
from .pipeline import run, Result
from .full import solve, export_completed, FullResult
from .adiabatic import homogeneous_subtraction, AdiabaticSubtraction
from .evaluate import evaluate
from .bridge import verify_all

__all__ = ["KIN", "PHI2", "RPHI2", "PHI4", "ScalarAction", "conformal_scalar_action", "reduced_lagrangian",
           "legendre", "hilbert_density", "to_local", "to_vertex", "to_polynomial", "enumerate_diagrams",
           "evaluate_diagrams", "run", "Result", "solve", "export_completed", "FullResult",
           "homogeneous_subtraction", "AdiabaticSubtraction", "evaluate", "verify_all"]
