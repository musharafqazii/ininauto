"""Free Dirac field on perturbed FLRW (scalar Newtonian gauge, tensor TT), explicit to O(eps^2).

    from ininauto.dirac import DiracField, inin_term
    D = DiracField("scalar", order=2)          # or "tensor"
    D.H(1), D.rho_n(2)                         # interaction Hamiltonian / energy density bilinears
    display(Math(D.clifford(D.H(1))))          # Psi^dag [ sum c_A Gamma_A ] Psi form
    T = inin_term(D, 0, (1,))                  # -i <[rho_0(x), H_1(t1)]>  with explicit fermion Wick contractions
    display(Math(T.latex()))

Majorana field (psi = i gamma^2 psi^*):
    M = MajoranaField("scalar", order=2)
    T = majorana_inin_term(M, 0, (1,))         # explicit Majorana Wick contractions, incl. <psi psi>
"""
from .core import DiracField, GAM, G0, G5, clifford_basis, dagger
from .wick import inin_term, loop_word, make_bilinear, P_matrix, N_matrix_minus, DiracTerm, reduce_to_invariants
from .majorana import MajoranaField, majorana_inin_term, majorana_word

__all__ = ["MajoranaField", "majorana_inin_term", "majorana_word", "DiracField", "inin_term", "loop_word", "make_bilinear", "P_matrix", "N_matrix_minus", "DiracTerm",
           "reduce_to_invariants", "GAM", "G0", "G5", "clifford_basis", "dagger"]
