"""Map scalar/tensor contractions in jet algebra to ordered QFT operators.

Spatial dummy indices are never conflated across independent insertions;
the Wick backend alpha-renames those indices before contracting.
"""
import sympy as s
from itertools import permutations
from .symbols import *
from .wick import F,S,Poly,Monomial,FieldAtom,SourceAtom,local,vertex

f=F('phi');p=F('phi',dt=1)
fi=F('phi',dx=('i',));fj=F('phi',dx=('j',)); fij=F('phi',dx=('i','j'))
hi=S('h',indices=('i','j'))
hik=S('h',indices=('i','k'))
hkj=S('h',indices=('k','j'))

R3_poly=(S('h',indices=('i','j'))*S('h',indices=('i','j'),dx=('k','k'))
         +s.Rational(3,4)*S('h',indices=('i','j'),dx=('k',))*S('h',indices=('i','j'),dx=('k',))
         -s.Rational(1,2)*S('h',indices=('i','k'),dx=('j',))*S('h',indices=('i','j'),dx=('k',)))

REGISTRY={
  P:f, PI:p, V:p, G:fi*fi, LAP:F('phi',dx=('i','i')),
  PS:S('psi'),PST:S('psi',dt=1),PSTT:S('psi',dt=2),
  PSL:S('psi',dx=('i','i')),
  PSG:S('psi',dx=('i',))*S('psi',dx=('i',)),
  MIX:S('psi',dx=('i',))*fi,
  SH:hi*hi, SHP:hi*S('h',indices=('i','j'),dt=1),
  SHPP:hi*S('h',indices=('i','j'),dt=2),
  HP2:S('h',indices=('i','j'),dt=1)*S('h',indices=('i','j'),dt=1),
  HG:hi*fi*fj,
  CG:hik*hkj*fi*fj,
  HESS:hi*f*fij,
  CHESS:hik*hkj*f*fij,
  DIVC:f*hik*S('h',indices=('k','j'),dx=('i',))*fj,
  HDER:f*S('h',indices=('j','k'))*S('h',indices=('j','k'),dx=('i',))*fi,
  R3:R3_poly,
}

def symmetrize_quantum(poly):
    """Symmetric (Weyl) ordering of quantum fields within each insertion.

    Crucial for mixed phi*pi and phi*lap(phi), whose antisymmetric parts
    produce equal-time contact terms. Background sources are commuting.
    """
    terms=[]
    for mono in poly.monos:
        if mono.coefficient == 0:
            continue
        classical=tuple(a for a in mono.atoms if isinstance(a,SourceAtom))
        quantum=tuple(a for a in mono.atoms if isinstance(a,FieldAtom))
        orders=set(permutations(quantum)) if len(quantum)>1 else {quantum}
        for ordering in sorted(orders,key=str):
            terms.append(Monomial(mono.coefficient/s.Integer(len(orders)),classical+ordering))
    return Poly(tuple(terms))


def to_polynomial(expression):
    """Convert jet monomials to ordered field factors, preserving c-number coefficients."""
    out=Poly.of(0)
    for summand in s.Add.make_args(s.expand(expression)):
        factors=s.Mul.make_args(summand)
        poly=Poly.of(1)
        for factor in factors:
            base,power=factor.as_base_exp()
            if base in REGISTRY and power.is_Integer and power>=0:
                poly=poly*(REGISTRY[base]**int(power))
            elif any(z in factor.free_symbols for z in REGISTRY):
                raise ValueError(f'Unable to lower jet expression to QFT fields: {factor}')
            else:
                poly=poly*factor
        out=out+poly
    return symmetrize_quantum(out)


def to_local(expression,time=T,point='x'):
    return local(to_polynomial(expression),time,point)


def to_vertex(expression,time):
    return vertex(to_polynomial(expression),time)
