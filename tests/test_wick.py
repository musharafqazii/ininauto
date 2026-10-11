import sympy as s
from ininauto.wick import *

def test_four_field_pairings():
    phi=F('phi')
    w=wick_vev([local(phi**4,s.Symbol('t'))])
    assert len(w.terms)==1
    k1,k2=s.symbols('k1 k2'); field=s.Symbol('phi'); t=s.Symbol('t')
    assert s.simplify(w.terms[0].integrand-
        3*U(field,k1,0,t)*Ub(field,k1,0,t)*U(field,k2,0,t)*Ub(field,k2,0,t))==0

def test_six_connected_wick_pairs():
    phi=F('phi');t,t1,t2=s.symbols('t t1 t2')
    from ininauto.wick import _pairings,_is_connected,Leg,FieldAtom
    legs=[Leg(FieldAtom('phi'),i//2,s.Symbol('t'),s.Symbol('x')) for i in range(6)]
    assert len(_pairings(6))==15
    assert sum(_is_connected(3,p,legs) for p in _pairings(6))==8

def test_nested_commutator_sign():
    phi=F('phi'); t,t1=s.symbols('t t1')
    o=local(phi*phi,t);h=vertex(S('psi')*phi*phi,t1)
    x=nested_commutator_vev([o,h],connected=True)
    y=wick_vev([o,h],connected=True)-wick_vev([h,o],connected=True)
    assert len(x.terms)==len(y.terms)
    for a,b in zip(x.terms,y.terms):
        assert s.expand(a.integrand-b.integrand)==0
