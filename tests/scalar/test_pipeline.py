import sympy as s
import pytest
from ininauto import *
from ininauto.core.symbols import *
from ininauto.core.geometry import scalar_geometry,tensor_geometry
from ininauto.scalar.expansion import reduced_lagrangian
from ininauto.scalar.hamiltonian import legendre
from ininauto.scalar.stress import hilbert_density
from ininauto.scalar.operators import to_polynomial,to_local
from ininauto.scalar.inin import enumerate_diagrams
from ininauto.scalar.renormalization import prescription
from ininauto.core.wick import U,Ub,SQ,SP

ACTION=ScalarAction.parse(conformal_scalar_action())


def eq(got,want):
    assert s.simplify(s.expand(got-want))==0, (s.expand(got-want))


def test_action_is_only_required_input():
    assert ACTION.xi==s.Rational(1,6)
    assert ACTION.mass_squared==s.Symbol('m',positive=True)**2
    with pytest.raises(ValueError,match='xi'):
        ScalarAction.parse(-KIN/2-PHI2/2-RPHI2/4)
    with pytest.raises(ValueError,match='unsupported'):
        ScalarAction.parse(-KIN/2-RPHI2/12-s.Symbol('NEW'))


def test_scalar_curvature():
    geom=scalar_geometry()
    expected=E*(-6*PSTT+2*PSL)+E**2*(16*PS*PSL+6*PST**2+10*PSG)
    eq(geom.ricci,expected)
    eq(geom.einstein00,2*E*PSL+E**2*(12*PS*PSL+3*PSG+3*PST**2))


def test_scalar_hamiltonian_coefficients_reference():
    l=reduced_lagrangian(ACTION,'scalar')
    h=legendre(l,ACTION.coupling)
    eq(h.interaction_coeffs[(0,1)],2*PS*PI**2+PST*P*PI-P*MIX/3-M*PS*P**2)
    eq(h.interaction_coeffs[(0,2)],5*PS**2*PI**2+2*PS*PST*P*PI
         -PS**2*G-2*PS*P*MIX-M*PS**2*P**2-PSG*P**2/6)
    eq(h.interaction_coeffs[(1,0)],P**4/24)
    eq(h.interaction_coeffs[(1,1)],-PS*P**4/12)
    eq(h.interaction_coeffs[(1,2)],-PS**2*P**4/12)


def test_scalar_hilbert_density_reference():
    h=legendre(reduced_lagrangian(ACTION,'scalar'),ACTION.coupling)
    r=hilbert_density('scalar',h,ACTION)
    eq(r.coefficients[(0,0)]*a**4,PI**2/2+G/6+M*P**2/2-P*LAP/3)
    eq(r.coefficients[(0,1)]*a**4,3*PS*PI**2-2*PS*P*LAP/3+PS*G/3
       +PSL*P**2/3+P*MIX/3)
    eq(r.coefficients[(0,2)]*a**4,12*PS**2*PI**2-4*PS**2*P*LAP/3
       +2*PS**2*G/3+4*PS*P*MIX/3+4*PS*PSL*P**2/3+PSG*P**2/2)
    eq(r.coefficients[(1,0)]*a**4,P**4/24)


def test_tensor_hamiltonian_reference():
    h=legendre(reduced_lagrangian(ACTION,'tensor'),ACTION.coupling)
    eq(h.interaction_coeffs[(0,1)],-HG/2)
    eq(h.interaction_coeffs[(0,2)],SH*(PI**2-G-M*P**2)/8+CG/2
       +(R3/12-SHPP/12-HP2/16)*P**2)
    eq(h.interaction_coeffs[(1,2)],-SH*P**4/96)
    assert (1,1) not in h.interaction_coeffs


def test_tensor_density_reference():
    h=legendre(reduced_lagrangian(ACTION,'tensor'),ACTION.coupling)
    r=hilbert_density('tensor',h,ACTION)
    eq(r.coefficients[(0,1)]*a**4,-HG/6+HESS/3)
    eq(r.coefficients[(0,2)]*a**4,
        SH*PI**2/4+CG/6-CHESS/3-DIVC/3+HDER/6
        +(R3/12-HP2/48)*P**2-SHP*P*PI/6)


def test_scalar_inin_counts_reference():
    for family,expected in [('scalar',(1,2,4,2,5,14)),('tensor',(1,2,4,2,4,11))]:
        h=legendre(reduced_lagrangian(ACTION,family),ACTION.coupling)
        rho=hilbert_density(family,h,ACTION)
        terms=enumerate_diagrams(rho.coefficients,
             {k:v for k,v in h.interaction_coeffs.items() if k!=(0,0)})
        assert tuple(len(z) for z in terms.values())==expected
        if family=='scalar':
            cubic=[s0 for s0 in terms[(1,2)] if len(s0[1])==3]
            assert len(cubic)==3
            assert sorted(tuple(s0[1]) for s0 in cubic)==sorted([
                ((1,0),(0,1),(0,1)),((0,1),(1,0),(0,1)),((0,1),(0,1),(1,0))])


def test_wick_local_and_symmetrization():
    h=legendre(reduced_lagrangian(ACTION,'scalar'),ACTION.coupling)
    r=hilbert_density('scalar',h,ACTION)
    op=to_local(r.coefficients[(0,0)])
    vev=wick_vev([op])
    assert len(vev.terms)==1
    k=s.Symbol('k1');name=s.Symbol('phi')
    u=U(name,k,0,T)*Ub(name,k,0,T)
    up=U(name,k,1,T)*Ub(name,k,1,T)
    want=(up+(SP(k,k)+M)*u)/(2*a**4)
    eq(vev.terms[0].integrand,want)
    # Phi pi is explicitly symmetrized: both orderings occur with 1/2.
    h1=to_polynomial(h.interaction_coeffs[(0,1)])
    pp=[mo for mo in h1.monos if any(getattr(at,'dt',-1)==1 for at in mo.atoms)
        and any(getattr(at,'dt',-1)==0 and getattr(at,'name',None)=='phi' for at in mo.atoms)]
    assert any(tuple(at.dt for at in mo.atoms if getattr(at,'name',None)=='phi')==(0,1) for mo in pp)
    assert any(tuple(at.dt for at in mo.atoms if getattr(at,'name',None)=='phi')==(1,0) for mo in pp)


def test_quartic_wick_factor():
    h=legendre(reduced_lagrangian(ACTION,'scalar'),ACTION.coupling)
    r=hilbert_density('scalar',h,ACTION)
    w=wick_vev([to_local(r.coefficients[(1,0)])])
    assert len(w.terms)==1
    integrand=w.terms[0].integrand
    expected=s.Rational(1,8)/a**4
    for j in (1,2):
        k=s.Symbol(f'k{j}')
        expected*=U(s.Symbol('phi'),k,0,T)*Ub(s.Symbol('phi'),k,0,T)
    eq(integrand,expected)


def test_renormalization_is_explicitly_formal():
    ct=prescription(ACTION,'scalar')
    assert 'Hadamard' in ct.conditions
    assert ct.mass_counterterm.has(ACTION.coupling)
    assert set(ct.finite_coefficients)=={(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}


def test_full_pipeline_smoke(tmp_path):
    res=run(ACTION,output_dir=tmp_path)
    assert (tmp_path/'derivation.tex').is_file()
    assert (tmp_path/'derivation.txt').is_file()
    assert (tmp_path/'results.json').is_file()
    assert len(res.families['scalar'][4][(1,2)])==14
    assert len(res.families['tensor'][4][(1,2)])==11


def test_free_scalar_is_supported(tmp_path):
    action=-KIN/2-RPHI2/12-s.Symbol('m2')*PHI2/2
    res=run(action,tmp_path,families=('scalar',))
    H,rho,g,L,diagrams,ct=res.families['scalar']
    assert all(degree[0]==0 for degree in H.interaction_coeffs)
    assert all(not ds for degree,ds in diagrams.items() if degree[0]==1)
