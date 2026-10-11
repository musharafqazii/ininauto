import json
import sqlite3
import sympy as sp
from ininauto import solve, KIN,PHI2,RPHI2,PHI4, homogeneous_subtraction
from ininauto.full import _read_jobs,_worker
from ininauto.action import ScalarAction
from ininauto.wick import _pairings
from ininauto.counterterm_operators import local_counterterms

def action():
    m,lam=sp.symbols('m lambda',real=True)
    return -KIN/2-m*m*PHI2/2-RPHI2/12-lam*PHI4/24


def test_all_diagrams_scheduled():
    jobs,_=_read_jobs(ScalarAction.parse(action()),('scalar','tensor'))
    assert len(jobs)==sum((1,2,4,2,5,14))+sum((1,2,4,2,4,11))
    assert len({j.identifier for j in jobs})==len(jobs)
    assert max(len(j.vertices) for j in jobs)==3


def test_exact_wick_unbounded_pairings():
    assert len(_pairings(10))==945
    a=ScalarAction.parse(action())
    jobs,_=_read_jobs(a,('scalar',))
    out,_=_worker(jobs[0],True)
    assert out['wick_terms']
    assert 'momentum_constraints' in out['wick_terms'][0]


def test_resume(tmp_path):
    a=action()
    run=solve(a,tmp_path,families=('scalar',),stop_after=1,progress=False)
    assert run.status['complete']==1
    run=solve(a,tmp_path,families=('scalar',),stop_after=1,progress=False)
    assert run.status['complete']==2
    with sqlite3.connect(tmp_path/'diagrams.sqlite') as db:
        n=db.execute('select count(*) from diagrams where status="complete"').fetchone()[0]
        assert n==2
    lines=(tmp_path/'expanded_wick.jsonl').read_text().splitlines()
    assert len(lines)==2
    assert len(json.loads(lines[0])['wick_terms'])>0


def test_adiabatic_counterterms():
    out=homogeneous_subtraction()
    assert any(sym.name=='k' for sym in out.energy_subtraction.free_symbols)
    assert out.mode_equation.lhs.has(sp.Function('u'))
    assert out.energy_finite_integral.has(sp.Integral)


def test_counterterms_are_actual_vertices():
    a=ScalarAction.parse(action())
    ct=local_counterterms(a,'scalar')
    z=sp.Symbol('zeta', real=True)
    f=sp.Function('f_sw')(sp.Symbol('tau',real=True))
    assert sp.simplify(ct.mass_shift_per_lambda-f*a.mass_squared/(32*sp.pi**2*z))==0
    assert set(ct.interaction)=={(1,0),(1,1),(1,2)}
    assert set(ct.observable)=={(1,0)}
    tensor=local_counterterms(a,'tensor')
    assert (1,1) not in tensor.interaction
    jobs,_=_read_jobs(a,('scalar',))
    jt=next(j for j in jobs if j.degree==(1,0) and j.observable=='V')
    assert jt.observable_expression.has(z)


def test_counterterms_constant_momentum():
    a=ScalarAction.parse(action())
    jobs,_=_read_jobs(a,('scalar',))
    j=next(j for j in jobs if j.identifier=='scalar:1:0:000')
    result,_=_worker(j,True)
    assert any(not term['loop_momenta'] for term in result['wick_terms'])


def test_adiabatic_constant_background():
    t=sp.Symbol('tau',positive=True,real=True)
    out=homogeneous_subtraction(mass_squared=sp.Integer(4),scale_factor=sp.Integer(2))
    assert sp.simplify(out.energy_subtraction-out.omega/2)==0
    assert sp.simplify(out.variance_subtraction-1/(2*out.omega))==0


def test_tt_trace_and_transversality():
    from ininauto.wick import S,local,wick_vev
    t=sp.Symbol('t')
    assert not wick_vev([local(S('h',indices=('i','i')),t)]).terms
    assert not wick_vev([local(S('h',indices=('i','j'),dx=('i',)),t)]).terms
    assert len(wick_vev([local(S('h',indices=('i','j')),t)]).terms)==1
