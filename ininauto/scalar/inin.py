"""Automatic Dyson-degree bookkeeping and ordered nested commutators.

If O has degree (m,n) and each H contributes (mi,ni), enumerate only
nonnegative solutions to sum degrees=(target lambda power, eps power).
The order of vertices denotes LATEST to EARLIEST time; no extra 1/n!
occurs, because the integration domain is the ordered simplex.
"""
from dataclasses import dataclass
from itertools import product
import sympy as s
from ..core.wick import wick_vev,nested_commutator_vev,GaussianState
from .operators import to_local,to_vertex
from ..core.symbols import T

@dataclass(frozen=True)
class InInDiagram:
    degree: tuple[int,int]
    observable: str
    vertices: tuple[str,...]  # latest first
    times: tuple[s.Symbol,...]
    coefficient: s.Expr
    wick: object|None
    status: str

    @property
    def compact(self):
        if not self.vertices:
            return f'<{self.observable}>'
        nested=self.observable
        for i,name in enumerate(self.vertices,1):
            nested=f'[{nested},{name}(t{i})]'
        return f'(-i)^{len(self.vertices)} integrated <{nested}>'

    @property
    def latex_integral(self):
        """Full Dyson simplex including signs, with latest interaction innermost."""
        ob=r'\mathcal O_{\mathrm{'+self.observable+r'}}(\tau)'
        if not self.vertices:
            return r'\langle '+ob+r'\rangle_0'
        val=ob
        for i,name in enumerate(self.vertices,1):
            val=r'\left['+val+r',H_{\mathrm{'+name+r'}}(t_{'+str(i)+r'})\right]'
        pref=r'(-\mathrm i)^{'+str(len(self.vertices))+r'}'
        inners=[]
        for i in range(1,len(self.vertices)+1):
            upper=r'\tau' if i==1 else r't_{'+str(i-1)+r'}'
            inners.append(r'\int_{\tau_i}^{'+upper+r'}\mathrm d t_{'+str(i)+r'}')
        return pref+' '.join(inners)+r'\left\langle '+val+r'\right\rangle_0'


def default_labels():
    return {(0,0):'R0',(0,1):'R1',(0,2):'R2',(1,0):'V'}, {
        (1,0):'A',(0,1):'B',(0,2):'D',(1,1):'C',(1,2):'E'}


def enumerate_diagrams(observables,vertices,targets=((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)),max_insertions=3):
    obs_names,vert_names=default_labels()
    diagrams={}
    for target in targets:
        entries=[]
        for n in range(max_insertions+1):
            for od in observables:
                for seq in product(vertices,repeat=n):
                    degree=(od[0]+sum(x[0] for x in seq), od[1]+sum(x[1] for x in seq))
                    if degree!=target: continue
                    entries.append((od,seq))
        diagrams[target]=entries
    return diagrams


def _count_fields(expr):
    from .operators import to_polynomial
    poly=to_polynomial(expr)
    if not poly.monos: return 0
    from ..core.wick import FieldAtom
    return max(sum(isinstance(a,FieldAtom) for a in m.atoms) for m in poly.monos)


def evaluate_diagrams(diagram_definitions,obs_exprs,vertex_exprs,
                      max_wick_legs=4,max_wick_insertions=1,state=None):
    """Evaluate manageable Gaussian Wick integrands, retaining all heavier graphs formally.

    Use max_wick_legs=None and max_wick_insertions=None for full Wick expansion,
    which is combinatorially expensive (8 fields -> 105 pairings per monomial).
    """
    obs_names,vert_names=default_labels()
    state=state or GaussianState()
    result={}
    for degree,definitions in diagram_definitions.items():
        jobs=[]
        for obs_degree,ins in definitions:
            times=tuple(s.Symbol(f't{i}',real=True) for i in range(1,len(ins)+1))
            label=obs_names[obs_degree]
            labels=tuple(vert_names[v] for v in ins)
            nc=_count_fields(obs_exprs[obs_degree])+sum(_count_fields(vertex_exprs[v]) for v in ins)
            within=(max_wick_legs is None or nc<=max_wick_legs) and (max_wick_insertions is None or len(ins)<=max_wick_insertions)
            wr=None
            if within:
                o=to_local(obs_exprs[obs_degree],T,'x')
                h=[to_vertex(vertex_exprs[v].subs(T,tm),tm) for v,tm in zip(ins,times)]   # time-dependent c-numbers at the vertex time
                if ins:
                    wr=(-s.I)**len(ins)*nested_commutator_vev([o]+h,connected=True,state=state)
                else:
                    wr=wick_vev([o],state=state)
            jobs.append(InInDiagram(degree,label,labels,times,(-s.I)**len(ins),wr,
                                    'wick-expanded' if within else f'formal (budget: {nc} field legs, {len(ins)} insertions)'))
        result[degree]=jobs
    return result
