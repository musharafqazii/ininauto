"""Symbolic Gaussian Wick contractions and Schwinger--Keldysh building blocks.

Requires: sympy>=1.12.  This is an algebraic engine; it does NOT renormalize
composite operators, evaluate loop/time integrals, or derive Hamiltonians.

Conventions
-----------
phi(t,x) = integral d^3k/(2pi)^3 [u_k(t) a_k e^{ikx} + h.c.]
<phi_A phi_B> uses *ordered*, not time-ordered, Wightman functions.
Each unintegrated source Fourier mode is S(field, q, derivative, time, indices...).
A result term represents the integrand multiplied by:
  product_v integral d^3v/(2pi)^3,
  product_x exp(i * phase[x] . x),
  product_P (2pi)^3 delta^3(P).
All momentum integrals remain formal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable
import sympy as sy

I = sy.I


# ----------------------------------------------------------------------------------- printed building blocks
# The Wick kernels are sympy functions with fixed argument conventions (documented on each class).  str() keeps the
# code form, e.g. U(phi, k1, 1, tau); LaTeX shows the physics notation, e.g. u'_{k_1}(tau).
def _tex(printer, x):
    return printer._print(x)


def _primes(n):
    if getattr(n, "is_Integer", False) or isinstance(n, int):
        n = int(n)
        return "'" * n if n <= 3 else "^{(%d)}" % n
    return "^{(%s)}" % sy.latex(n)


def _mom(printer, k):
    if k.is_Atom:
        return _tex(printer, k)
    if k.could_extract_minus_sign():          # u depends on |k| only: show |k1 + q1| rather than |-k1 - q1|
        k = -k
    return r"\left|" + _tex(printer, k) + r"\right|"


def _species(printer, sp_):
    return "" if str(sp_) == "phi" else "^{(" + _tex(printer, sp_) + ")}"


def _power(body, exp):
    return body if exp is None else r"\left[" + body + r"\right]^{" + str(exp) + "}"


class U(sy.Function):
    """U(species, k, n, t) = d^n u_k(t)/dt^n, the positive-frequency mode function of the species."""

    def _latex(self, printer, exp=None):
        sp_, k, n, t = self.args[:4]
        body = "u" + _species(printer, sp_) + _primes(n) + "_{" + _mom(printer, k) + r"}\!\left(" + _tex(printer, t) + r"\right)"
        return _power(body, exp)


class Ub(sy.Function):
    """Ub(species, k, n, t) = d^n u_k^*(t)/dt^n (complex-conjugate mode function)."""

    def _latex(self, printer, exp=None):
        sp_, k, n, t = self.args[:4]
        body = "u" + _species(printer, sp_) + _primes(n) + "^{*}_{" + _mom(printer, k) + r"}\!\left(" + _tex(printer, t) + r"\right)"
        return _power(body, exp)


class SQ(sy.Function):
    """SQ(source, q, n, t, *indices): Fourier amplitude of the classical source (psi or h_ij) at momentum q,
    n-th time derivative, time t."""

    def _latex(self, printer, exp=None):
        lab, q, n, t = self.args[:4]
        ind = "".join(_tex(printer, i) for i in self.args[4:])
        body = (r"\tilde{" + _tex(printer, lab) + "}" + _primes(n) + ("_{" + ind + "}" if ind else "")
                + r"\!\left(" + _tex(printer, q) + ", " + _tex(printer, t) + r"\right)")
        return _power(body, exp)


class SP(sy.Function):
    """SP(p, q) = p . q (spatial scalar product of two formal momenta)."""

    def _latex(self, printer, exp=None):
        p, q = self.args
        if p == q:
            return _tex(printer, p) + "^{" + str(2 if exp is None else 2 * exp) + "}"
        body = r"\left(" + _tex(printer, p) + r"\cdot " + _tex(printer, q) + r"\right)"
        return body if exp is None else body + "^{" + str(exp) + "}"


class Comp(sy.Function):
    """Comp(p, i) = p_i, the i-th component of the momentum p."""

    def _latex(self, printer, exp=None):
        p, i = self.args
        body = (_tex(printer, p) if p.is_Atom else r"\left(" + _tex(printer, p) + r"\right)") + "_{" + _tex(printer, i) + "}"
        return _power(body, exp)


class Occ(sy.Function):
    """Occ(species, k) = n_k, the occupation number of the Gaussian state."""

    def _latex(self, printer, exp=None):
        sp_, k = self.args
        return _power("n" + _species(printer, sp_) + "_{" + _mom(printer, k) + "}", exp)


class TT(sy.Function):
    """TT(k, i, j, l, m) = Lambda_{ij,lm}(k), the transverse-traceless projector."""

    def _latex(self, printer, exp=None):
        ind = [_tex(printer, i) for i in self.args[1:]]
        body = r"\Lambda_{" + "".join(ind[:2]) + "," + "".join(ind[2:]) + r"}\!\left(" + _tex(printer, self.args[0]) + r"\right)"
        return _power(body, exp)


class Cov(sy.Function):
    """Cov(species, k, *indices, D, n_a, n_b, t_a, t_b): generic (non-TT) tensor covariance, to be supplied by a
    kernel."""

    def _latex(self, printer, exp=None):
        body = (r"\mathcal{C}^{(" + _tex(printer, self.args[0]) + r")}\!\left("
                + ", ".join(_tex(printer, a_) for a_ in self.args[1:]) + r"\right)")
        return _power(body, exp)


def _sym(x):
    return x if isinstance(x, sy.Basic) else sy.Symbol(str(x))


def _idx_tuple(items):
    return tuple(_sym(x) for x in items)


@dataclass(frozen=True)
class FieldAtom:
    """Quantum field (optionally a tensor component), with derivatives."""
    name: str
    dt: int = 0
    dx: tuple = ()
    indices: tuple = ()


@dataclass(frozen=True)
class SourceAtom:
    """Classical c-number spatial Fourier source (optionally tensor-valued)."""
    name: str
    dt: int = 0
    dx: tuple = ()
    indices: tuple = ()


@dataclass(frozen=True)
class Monomial:
    coefficient: sy.Expr
    atoms: tuple[FieldAtom | SourceAtom, ...]


class Poly:
    """Noncommutative polynomial: quantum field order is retained."""
    def __init__(self, monos=()):
        self.monos = tuple(monos)

    @staticmethod
    def of(x):
        if isinstance(x, Poly):
            return x
        if isinstance(x, (FieldAtom, SourceAtom)):
            return Poly((Monomial(sy.S.One, (x,)),))
        return Poly((Monomial(sy.sympify(x), ()),))

    def __add__(self, other):
        return Poly(self.monos + Poly.of(other).monos)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + -Poly.of(other)

    def __rsub__(self, other):
        return Poly.of(other) - self

    def __mul__(self, other):
        right = Poly.of(other)
        return Poly(tuple(Monomial(sy.expand(a.coefficient * b.coefficient), a.atoms + b.atoms)
                          for a in self.monos for b in right.monos))

    def __rmul__(self, other):
        return Poly.of(other) * self

    def __truediv__(self, other):
        return self * (sy.S.One / sy.sympify(other))

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Powers of operator expressions must be nonnegative integers")
        result = Poly.of(1)
        for _ in range(power):
            result *= self
        return result

    def __repr__(self):
        return f"Poly({len(self.monos)} monomials)"


def F(name: str = "chi", *, dt: int = 0, dx=(), indices=()) -> Poly:
    """A quantum field; dx are derivative indices; indices are tensor component indices."""
    return Poly.of(FieldAtom(name, dt, _idx_tuple(dx), _idx_tuple(indices)))


def S(name: str = "h", *, dt: int = 0, dx=(), indices=()) -> Poly:
    """A commuting classical source (possibly h_ij, with indices=('i','j'))."""
    return Poly.of(SourceAtom(name, dt, _idx_tuple(dx), _idx_tuple(indices)))


def anti(a, b):
    return Poly.of(a) * b + Poly.of(b) * a


@dataclass(frozen=True)
class Operator:
    polynomial: Poly
    time: sy.Expr
    point: sy.Symbol
    integrated: bool = False


def local(expr, time, point="x") -> Operator:
    return Operator(Poly.of(expr), sy.sympify(time), _sym(point), False)


_vertex_id = 0


def vertex(expr, time) -> Operator:
    global _vertex_id
    _vertex_id += 1
    return Operator(Poly.of(expr), sy.sympify(time), sy.Symbol(f"_y{_vertex_id}"), True)


def dot(a, b):
    """Bilinear, symmetric spatial scalar product of formal linear momenta."""
    a, b = sy.expand(sy.sympify(a)), sy.expand(sy.sympify(b))
    def coefficients(e):
        base = sorted((s for s in e.free_symbols if _is_momentum(s)), key=str)
        if not base:
            if sy.simplify(e) == 0:
                return []
            raise ValueError(f"Momentum must be linear in k1,q1,...: {e}")
        out = [(v, sy.expand(e).coeff(v)) for v in base]
        rem = sy.expand(e - sum(c*v for v,c in out))
        if rem != 0:
            raise ValueError(f"Unsupported momentum expression: {e} (remainder {rem})")
        return [(v,c) for v,c in out if c != 0]
    aa, bb = coefficients(a), coefficients(b)
    value = sy.S.Zero
    for va,ca in aa:
        for vb,cb in bb:
            x,y = sorted((va,vb), key=str)      # (v0.2 overwrote va here -> wrong composite products)
            value += ca*cb*SP(x,y)
    return sy.expand(value)


def _is_momentum(symbol):
    import re
    return isinstance(symbol, sy.Symbol) and re.fullmatch(r"[kq][1-9][0-9]*",symbol.name) is not None


def _canonical_momentum(v):
    """U(k)=U(-k) when modes are rotationally invariant; not for a general kernel."""
    v = sy.expand(v)
    mom = sorted((s for s in v.free_symbols if _is_momentum(s)), key=str)
    if mom and v.coeff(mom[0]).is_negative:
        return -v
    return v


def _tt_forbidden(summand):
    """Detect trace/divergence of a classical transverse-traceless h_ij.

    In Fourier space, h_ii(q)=0 and q_i*h_ij(q)=0.  These are the
    constraints on the *prescribed source*, independent of the Gaussian
    state of the quantum scalar.  Only obvious proportional momenta are
    simplified; mixed momentum sums are intentionally left symbolic.
    """
    factors=list(sy.Mul.make_args(summand))
    hs=[];components=[]
    for f in factors:
        base,power=f.as_base_exp()
        if base.func==SQ and len(base.args)>=6 and base.args[0]==sy.Symbol('h'):
            hs.extend([base]*int(power) if power.is_Integer and power>0 else [base])
        if base.func==Comp:
            components.append(base.args)
    for h in hs:
        mom=h.args[1];inds=h.args[4:]
        if len(inds)==2 and inds[0]==inds[1]:
            return True
        for p,index in components:
            if index in inds and (sy.expand(p-mom)==0 or sy.expand(p+mom)==0
                                  or p==0):
                return True
    return False


def contract_indices(expr):
    """TT source identities, then pair Comp(p,i) Comp(q,i) -> p.q."""
    expr = sy.expand(expr)
    expanded = []
    for summand in sy.Add.make_args(expr):
        if _tt_forbidden(summand):
            continue
        factors = list(sy.Mul.make_args(summand))
        c = defaultdict(list)
        other = []
        for fac in factors:
            base, power = fac.as_base_exp()
            if base.func == Comp and power.is_Integer and power > 0:
                vec, ix = base.args
                c[ix].extend([vec] * int(power))
            else:
                other.append(fac)
        for index, v in c.items():
            if len(v) == 2:
                other.append(dot(v[0], v[1]))
            else:
                # A single uncontracted derivative index is meaningful; four or
                # more occurrences is ambiguous Einstein notation and not guessed.
                if len(v) > 2:
                    raise ValueError(f"Index {index} repeated {len(v)} times in one term")
                other.extend(Comp(p,index) for p in v)
        expanded.append(sy.Mul(*other))
    return sy.expand(sy.Add(*expanded))


@lru_cache(None)
def _pairings(n):
    if n % 2:
        return ()
    if n == 0:
        return ((),)
    ans=[]
    for j in range(1,n):
        remaining=[i for i in range(1,n) if i!=j]
        for smaller in _pairings(n-2):
            mapped = tuple((remaining[a],remaining[b]) for a,b in smaller)
            ans.append(((0,j),)+mapped)
    return tuple(ans)


def _is_connected(nops, pairing, legs):
    if nops <= 1:
        return True
    graph=[set() for _ in range(nops)]
    for i,j in pairing:
        a,b=legs[i].op,legs[j].op
        graph[a].add(b);graph[b].add(a)
    visited={0};queue=[0]
    while queue:
        a=queue.pop()
        for b in graph[a]-visited:
            visited.add(b);queue.append(b)
    return len(visited)==nops


@dataclass(frozen=True)
class Leg:
    field: FieldAtom
    op: int
    time: sy.Expr
    point: sy.Symbol


Kernel = Callable[[Leg, Leg, sy.Symbol], sy.Expr]


@dataclass
class GaussianState:
    """Gaussian initial state and the elementary ordered two-point kernel.

    occupation=False -> vacuum; True -> diagonal occupation Occ(species,k).
    tt_fields={'gamma'} gives a symbolic transverse-traceless projector for
    tensor fields with two component indices. For custom/species-mixing
    covariance set kernel=callable(left_leg,right_leg,momentum).
    """
    occupation: bool = False
    tt_fields: frozenset[str] = frozenset()
    kernel: Kernel | None = None

    def wightman(self, a: Leg, b: Leg, momentum):
        if self.kernel is not None:
            return self.kernel(a,b,momentum)
        fa,fb=a.field,b.field
        if fa.name!=fb.name or len(fa.indices)!=len(fb.indices):
            return sy.S.Zero
        species=sy.Symbol(fa.name)
        k=_canonical_momentum(momentum)
        if self.occupation:
            n=Occ(species,k)
            scalar=(1+n)*U(species,k,fa.dt,a.time)*Ub(species,k,fb.dt,b.time)\
                +n*Ub(species,k,fa.dt,a.time)*U(species,k,fb.dt,b.time)
        else:
            scalar=U(species,k,fa.dt,a.time)*Ub(species,k,fb.dt,b.time)
        if len(fa.indices)==0:
            return scalar
        if fa.name in self.tt_fields and len(fa.indices)==2:
            return scalar*TT(k,*fa.indices,*fb.indices)
        # Generic tensor covariance is NOT assumed diagonal in the indices.
        # For non-TT tensors, supply a kernel to define their correlators.
        return Cov(species,k,*fa.indices,*fb.indices,sy.Symbol("D"),fa.dt,fb.dt,a.time,b.time)


@dataclass(frozen=True)
class WickTerm:
    integrand: sy.Expr
    loops: tuple
    sources: tuple
    phase: tuple        # pairs (x, momentum_at_x)
    deltas: tuple       # residual (2pi)^3 delta^3(momentum)

    def pretty(self):
        ints=" ".join(f"Int[d^3{v}/(2pi)^3]" for v in self.loops+self.sources)
        phases=" ".join(f"exp(I*({p}).{x})" for x,p in self.phase if p!=0)
        deltas=" ".join(f"(2pi)^3 delta^3({d})" for d in self.deltas)
        return f"{ints} {phases} {deltas} * ({self.integrand})".strip()


@dataclass(frozen=True)
class WickResult:
    terms: tuple[WickTerm,...]

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __mul__(self, c):
        return WickResult(tuple(replace(t,integrand=sy.expand(c*t.integrand)) for t in self.terms)).group()

    __rmul__=__mul__

    def __neg__(self):
        return -1*self

    def __sub__(self, other):
        return self+(-other)

    def __add__(self, other):
        if not isinstance(other,WickResult):
            return NotImplemented
        return WickResult(self.terms+other.terms).group()

    def group(self):
        grouped=defaultdict(lambda:sy.S.Zero)
        for t in self.terms:
            key=(t.loops,t.sources,t.phase,t.deltas)
            grouped[key]+=t.integrand
        result=[]
        for (loops,sources,phase,deltas),inte in grouped.items():
            inte=contract_indices(inte)
            if inte!=0:
                result.append(WickTerm(inte,loops,sources,phase,deltas))
        return WickResult(tuple(result))

    def show(self):
        if not self.terms:
            print("0")
        for j,t in enumerate(self.terms,1):
            print(f"\nTERM {j}:\n{t.pretty()}\n")


def _rename_indices(mono, opnum):
    """Dummy spatial indices at different operator insertions are distinct."""
    allindices=set()
    for atom in mono.atoms:
        allindices.update(atom.dx);allindices.update(atom.indices)
    names={x:sy.Symbol(f"_{x}_op{opnum}") for x in allindices}
    return tuple(replace(atom,
                         dx=tuple(names[x] for x in atom.dx),
                         indices=tuple(names[x] for x in atom.indices)) for atom in mono.atoms)


def _eliminate(coefficient, momentum_at_point, ops, nloops, nsources):
    loops=[sy.Symbol(f"k{j}") for j in range(1,nloops+1)]
    sources=[sy.Symbol(f"q{j}") for j in range(1,nsources+1)]
    rules={};deltas=[]
    allpoints=[]
    for o in ops:
        if o.point not in allpoints:
            allpoints.append(o.point)
    intpoints={o.point for o in ops if o.integrated}
    for pt in allpoints:
        if pt not in intpoints:
            continue
        c=sy.expand(momentum_at_point.get(pt,sy.S.Zero).subs(rules,simultaneous=False))
        pivots=[p for p in reversed(loops) if c.coeff(p)!=0]
        if not pivots:
            pivots=[p for p in reversed(sources) if c.coeff(p)!=0]
        if not pivots:
            if c != 0:
                deltas.append(c)
            continue
        pivot=pivots[0];factor=c.coeff(pivot)
        solution=sy.expand(pivot-c/factor)
        rules={p:sy.expand(q.subs(pivot,solution)) for p,q in rules.items()}
        rules[pivot]=solution
        if pivot in loops:
            loops.remove(pivot)
        else:
            sources.remove(pivot)
    coefficient=coefficient.subs(rules, simultaneous=True)
    mom_phase={pt:sy.expand(expr.subs(rules,simultaneous=True))
               for pt,expr in momentum_at_point.items() if pt not in intpoints}
    # Canonicalize free integration variables to k1,... and q1,...
    mapping={v:sy.Symbol(f"k{j}") for j,v in enumerate(loops,1)}
    mapping.update({v:sy.Symbol(f"q{j}") for j,v in enumerate(sources,1)})
    coefficient=coefficient.subs(mapping,simultaneous=True)
    ph=tuple((pt,sy.expand(mom_phase.get(pt,sy.S.Zero).subs(mapping,simultaneous=True)))
             for pt in allpoints if pt not in intpoints)
    ds=tuple(sy.expand(d.subs(mapping,simultaneous=True)) for d in deltas)
    return WickTerm(contract_indices(coefficient),
                    tuple(sy.Symbol(f"k{j}") for j in range(1,len(loops)+1)),
                    tuple(sy.Symbol(f"q{j}") for j in range(1,len(sources)+1)), ph, ds)


def wick_vev(operators: Iterable[Operator], *, connected=False,
             state: GaussianState | None = None, allow_internal=True) -> WickResult:
    """Ordered Gaussian vacuum/state expectation of an operator product.

    connected: require the Wick graph to connect all operator insertions.
    allow_internal=False: discard same-insertion contractions (normal ordering).
    The operator product order, NOT the chronological times, determines G^>.
    """
    ops=tuple(operators)
    if not ops:
        return WickResult((WickTerm(sy.S.One,(),(),(),()),))
    state=state or GaussianState()
    output=[]
    for ms in product(*(o.polynomial.monos for o in ops)):
        coefficient=sy.prod(m.coefficient for m in ms)
        legs=[];source_atoms=[]
        for j,(m,o) in enumerate(zip(ms,ops)):
            for atom in _rename_indices(m,j):
                if isinstance(atom,FieldAtom):
                    legs.append(Leg(atom,j,o.time,o.point))
                else:
                    source_atoms.append((atom,o))
        if len(legs)%2:
            continue
        source_mom=defaultdict(lambda:sy.S.Zero)
        for s,(atom,op) in enumerate(source_atoms,1):
            q=sy.Symbol(f"q{s}");label=sy.Symbol(atom.name)
            coefficient *= SQ(label,q,atom.dt,op.time,*atom.indices)
            for ix in atom.dx:
                coefficient *= I*Comp(q,ix)
            source_mom[op.point]+=q
        for matching in _pairings(len(legs)):
            if connected and not _is_connected(len(ops),matching,legs):
                continue
            if not allow_internal and any(legs[i].op==legs[j].op for i,j in matching):
                continue
            amplitude=coefficient
            mom=source_mom.copy()
            for j,(ia,ib) in enumerate(matching,1):
                a,b=legs[ia],legs[ib]
                k=sy.Symbol(f"k{j}")
                propagator=state.wightman(a,b,k)
                amplitude *= propagator
                for ix in a.field.dx:
                    amplitude *= I*Comp(k,ix)
                for ix in b.field.dx:
                    amplitude *= -I*Comp(k,ix)
                mom[a.point]+=k
                mom[b.point]-=k
            if amplitude==0:
                continue
            output.append(_eliminate(amplitude,mom,ops,len(matching),len(source_atoms)))
    return WickResult(tuple(output)).group()


def nested_commutator_vev(operators: Iterable[Operator], **kwargs) -> WickResult:
    """<[[...[A,B],C]...]>, with nesting to the right as in the in-in series."""
    operators=tuple(operators)
    if not operators:
        raise ValueError("At least one operator required")
    terms=[(1,(operators[0],))]
    for next_op in operators[1:]:
        terms=[(s, seq+(next_op,)) for s,seq in terms]+\
              [(-s, (next_op,)+seq) for s,seq in terms]
    result=WickResult(())
    for sign,order in terms:
        result=result+sign*wick_vev(order,**kwargs)
    return result.group()


def inin_second_order(rho0, rho1, rho2, h1_at_t1, h2_at_t1, h1_at_t2,
                      *, state=None, connected=True):
    """Return pieces (WITHOUT time integration) of <rho> through h^2.

    V0=<rho0>, V1=<rho1>, C01=-i<[rho0,H1(t1)]>,
    V2=<rho2>, C11=-i<[rho1,H1(t1)]>, C02=-i<[rho0,H2(t1)]>,
    C011=-<[[rho0,H1(t1)],H1(t2)]>.
    The C011 integral has t_initial <= t2 <= t1 <= t_observation.
    Pass rho1/rho2=None to omit undefined operator corrections.
    """
    opts=dict(state=state,connected=connected)
    out={"rho0":wick_vev([rho0],state=state),
         "rho0_H1":-I*nested_commutator_vev([rho0,h1_at_t1],**opts),
         "rho0_H2":-I*nested_commutator_vev([rho0,h2_at_t1],**opts),
         "rho0_H1_H1":-nested_commutator_vev([rho0,h1_at_t1,h1_at_t2],**opts)}
    if rho1 is not None:
        out["rho1"]=wick_vev([rho1],state=state)
        out["rho1_H1"]=-I*nested_commutator_vev([rho1,h1_at_t1],**opts)
    if rho2 is not None:
        out["rho2"]=wick_vev([rho2],state=state)
    return out


def latex_result(res: WickResult) -> str:
    """LaTeX with explicit formal integration measures, phases, and residual deltas."""
    chunks = []
    for term in res.terms:
        measure = r"\,".join(r"\int\!\frac{d^3%s}{(2\pi)^3}" % sy.latex(v) for v in term.loops + term.sources)
        phases = []
        for x, p in term.phase:
            if p == 0:
                continue
            pt = sy.latex(p)
            if not p.is_Atom:
                pt = r"\left(" + pt + r"\right)"
            phases.append(r"e^{i\,%s\cdot %s}" % (pt, sy.latex(x)))
        delta = r"\,".join(r"(2\pi)^3\delta^{(3)}\!\left(%s\right)" % sy.latex(p) for p in term.deltas)
        body = sy.latex(term.integrand)
        if isinstance(term.integrand, sy.Add):
            body = r"\left[" + body + r"\right]"
        pieces = [t for t in (measure, r"\,".join(phases), delta, body) if t]
        chunks.append(r"\,".join(pieces))
    return " + ".join(chunks) if chunks else "0"


__all__=["F","S","anti","local","vertex","dot","contract_indices","GaussianState",
         "wick_vev","nested_commutator_vev","inin_second_order","latex_result",
         "WickTerm","WickResult","U","Ub","SQ","SP","Comp","Occ","TT","Cov"]
