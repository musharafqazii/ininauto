"""Unrestricted, restartable evaluation of every Schwinger-Keldysh Wick term.

Storage is transactional: each exact diagram is committed to SQLite before the
next job is attempted.  A run can be interrupted and resumed without deleting
completed work.  No Wick leg or vertex budget is imposed.  Only Gaussian Wick
expectations are evaluated: UV-renormalized distributions are a distinct task.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sympy as sp

from .action import ScalarAction
from ..core.geometry import scalar_geometry, tensor_geometry
from .expansion import reduced_lagrangian
from .hamiltonian import legendre
from .stress import hilbert_density
from .free_modes import scalar_modes
from .inin import enumerate_diagrams, default_labels
from .operators import to_local, to_vertex
from ..core.symbols import T
from ..core.wick import GaussianState, nested_commutator_vev, wick_vev
from .renormalization import prescription
from .counterterm_operators import local_counterterms
from .report import write_reports

SCHEMA = '''CREATE TABLE IF NOT EXISTS diagrams (
 id TEXT PRIMARY KEY, family TEXT NOT NULL, coupling_order INTEGER NOT NULL,
 metric_order INTEGER NOT NULL, observable TEXT NOT NULL, vertices TEXT NOT NULL,
 insertion_count INTEGER NOT NULL, status TEXT NOT NULL, terms INTEGER,
 elapsed_seconds REAL, payload BLOB, error TEXT,
 updated_at TEXT DEFAULT (datetime('now'))
)'''

@dataclass(frozen=True)
class Job:
    identifier: str
    family: str
    degree: tuple[int, int]
    observable: str
    vertices: tuple[str, ...]
    observable_expression: sp.Expr
    vertex_expressions: tuple[sp.Expr, ...]


def _read_jobs(action: ScalarAction, families: Sequence[str], include_counterterms=True):
    catalog = []
    data = {}
    obs_names, vert_names = default_labels()
    for fam in families:
        if fam not in ('scalar','tensor'):
            raise ValueError(f'Unknown metric family: {fam}')
        geom = scalar_geometry() if fam == 'scalar' else tensor_geometry()
        lag = reduced_lagrangian(action,fam)
        h = legendre(lag,action.coupling)
        rho = hilbert_density(fam,h,action)
        interactions = {k:v for k,v in h.interaction_coeffs.items() if k != (0,0)}
        observables = dict(rho.coefficients)
        if include_counterterms:
            cterms=local_counterterms(action,fam)
            for degree,expr in cterms.interaction.items():
                interactions[degree] = interactions.get(degree,sp.S.Zero)+expr
            for degree,expr in cterms.observable.items():
                observables[degree] = observables.get(degree,sp.S.Zero)+expr
        enumerated = enumerate_diagrams(observables,interactions)
        ren = prescription(action,fam)
        data[fam] = (h,rho,geom,lag,enumerated,ren)
        for degree, definitions in enumerated.items():
            for i, (obs_deg, verts) in enumerate(definitions):
                oid = obs_names[obs_deg]
                vnames = tuple(vert_names[v] for v in verts)
                catalog.append(Job(f'{fam}:{degree[0]}:{degree[1]}:{i:03d}',
                                   fam,degree,oid,vnames,observables[obs_deg],
                                   tuple(interactions[v] for v in verts)))
    return catalog,data


def _serialize_wick(wick):
    """Self-contained exact integral representation; no truncated expression."""
    return [{'integrand':str(term.integrand),
             'loop_momenta':[str(x) for x in term.loops],
             'source_momenta':[str(x) for x in term.sources],
             'phase':[(str(x),str(v)) for x,v in term.phase],
             'momentum_constraints':[str(x) for x in term.deltas],
             'integral':term.pretty()}
            for term in wick.terms]


def _worker(job: Job, connected: bool, state=None):
    start=time.perf_counter()
    state = state if state is not None else GaussianState()
    ts=tuple(sp.Symbol(f't{i}',real=True) for i in range(1,len(job.vertices)+1))
    # Operator points are freshly generated for each job, so independent vertex
    # spatial integrals remain independent before momentum conservation.
    o=to_local(job.observable_expression,T,'x')
    ins=[to_vertex(e.subs(T,t),t) for e,t in zip(job.vertex_expressions,ts)]
    if ins:
        val= (-sp.I)**len(ins)*nested_commutator_vev([o,*ins],connected=connected,
                                                       state=state)
    else:
        val=wick_vev([o],state=state)
    payload={'id':job.identifier,'family':job.family,'degree':job.degree,
             'observable':job.observable,'vertices':job.vertices,
             'integration_times':list(map(str,ts)),
             'prefactor':str((-sp.I)**len(ins)),
             'connected_only':bool(connected),
             'wick_terms':_serialize_wick(val),
             'notes':('The expression is a bare Gaussian Wightman integrand; '
                      'the nested ordered simplex time integrals are not evaluated. '
                      'Full free gravitational counterterms and finite CTP diagonal contact extensions still require renormalization.')}
    return payload,time.perf_counter()-start


def _open_database(path:Path):
    connection=sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA busy_timeout=30000')
    connection.execute(SCHEMA)
    connection.commit()
    return connection


def _fingerprint(action,families,connected,include_counterterms,state):
    from hashlib import sha256
    txt=sp.srepr(action.expression)+repr(tuple(families))+str(bool(connected))+str(bool(include_counterterms))+repr(state)
    return sha256(txt.encode()).hexdigest()


def _run_one(job,connected,state):
    return _worker(job,connected,state)


def solve(action, output_dir='full_results', *, families=('scalar','tensor'),
          workers=1, resume=True, connected=True, stop_after=None,
          progress=True, retry_failed=True, include_counterterms=True,
          state=None, strict=True):
    """Calculate *all* Gaussian Wick contractions through lambda*h^2.

    ``action`` is the only REQUIRED argument. Every formal Wick term has a job,
    no leg/insertion cutoff exists. The chosen CTP vacuum is Gaussian with
    unspecified free mode functions. UV finite numerical results are impossible
    to infer from the action alone and are not claimed here.

    jobs are stored in output_dir/diagrams.sqlite (durable, restartable).
    ``stop_after`` is solely for controlled tests and partial runs; otherwise
    all diagrams are processed. For heavy runs use ``workers=os.cpu_count()``.
    """
    action=action if isinstance(action,ScalarAction) else ScalarAction.parse(action)
    if action.coupling != 0 and not action.coupling.is_Symbol:
        raise ValueError('Quartic coupling must be a symbol or zero')
    families=tuple(families)
    state=state if state is not None else GaussianState()
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    signature=_fingerprint(action,families,connected,include_counterterms,state)
    state_file=root/'run.json'
    if state_file.exists() and resume:
        old=json.loads(state_file.read_text())
        if old['fingerprint']!=signature:
            raise ValueError('Existing run has different input; choose a new output directory')
    if not resume and (root/'diagrams.sqlite').exists():
        (root/'diagrams.sqlite').unlink()
    state_file.write_text(json.dumps({'fingerprint':signature,'action':str(action.expression),
        'families':families,'connected_only':connected,
        'mass_vacuum_counterterms':include_counterterms,
        'Gaussian_state':repr(state),
        'description':'All Wick diagrams, exact symbolic Gaussian state; bare distributions plus optional local mass/vacuum counterterm insertions; full renormalization incomplete'},indent=2))

    jobs,summary=_read_jobs(action,families,include_counterterms=include_counterterms)
    db=_open_database(root/'diagrams.sqlite')
    already={row[0]:row[1] for row in db.execute('SELECT id,status FROM diagrams')}
    pending=[j for j in jobs if j.identifier not in already
             or (already[j.identifier]=='failed' and retry_failed)]
    if stop_after is not None:
        pending=pending[:stop_after]

    # The compact overview is written immediately, even for calculations that
    # take days. The exact expanded integrands appear only in SQLite and JSONL.
    from .inin import evaluate_diagrams
    skim={}
    for fam,(h,r,g,l,definitions,ct) in summary.items():
        v={k:x for k,x in h.interaction_coeffs.items() if k!=(0,0)}
        ds=evaluate_diagrams(definitions,r.coefficients,v,max_wick_legs=0,
                             max_wick_insertions=0)
        skim[fam]=(h,r,g,l,ds,ct)
    overview=write_reports(root,action,skim,scalar_modes(action))
    notes = ("\n\nEXACT EXECUTION: Wick contractions are evaluated without a "
             "leg limit, saved diagram-by-diagram to diagrams.sqlite. "
             "When mass_vacuum_counterterms is true, each lambda-order "
             "interaction vertex and local observable also includes the "
             "analytically known mass and vacuum counterterm coefficients. "
             "The auxiliary overview reports the *bare* Hamiltonian and density; "
             "the regulated combinations are recorded in expanded_wick.jsonl. "
             "These do NOT constitute finite renormalized CTP expectation values "
             "without the metric-dependent Hadamard subtraction and contact terms.\n")
    with (root/'derivation.txt').open('a') as f:
        f.write(notes)

    if progress:
        print(f'[{len(jobs)} diagrams; {len(pending)} waiting; workers={workers}]',flush=True)

    def _commit(j,result=None,elapsed=None,error=None):
        if error is not None:
            status='failed';terms=None;blob=None
        else:
            status='complete';terms=len(result['wick_terms'])
            blob=zlib.compress(json.dumps(result,ensure_ascii=False).encode(),level=6)
        db.execute('''INSERT OR REPLACE INTO diagrams
          (id,family,coupling_order,metric_order,observable,vertices,insertion_count,
           status,terms,elapsed_seconds,payload,error,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))''',
          (j.identifier,j.family,*j.degree,j.observable,json.dumps(j.vertices),
           len(j.vertices),status,terms,elapsed,blob,str(error) if error else None))
        db.commit()
        if progress:
            print(f'[{j.identifier}] {status} terms={terms} '
                  f'elapsed={elapsed:.2f}s' if elapsed is not None else f'{j.identifier} failed',flush=True)

    try:
        if workers == 1:
            for j in pending:
                started=time.perf_counter()
                try:
                    result,seconds=_run_one(j,connected,state)
                    _commit(j,result,seconds)
                except Exception as exc:
                    _commit(j,elapsed=time.perf_counter()-started,error=repr(exc))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures={executor.submit(_run_one,j,connected,state):j for j in pending}
                for fut in as_completed(futures):
                    j=futures[fut]
                    try:
                        result,seconds=fut.result()
                        _commit(j,result,seconds)
                    except Exception as exc:
                        _commit(j,error=repr(exc))
    finally:
        db.close()
    counts=export_completed(root)
    if strict and counts['failed']:
        raise RuntimeError(f"{counts['failed']} diagram jobs failed. "
                           f"Inspect {root/'progress.json'} and rerun to retry them.")
    return FullResult(root,len(jobs),len(pending))


@dataclass(frozen=True)
class FullResult:
    path: Path
    total_jobs: int
    requested_jobs: int

    @property
    def status(self):
        with sqlite3.connect(self.path/'diagrams.sqlite') as db:
            return dict(db.execute('SELECT status,COUNT(*) FROM diagrams GROUP BY status').fetchall())


def export_completed(directory,*,include_integrands=True):
    """Export all available results as newline-delimited JSON, preserving exact terms."""
    root=Path(directory)
    db=_open_database(root/'diagrams.sqlite')
    counts={'complete':0,'failed':0}
    with (root/'expanded_wick.jsonl').open('w') as handle:
        for jid,st,payload in db.execute('SELECT id,status,payload FROM diagrams ORDER BY id'):
            if st=='complete':
                body=json.loads(zlib.decompress(payload))
                if not include_integrands:
                    body.pop('wick_terms',None)
                handle.write(json.dumps(body,ensure_ascii=False)+'\n')
                counts['complete']+=1
            elif st=='failed':
                counts['failed']+=1
    failures=[{'id':x,'error':y} for x,y in db.execute(
        'SELECT id,error FROM diagrams WHERE status="failed" ORDER BY id')]
    db.close()
    (root/'progress.json').write_text(json.dumps({'counts':counts,'failed':failures},indent=2))
    return counts
