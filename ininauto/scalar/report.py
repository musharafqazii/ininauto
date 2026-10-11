"""Export source-exact analytical stage outputs and audited limitations."""
import json
from pathlib import Path
import sympy as s

DEGREE_NAMES={(0,0):'O(1)',(0,1):'O(h)',(0,2):'O(h^2)',
              (1,0):'O(lambda)',(1,1):'O(lambda h)',(1,2):'O(lambda h^2)'}

def write_reports(output_dir,action,results,mode):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    lines=[r'\documentclass[11pt]{article}',r'\usepackage[margin=22mm]{geometry}',
           r'\usepackage{amsmath,amssymb,mathtools,longtable,mathrsfs}',
           r'\allowdisplaybreaks',r'\begin{document}',
           r'\section*{Symbolic in--in derivation from the covariant action}',
           r'\paragraph{Input action} $'+s.latex(action.expression)+r'$.',
           r'\paragraph{Conventions} $(-,+,+,+)$, $\xi=1/6$, $\Phi=\phi/a$, $M^2=a^2m^2$, '
           r'$\epsilon$ counts metric perturbations. Spatial fields use Wick-symmetric ordering.',
           r'\paragraph{Scope} This report displays bare formal Wick expressions and the covariant '
           r'counterterm/renormalized response identities. Finite Hadamard/CTP kernels are NOT '
           r'numerically evaluated; do not interpret unregulated loop integrals as physical densities.']
    manifest={'input_action':str(action.expression),'modes':{k:str(v) for k,v in mode.items()},'families':{}}
    for family,data in results.items():
        H,R,g,L,diags,ct=data
        f={'lagrangian':str(L),'momentum':str(H.momentum),
           'velocity':str(H.velocity),'Hamiltonian':str(H.full_density),
           'energy_density':str(R.canonical),'Hamiltonian_coeffs':{},'density_coeffs':{},
           'inin':{},'renormalization':{str(k):str(v) for k,v in ct.finite_coefficients.items()}}
        manifest['families'][family]=f
        lines.extend([r'\clearpage',r'\section*{'+family.capitalize()+' metric family}',
           r'\subsection*{Metric geometry}',
           r'\begin{align*}',r'\sqrt{-\gamma}&='+s.latex(g.jacobian)+r'\\',
           r'R[\gamma]&='+s.latex(g.ricci),r'\end{align*}',
           r'\subsection*{Action reduction and canonical transform}',
           r'\begin{align*}',r'\mathcal L&='+s.latex(L)+r'\\',
           r'\pi&='+s.latex(H.momentum)+r'\\',
           r'\dot\phi(\pi)&='+s.latex(H.velocity)+r'\\',
           r'\mathcal H&='+s.latex(H.full_density),r'\end{align*}',
           r'\subsection*{Hamiltonian interaction coefficients}'])
        for degree,expr in sorted(H.interaction_coeffs.items()):
            f['Hamiltonian_coeffs'][str(degree)]=str(expr)
            lines.append(r'\[\mathscr H_{'+str(degree[0])+str(degree[1])+'}='+s.latex(expr)+r'\]')
        lines.append(r'\subsection*{Canonical normal-observer density coefficients}')
        for degree,expr in sorted(R.coefficients.items()):
            f['density_coeffs'][str(degree)]=str(expr)
            lines.append(r'\[\rho_{'+str(degree[0])+str(degree[1])+'}='+s.latex(expr)+r'\]')
        lines.append(r'\subsection*{In--in responses by perturbative degree}')
        for degree,entries in diags.items():
            lines.append(r'\paragraph{'+DEGREE_NAMES.get(degree,str(degree))+': '+str(len(entries))+' insertion structures}')
            f['inin'][str(degree)]=[]
            for entry in entries:
                li={'observable':entry.observable,'vertices':list(entry.vertices),
                    'formal_integral_latex':entry.latex_integral,
                    'status':entry.status,'time_variables':[str(t) for t in entry.times]}
                if entry.wick is not None:
                    li['wick_terms']=[{'integrand':str(t.integrand),
                        'loops':list(map(str,t.loops)),'source_momenta':list(map(str,t.sources)),
                        'phase':[(str(x),str(p)) for x,p in t.phase],
                        'delta_constraints':list(map(str,t.deltas))} for t in entry.wick.terms]
                f['inin'][str(degree)].append(li)
                # Long integrands go in JSON, avoiding multi-megabyte TeX output.
                vs=','.join(entry.vertices) or 'none'
                lines.append(r'\[ '+entry.latex_integral+r' \]')
                lines.append(r'\noindent \textit{'+entry.status.replace('_',r'\_')+r'}.\\')
        lines.append(r'\subsection*{Renormalized symbolic coefficient rules}')
        for degree,expr in ct.finite_coefficients.items():
            lines.append(r'\[\rho^{R}_{'+str(degree[0])+str(degree[1])+'}='+s.latex(expr)+r'\]')
        lines.append(r'\paragraph{Regulator prescription} $\Gamma_{0,\zeta}^{\mathrm{pole}}/'+
                     r'\sqrt{g_E}='+s.latex(ct.free_pole_density)+r'$.')
        lines.append(r'\paragraph{Limitations} Finite $U_j,d_j,M_j$ depend on the actual state and '
                     r'the renormalized closed-time-path functional. These are not evaluated automatically.')
    lines.append(r'\end{document}')
    (output/'derivation.tex').write_text('\n'.join(lines))
    (output/'results.json').write_text(json.dumps(manifest,indent=2))
    summary=['INPUT ACTION:',str(action.expression),'',
             'Mode equation:',str(mode['mode_equation']),'',
             'Gaussian integrals are *bare* until full-metric renormalization.']
    for fam,data in results.items():
        H,R,g,L,ds,ct=data
        summary.extend(['',f'===== {fam.upper()} =====',
                        'Canonical Lagrangian:',str(L),
                        'Canonical momentum:',str(H.momentum),
                        'Hamiltonian:'])
        summary.extend(f'  H{d}: {v}' for d,v in sorted(H.interaction_coeffs.items()))
        summary.append('Normal observer energy density:')
        summary.extend(f'  rho{d}: {v}' for d,v in sorted(R.coefficients.items()))
        summary.append('In-in diagrams:')
        for deg,items in ds.items():
            summary.append(f'  {DEGREE_NAMES[deg]}: {len(items)} diagrams')
            summary.extend('    '+x.compact+' ['+x.status+']' for x in items)
        summary.append('Finite coefficients (formal, NOT evaluated):')
        summary.extend(f'  rho{d}: {v}' for d,v in ct.finite_coefficients.items())
    (output/'derivation.txt').write_text('\n'.join(summary))
    return output
