# InInAuto 0.3.1: from the action to renormalised numbers

InInAuto takes the covariant action of a conformally coupled real scalar with a $\lambda\Phi^4/24$ term and returns the
**renormalised in-in energy density** $\langle\rho\rangle$ through $O(\lambda h^2)$, as numbers, for scalar and tensor
metric perturbations. It also still produces the exact symbolic Wick integrands of version 0.2.

## What changed in 0.3.1

Symbolic output now renders in physics notation. `sympy.latex(...)`, Jupyter display and `sympy.pprint` show
$(\nabla\phi)^2$, $\phi'$, $\psi'$, $\nabla\psi\cdot\nabla\phi$, $\epsilon$, $M^2(\tau)$, $h_{ij}h_{ij}$, ... instead of
`grad_phi_squared`, `v`, `psi_t`, `eps`, `M_2`. The action blocks also render properly: `KIN`, `PHI2`, `RPHI2` and
`PHI4` show as $(\nabla\Phi)^2$, $\Phi^2$, $R\Phi^2$ and $\Phi^4$. The code names, `str()` output and stored results
are unchanged.

## What changed from 0.2

**Bug fixes**, each with a regression test in `tests/test_fixes.py`:

1. `wick.dot` overwrote its own loop variable. Every scalar product of composite momenta was wrong; for example $(q-k)^2$
   came out as $k^2-k\cdot q$. This affected every gradient term carried by a shifted leg.
2. $M^2=a^2m^2$ was a constant symbol, so vertex mass terms were evaluated at the observation time. It is now
   $M^2(\tau)$ and is relabelled to each vertex time. The pipeline's preview path now relabels all time-dependent
   c-numbers as well.
3. A new test checks $\langle\rho_0(\tau,\mathbf x)H_1(t_1)\rangle$ numerically against the reference derivation.

**End-to-end solver**: `evaluate(action, values)`, described next.

## Installation

```bash
pip install git+https://github.com/<github-username>/ininauto.git          # from GitHub
pip install -e .                                                            # from a local clone (editable)
```

The distribution is called `ininauto-qft`; import it as `import ininauto` (Python >= 3.10, needs sympy and numpy;
matplotlib is optional and only used for the overview plot: `pip install "ininauto-qft[plots] @ git+https://...`).

To publish this folder to your GitHub account: `bash publish_to_github.sh <github-username> [repo-name] [--private]`.

## End-to-end use

```python
import sympy as sp
from ininauto import KIN, PHI2, RPHI2, PHI4, evaluate
m, lam = sp.symbols('m lam', real=True)
res = evaluate(-KIN/2 - m**2*PHI2/2 - RPHI2/12 - lam*PHI4/24, values={'m': 1.0, 'lam': 0.1})
print(res.summary())
```

```bash
python3 -m ininauto --action='-KIN/2-m**2*PHI2/2-RPHI2/12-lam*PHI4/24' --value m=1 --value lam=0.1 --output results
python3 -m ininauto ... --set pmax=8 --set np=60 --set mu_R=2 --q 0.1 0.2 0.3 --family scalar
```

`values` gives numbers for the action's symbols, with $m$ in units of $H_e$. If $\lambda$ has no value, the $O(\lambda)$
results are returned per unit $\lambda$. Runs use half of the cores by default (`workers=` overrides this). Finished
runs are cached under `<output>/runs/`, so repeating a call only recomputes what changed.

### What happens

1. **Symbolic derivation from the action**: geometry, reduced Lagrangian, Legendre transform, canonical Hamiltonian and
   normal-observer (Hilbert) energy density, for both metric families. This is unchanged from 0.2.
2. **Verification against the action** (`ininauto.bridge`). Every kernel used by the numerical back end is re-derived and
   compared exactly. The numerics do not run if anything differs.
   - The mode operators $D_0$, $\mathcal P_1$ and $\mathcal P_2$ come from the Euler–Lagrange equation of the reduced
     Lagrangian (`sympy.euler_equations` on explicit fields), projected as the engine uses them: the first-order side band,
     and the second-order zero mode of $\psi^2$ or $h^2$ (isotropic for tensors).
   - The energy density in velocity variables, at orders $\epsilon^{0,1,2}$.
   - The quartic vertex and its determinant factors $j_1$, $j_2$, which are the Hartree tadpoles $\Sigma_{1,2}$.
   - $\xi=1/6$ and $m^2$.
3. **Numerics** (`ininauto.numerics`). For each Fourier mode $q$ of the metric perturbation, the engine integrates the
   retarded hierarchy $F_k=u+\epsilon v+\epsilon^2 z+\lambda(\ell+\epsilon w+\epsilon^2 r)$ in the in-vacuum. This is
   equivalent to the canonical in-in series of nested commutators. It then evaluates
   $\langle\rho\rangle=\sum_k\mathcal E(F_k,F_k)$ and renormalises:
   - 4th-order adiabatic subtraction of the FRW vacuum, plus the full 4th-order local subtraction of every metric kernel.
     This removes the non-covariant $q^2p_{\rm cut}^2$ tail of a comoving mode sum and the curvature-squared logarithm.
     It equals the covariant proper-time result at $\mu_R=m$. For another scale (`mu_R=`), the curvature-squared term
     $-\mathcal B\ln(\mu_R/m)/(1440\pi^2)$ is added, with $\mathcal B=n^\mu n^\nu(3\mathscr H^{(2)}-\mathscr H^{(1)})_{\mu\nu}$.
   - $O(\lambda)$: renormalised Hartree tadpoles $\Sigma_{0,1,2}$, the exchange subtraction, and contacts
     $\tfrac{\lambda}{8}\Delta^2$, with the quartic coupling switched on smoothly before the first record.
4. **Assembly and output.**

### Outputs

These are written to `output_dir`; all quantities are comoving $a^4\langle\rho\rangle$ in units of $H_e^4$.

| quantity | meaning |
|---|---|
| `rho_00`, `rho_10_per_lambda` | homogeneous energy density and its $O(\lambda)$ correction, as time series in $N=\ln a$ |
| `rho_01`, `rho_11_per_lambda` | coefficient of $c\,e^{iqx}$ for $\psi=T(\eta)(c\,e^{iqx}+{\rm c.c.})$ (scalar); quadrupole moment (tensor) |
| `rho_02`, `rho_12_per_lambda` | ensemble mean per unit power, $\langle\psi^2\rangle=T^2$ or $\langle h_{ij}h_{ij}\rangle=T^2$ |
| `*_int` | integrated over $\ln q$ on the $q$ grid |
| `rho_02_hybrid`, `particles_02` | alternative organisation (particle part per source momentum) and the particle-number part |
| `rho_total_homogeneous`, `rho_2_total_int` | $\rho_{00}+\lambda\rho_{10}$ and $\rho_{02}+\lambda\rho_{12}$ (summed over $\ln q$), when a value of $\lambda$ is given |

The files are `rho_results.json`, `summary.txt` and `rho_overview.png`, plus the raw engine output (`runs/*.npz`) and the
verification report (`bridge` in the JSON).

### Physical input that the action does not fix

These are fixed here and recorded in the output:

- **Background**: smooth inflation to radiation, $H(N)=H_e\,2^{1/4}(1+e^{8N})^{-1/4}$.
- **Perturbation profiles $T(N)$**:
  - scalar: the Newtonian potential, constant outside the horizon during inflation and oscillating inside during radiation;
  - tensor: the free tensor mode, with a $C^2$ switch-on at horizon entry.
- **State**: the adiabatic (4th-order) in-vacuum of each mode.
- **Renormalisation**: the scale $\mu_R$ (default $m$).

To change the background or profiles, edit `numerics/engine.py` (`bg`, `profile`).

### Known limitations of the numerics

- **Not subtracted**:
  - the mixed local terms $q^2\times$(time derivatives of $M^2$), which are finite and suppressed by $\mathcal H^2/M^2$;
  - the local terms of the $\lambda$ sectors, which are UV-convergent and finite, of order $q^2\sigma/(300\pi^2)$.
- **Second-order means**: only the ensemble mean of the zero mode is computed, not the $2q$ harmonic.
- **Choosing between `rho_02` and `rho_02_hybrid`**:
  - `rho_02` is the cutoff-independent organisation and should be used at early times ($N\lesssim2$).
  - `rho_02_hybrid` re-routes the particle part, which reintroduces a cutoff tail at early times.
  - At late times the two differ through grid leakage. Check convergence in `np`, `nmu` and `pmin` before quoting late-time
    second-order values.
- **Momentum range**: at $N\approx1$, physical contributions reach $p\approx5$–$10$. Raise `pmax` if early times matter.
- **Runtime**: with the default grids each perturbation run takes minutes to an hour, and the default call makes
  11 runs (1 homogeneous plus 2 families × 5 values of $q$).

## Symbolic layer (as in 0.2, with the fixes)

`solve(action)` runs every exact Wick contraction of every in-in insertion through $O(\lambda h^2)$ and stores bare
integrands in SQLite (`python3 -m ininauto --symbolic ...`). `run(action)` is the small preview (`--quick`). These
integrands are unrenormalised and unintegrated; the numbers come from `evaluate`.

## Tests

```bash
python3 -m pip install -e ".[plots]" pytest
python3 -m pytest -q tests
```

The suite covers:
- action parsing, geometry, Legendre transform and Hilbert coefficients;
- in-in structures, Wick pairings and counterterms;
- the two bug fixes, plus a numerical reference check of $\langle\rho_0H_1\rangle$;
- the bridge, including a negative test that a deliberately wrong kernel is detected;
- a tiny numerical run.
