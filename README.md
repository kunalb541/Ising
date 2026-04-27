### Predictive Advantage, Regime Dependence, and Direction-Specific Causal Control in Kinetic Ising Dynamics

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Preprint — targeting *Physical Review E* (APS)

---

## Overview

This repository contains all simulation code and analysis for the paper. The paper studies how the choice of description affects **predictive and causal information** about future evolution in the 2D kinetic Ising model with Glauber dynamics.

Five results are reported:

| Axis | Description | Key result |
|------|-------------|------------|
| **1** | Organism comparison (ordered phase) | $E_{\geq 3}+J$ beats $\phi_\text{var8}$ (near-null baseline, $R^2\approx 0$): $\Delta R^2 = +0.391$, 95% CI $[+0.367, +0.418]$; against stronger rival $[W_0, B_\text{mag}]$, gap = $0.004$ (CI includes zero) |
| **2** | Regime crossover | Fine-over-coarse advantage reverses sign: $+0.178$ ordered, $-0.071$ disordered; near-$T_c$ probe ($T=2.20$) shows $\Delta R^2 = -0.013$ (CI spans zero) |
| **3** | Target specificity | $\Delta W$ is the most fine-specifically favoured target: $\Delta R^2 = +0.392$ |
| **4** | Direction-specific causal control | Structure-aligned generator bias produces 4/4 sign-flip; random-site arm (+4.70) > structural arm (+2.69) at $\varepsilon=0.10$ — two mechanisms identified |
| **D** | Predictor–handle dissociation | $[W_0, B_\text{mag}]$ ($R^2 = 0.389$) and $E_{\geq 3}+J$ ($R^2 \approx 0.393$) are statistically indistinguishable as predictors ($\Delta R^2 = 0.004$, CI includes zero), yet only the fine structure serves as a causal handle |

---

## System details

- 2D kinetic Ising model, $L=64$, periodic boundary conditions
- Glauber dynamics (heat-bath rule, checkerboard update)
- $T_\text{ordered} = 1.80$, $T_\text{disordered} = 2.50$ ($T_c \approx 2.269$)
- 1000 equilibration sweeps from random initial state; $K=20$ step prediction horizon
- 3000 independent samples for prediction experiments; 3000 for causal experiments
- Sub-experiments: near-$T_c$ crossover ($T=2.20$, $N=1000$); finite-size ($L\in\{32,64,128\}$, $N=500$ each); horizon ($K\in\{10,20,50\}$, $N=500$ each)

---

## Structural observables

| Symbol | Definition |
|--------|------------|
| $d_c(i)$ | Exposure degree: count of unlike neighbours of site $i$ |
| $E$ | Count of sites with $d_c = 4$ (isolated minority spins) |
| $E_{\geq 3}$ | Count of sites with $d_c \geq 3$ (highly exposed) |
| $J$ | Count of $d_c = 2$ sites with non-collinear unlike neighbours (junction bends) |
| $E_{\geq 3}+J$ | Two-column design matrix $[E_{\geq 3},\; J]$; each column gets its own Ridge coefficient |
| $\phi_\text{var8}$ | Variance of $|\bar{\sigma}_b|$ over $8\times 8$ blocks (coarse rival) |
| $W_0$ | Domain-wall count at $t_0$ |
| $B_\text{mag}$ | Absolute magnetisation $|\bar{\sigma}|$ at $t_0$ |
| $\Delta W$ | Future domain-wall change $W(t_0+K) - W(t_0)$ |

---

## Repository structure

```
ising/
├── ising.py                    # Main simulation: Axes 1-4 + Dissociation + sub-experiments
├── ising_c2_redesigned.py      # Supplementary C2 budget-normalisation probe
├── tests.py                    # Regression tests for core functions
├── paper.tex                   # Manuscript (REVTeX 4.2)
├── paper.pdf                   # Compiled preprint
├── build.sh                    # Runs simulation + compiles LaTeX
├── .gitignore
├── LICENSE                     # MIT
└── outputs/
    ├── figures/                # 7 figures (PNG + PDF)
    ├── tables/                 # CSV tables (axis1, axis2, axisB)
    └── data/                   # paper_macros.tex, paper_values.json
```

### File descriptions

**`ising.py`** runs the complete paper computation in a single script:
- Axis 1 (organism comparison), Axis 2 (regime crossover + near-Tc probe),
  Axis 3 (target specificity), Axis C (dissociation), Axis 4 (causal sign-flip + random-site control)
- Sub-experiments: near-$T_c$ crossover, finite-size ($L=32/64/128$), horizon ($K=10/20/50$)
- Generates all 7 figures and 3 CSV tables
- Writes `outputs/data/paper_macros.tex` and `paper_values.json` with all 50 paper numbers

**`ising_c2_redesigned.py`** is a standalone supplementary probe ($N=2000$). It tests whether site-selection alignment matters by comparing a site-selection aligned arm against a site-selection anti-aligned arm at matched perturbation budget. No CI-supported site-selection advantage was found at any $\varepsilon$; the causal leverage is direction-dominated (discussed in the paper, Axis 4). Outputs go to `outputs/ising_c2_redesigned/`.

**Axis naming:** The code uses internal names that differ from paper labels:

| Code | Paper |
|------|-------|
| Axis 1 | Axis 1 (organism comparison) |
| Axis A | Axis 2 (regime crossover) |
| Axis 2 | Axis 3 (target specificity) |
| Axis C | Dissociation |
| Axis B | Axis 4 (causal sign-flip) |

---

## Causal protocol (Axis 4 / paper Axis B)

The causal experiment modifies the Glauber transition kernel throughout the $K$-step evolution. Exposed sites ($d_c = 4$) are identified **dynamically at each sweep** from the current spin state (not from a mask fixed at $t_0$); this ensures structural targeting tracks the evolving configuration. The local wall-change contribution of flipping is:

$$\delta W_i = \sum_{\langle j \rangle}\bigl[\mathbf{1}[\sigma_j \neq -\sigma_i] - \mathbf{1}[\sigma_j \neq \sigma_i]\bigr]$$

For a fully exposed site, $\delta W_i = -4 < 0$.

| Arm | Modification | Physical effect |
|-----|-------------|----------------|
| **Aligned** | $\delta W_i < 0$ flips: $p \to p + \varepsilon$ | Promotes wall-reducing flips → accelerates coarsening |
| **Misaligned** | $\delta W_i < 0$ flips: $p \to p - \varepsilon$ | Suppresses wall-reducing flips → retards coarsening |
| **Random-site aligned** | Same direction logic at budget-matched random sites ($\rho = E_0/L^2$) | Class-level control: tests whether $d_c=4$ identity matters |
| **Baseline** | Pure Glauber | Unmodified |

Gap $= \Delta W_\text{baseline} - \Delta W_\text{arm}$. Positive gap = arm achieves greater wall reduction than baseline.

**Sign-flip discriminator:** aligned gap $> 0$ AND misaligned gap $< 0$, both 95% CIs excluding zero.  
**Result: 4/4** across $\varepsilon \in \{0.02, 0.05, 0.10, 0.20\}$.

**Class-level control result:** at $\varepsilon=0.10$, random arm gap $= +4.70$ vs structural arm gap $= +2.69$. The random arm is larger because at $T=1.80$, ~99% of random sites have $d_c=0$ (like neighbours), so "aligned" logic suppresses their flips → bulk fluctuation suppression, a qualitatively different mechanism from the structural arm (which is ceiling-limited at $d_c=4$ sites with $P_\text{flip}\approx 0.988$). Both mechanisms produce positive gaps; they are not equivalent.

---

## Predictive evaluation

Ridge regression ($\alpha = 1$), 5-fold cross-validation on $N = 3000$ configurations. Out-of-sample predictions are computed once on clean folds; bootstrap CIs resample $(y, \hat{y}_f, \hat{y}_c)$ triples (no train/test leakage). Primary metric:

$$\Delta R^2 = R^2(\text{fine organism}) - R^2(\phi_\text{var8})$$

---

## Requirements

```
numpy >= 1.23
numba >= 0.56
matplotlib
seaborn
pandas
pillow          # for figure format conversion
```

Python >= 3.9.

---

## Running

```bash
# Activate your Python environment (must have numba)
source /path/to/venv/bin/activate

# Run full simulation (parallelised over 8 workers; ~60-90 min)
python ising.py

# Run supplementary C2 probe
python ising_c2_redesigned.py

# Run regression tests
python tests.py

# Compile paper (requires TeX Live with REVTeX 4.2)
bash build.sh
```

All numerical outputs are written to `outputs/`. The paper PDF is written to `paper.pdf`.

---

## Tests

`tests.py` runs unit regression tests on the Numba core functions:

- **Glauber detailed balance** — acceptance probability matches analytic formula
- **Wall counting** — known configurations (all-aligned, single-defect) give correct counts
- **Exposure degree** — $d_c = 4$ for an isolated spin, $d_c = 0$ for a bulk spin
- **Junction classification** — corner vs collinear $d_c = 2$ sites correctly distinguished
- **Causal direction** — aligned arm achieves larger wall reduction than misaligned arm
- **Equilibration** — magnetisation distribution at $T > T_c$ is consistent with $\langle m \rangle \approx 0$

Run with: `python tests.py`

---

## License

MIT — see [LICENSE](LICENSE).
