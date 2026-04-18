# Predictive Advantage, Regime Dependence, and Direction-Specific Causal Control in Kinetic Ising Dynamics

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Preprint — targeting *Physical Review E* (APS)

---

## Overview

This repository contains all simulation code and analysis for the paper. The paper studies how the choice of description affects **predictive and causal information** about future evolution in the 2D kinetic Ising model with Glauber dynamics.

Five results are reported:

| Axis | Description | Key result |
|------|-------------|------------|
| **1** | Organism comparison (ordered phase) | $E_{\geq 3}+J$ beats coarse rival $\phi_\text{var8}$: $\Delta R^2 = +0.356$, 95% CI $[+0.331, +0.382]$ |
| **2** | Regime crossover | Fine-over-coarse advantage reverses sign: $+0.173$ ordered, $-0.050$ disordered |
| **3** | Target specificity | $\Delta W$ is the most fine-specifically favoured target: $\Delta R^2 = +0.355$ |
| **4** | Direction-specific causal control | Structure-aligned generator bias produces 4/4 sign-flip on future $\Delta W$ |
| **D** | Predictor–handle dissociation | Best predictor $[W_0, B_\text{mag}]$ ($R^2 = 0.375$) differs from best causal handle $E$ ($R^2 = 0.173$); gap $= 0.203$ |

---

## System details

- 2D kinetic Ising model, $L=64$, periodic boundary conditions
- Glauber dynamics (heat-bath rule, checkerboard update)
- $T_\text{ordered} = 1.80$, $T_\text{disordered} = 2.50$ ($T_c \approx 2.269$)
- 1000 equilibration sweeps from random initial state; $K=20$ step prediction horizon
- 3000 independent samples for prediction experiments; 3000 for causal experiments

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
├── ising.py                    # Main simulation: Axes 1-4 + Dissociation
├── ising_c2_redesigned.py      # Supplementary C2 budget-normalisation probe
├── tests.py                    # Regression tests for core functions
├── paper.tex                   # Manuscript (REVTeX 4.2)
├── paper.pdf                   # Compiled preprint
├── build.sh                    # Runs simulation + compiles LaTeX
├── .gitignore
├── LICENSE                     # MIT
└── outputs/
    ├── figures/                # 6 figures (PNG + PDF)
    ├── tables/                 # CSV tables (axis1, axis2, axisB)
    └── data/                   # paper_macros.tex, paper_values.json
```

### File descriptions

**`ising.py`** runs the complete paper computation in a single script:
- Axis 1 (organism comparison), Axis A/paper-2 (regime crossover),
  Axis 2/paper-3 (target specificity), Axis C (dissociation), Axis B/paper-4 (causal sign-flip)
- Generates all 6 figures and 3 CSV tables
- Writes `outputs/data/paper_macros.tex` and `paper_values.json` with all paper numbers

**`ising_c2_redesigned.py`** is a standalone supplementary probe. It compares E-targeted, random-targeted, and wall-density-targeted causal interventions at matched perturbation budget. Outputs go to `outputs/ising_c2_redesigned/`. This analysis does not appear in the paper.

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

The causal experiment modifies the Glauber transition kernel throughout the $K$-step evolution. At sites identified as exposed ($d_c = 4$) at $t_0$ (mask fixed, not dynamically updated), the local wall-change contribution of flipping is:

$$\delta W_i = \sum_{\langle j \rangle}\bigl[\mathbf{1}[\sigma_j \neq -\sigma_i] - \mathbf{1}[\sigma_j \neq \sigma_i]\bigr]$$

For a fully exposed site, $\delta W_i = -4 < 0$.

| Arm | Modification | Physical effect |
|-----|-------------|----------------|
| **Aligned** | $\delta W_i < 0$ flips: $p \to p + \varepsilon$ | Promotes wall-reducing flips → accelerates coarsening |
| **Misaligned** | $\delta W_i < 0$ flips: $p \to p - \varepsilon$ | Suppresses wall-reducing flips → retards coarsening |
| **Baseline** | Pure Glauber | Unmodified |

Gap $= \Delta W_\text{baseline} - \Delta W_\text{arm}$. Positive gap = arm achieves greater wall reduction than baseline.

**Sign-flip discriminator:** aligned gap $> 0$ AND misaligned gap $< 0$, both 95% CIs excluding zero.  
**Result: 4/4** across $\varepsilon \in \{0.02, 0.05, 0.10, 0.20\}$.

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
