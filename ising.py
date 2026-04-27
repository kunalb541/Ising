"""
ising_paper.py  (v2 — causal-correct)
=======================================
Three locked axes for the Ising flagship paper.

Axis 1 — Organism:  which fine predictor best predicts future dW?
          Fine organisms vs coarse rival phi_var8.
          Winner chosen by ΔR² with bootstrap CI.

Axis 2 — Target:    which future quantity does the champion organism predict?
          Champion organism on dW, dW_norm, future_topshare,
          changed_sites, changed_frac.

Axis B — Causal:    structure-aligned generator, sign-flip discriminator.
          CORRECT PROTOCOL (from ising_sign_symmetry.py, earned 4/4):
            ALIGNED:    dW<0 at exposed site -> p_flip += eps
            MISALIGNED: dW<0 at exposed site -> p_flip -= eps
          Both vs pure baseline Glauber.
          EARN CONDITION: aligned>0 AND misaligned<0, CIs exclude zero.
          dM diagnostic: must stay near-null (target specificity).

NO Potts. NO generic baseline-gap as main causal result.
NO targeted-vs-random as causal discriminator.
"""

import math, time, json, os, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from numba import njit
except ImportError:
    raise ImportError("pip install numba")

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIG
# =============================================================================

FULL_RUN    = True
MAX_WORKERS = 8
SEED        = 42

if FULL_RUN:
    N_PRED   = 3000   # worlds for axes 1, 2, A, C
    N_CAUSAL = 3000   # worlds for axis B
    EQUIL    = 1000
    N_BOOT   = 1000
else:
    N_PRED   = 300
    N_CAUSAL = 300
    EQUIL    = 300
    N_BOOT   = 300

L        = 64
K        = 20
T_ORD    = 1.80;  BETA_ORD = 1.0 / T_ORD
T_DIS    = 2.50;  BETA_DIS = 1.0 / T_DIS
BLOCK    = 8
EPS_LIST = [0.00, 0.02, 0.05, 0.10, 0.20]
ALPHA    = 1.0

OUT      = Path("outputs")
FIGS     = OUT / "figures"
TABS     = OUT / "tables"
DATA     = OUT / "data"
LOGS     = OUT / "logs"

sns.set_theme(style="whitegrid", font_scale=1.25)
C_FINE    = "#2E86AB"
C_COARSE  = "#E84855"
C_ALIGN   = "#2E86AB"
C_MISALN  = "#E84855"
C_NEUTRAL = "#6C757D"

# =============================================================================
# 2. IO
# =============================================================================

def make_dirs():
    for d in [FIGS, TABS, DATA, LOGS]:
        d.mkdir(parents=True, exist_ok=True)

_LOG = []
def log(msg):
    _LOG.append(msg); print(msg, flush=True)

def flush_log():
    with open(LOGS / "run.log", "w") as f: f.write("\n".join(_LOG))

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def save_csv(df, path): df.to_csv(path, index=False)

def tf(x, d=3): return f"{x:.{d}f}"

# =============================================================================
# 3. STATS
# =============================================================================
"""
Drop-in replacement for ising.py lines 111-149.

Root fix: dr2_boot and r2_boot previously bootstrapped raw data before CV,
causing duplicate rows to appear in both train and test folds (~37% overlap
at N=3000, k=5). Now OOS predictions are computed once on clean k-fold
splits, and the bootstrap resamples only the evaluation metric.

ridge_cv becomes a thin wrapper so all call sites are unchanged.
"""

def r2_oos(y, yhat):
    ss_r = np.sum((y - yhat) ** 2)
    ss_t = np.sum((y - np.mean(y)) ** 2)
    return 0.0 if ss_t < 1e-20 else float(1 - ss_r / ss_t)


def get_cv_preds(X, y, alpha=ALPHA, k=5, seed=0):
    """
    True OOS predictions via k-fold CV -- no duplicate-row leakage.
    Scaling is fit on training folds only and applied to the test fold.
    Returns yhat array aligned with y.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    yhat = np.full(len(y), np.nan)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(tr) < 4 or len(te) < 2:
            continue
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        sd[sd < 1e-8] = 1.0
        Xs = (Xtr - mu) / sd
        ym = ytr.mean()
        yc = ytr - ym
        p = Xs.shape[1]
        b = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(p), Xs.T @ yc)
        yhat[te] = (Xte - mu) / sd @ b + ym
    return yhat


def ridge_cv(X, y, alpha=ALPHA, k=5, seed=0):
    """Thin wrapper -- unchanged call signature, no leakage."""
    yhat = get_cv_preds(X, y, alpha=alpha, k=k, seed=seed)
    mask = np.isfinite(yhat)
    return r2_oos(y[mask], yhat[mask])


def dr2_boot(Xf, Xc, y, nb=500, seed=0):
    """
    Bootstrap CI on delta-R2 (fine minus coarse).

    OOS predictions are computed once on the full dataset via clean k-fold.
    The bootstrap then resamples (y, yhat_f, yhat_c) triples to estimate
    the sampling distribution of delta-R2. No train/test contamination.
    """
    yhat_f = get_cv_preds(Xf, y, seed=seed)
    yhat_c = get_cv_preds(Xc, y, seed=seed + 1)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(nb)
    for b in range(nb):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        ss_t = np.sum((y_b - np.mean(y_b)) ** 2)
        if ss_t < 1e-20:
            vals[b] = 0.0
            continue
        r2_f = 1.0 - np.sum((y_b - yhat_f[idx]) ** 2) / ss_t
        r2_c = 1.0 - np.sum((y_b - yhat_c[idx]) ** 2) / ss_t
        vals[b] = r2_f - r2_c
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def r2_boot(X, y, nb=500, seed=0):
    """
    Bootstrap CI on R2.
    Same fix: compute OOS preds once, resample only the metric.
    """
    yhat = get_cv_preds(X, y, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(nb)
    for b in range(nb):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        ss_t = np.sum((y_b - np.mean(y_b)) ** 2)
        if ss_t < 1e-20:
            vals[b] = 0.0
            continue
        vals[b] = 1.0 - np.sum((y_b - yhat[idx]) ** 2) / ss_t
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))



def mci(vals, nb=400, seed=0):
    v = np.array(vals, float); rng = np.random.default_rng(seed)
    b = np.array([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(nb)])
    return float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# =============================================================================
# 4. NUMBA CORE
# =============================================================================

@njit(cache=False)
def glauber_sweep(spins, beta):
    L2 = spins.shape[0]
    for parity in range(2):
        for i in range(L2):
            im1 = i-1 if i>0 else L2-1; ip1 = i+1 if i<L2-1 else 0
            start = parity if (i%2==0) else 1-parity
            for j in range(start, L2, 2):
                jm1 = j-1 if j>0 else L2-1; jp1 = j+1 if j<L2-1 else 0
                h = (spins[im1,j]+spins[ip1,j]+spins[i,jm1]+spins[i,jp1])
                p = 1.0/(1.0+math.exp(-2.0*beta*h))
                spins[i,j] = 1 if np.random.random()<p else -1


@njit(cache=False)
def run_glauber(spins, beta, n):
    for _ in range(n): glauber_sweep(spins, beta)


@njit(cache=False)
def numba_seed(s):
    np.random.seed(s)


@njit(cache=False)
def wall_count(spins):
    L2=spins.shape[0]; W=0
    for i in range(L2):
        ip1=i+1 if i<L2-1 else 0
        for j in range(L2):
            jp1=j+1 if j<L2-1 else 0
            if spins[i,j]!=spins[ip1,j]: W+=1
            if spins[i,j]!=spins[i,jp1]: W+=1
    return float(W)


@njit(cache=False)
def abs_mag(spins):
    L2=spins.shape[0]; s=0.0
    for i in range(L2):
        for j in range(L2): s+=spins[i,j]
    return abs(s)/(L2*L2)


# =============================================================================
# 5. ISING OBSERVABLES
# =============================================================================

@njit(cache=False)
def local_dc(spins, i, j):
    L2=spins.shape[0]; s=spins[i,j]; c=0
    if spins[i-1 if i>0 else L2-1, j]!=s: c+=1
    if spins[i+1 if i<L2-1 else 0,  j]!=s: c+=1
    if spins[i, j-1 if j>0 else L2-1]!=s: c+=1
    if spins[i, j+1 if j<L2-1 else 0]!=s: c+=1
    return c


@njit(cache=False)
def count_E(spins):
    L2=spins.shape[0]; E=0
    for i in range(L2):
        for j in range(L2):
            if local_dc(spins,i,j)==4: E+=1
    return float(E)


@njit(cache=False)
def count_E_ge3(spins):
    L2=spins.shape[0]; E=0
    for i in range(L2):
        for j in range(L2):
            if local_dc(spins,i,j)>=3: E+=1
    return float(E)


@njit(cache=False)
def count_J(spins):
    L2=spins.shape[0]; J=0
    for i in range(L2):
        im1=i-1 if i>0 else L2-1; ip1=i+1 if i<L2-1 else 0
        for j in range(L2):
            jm1=j-1 if j>0 else L2-1; jp1=j+1 if j<L2-1 else 0
            s=spins[i,j]
            up=1 if spins[im1,j]!=s else 0; dn=1 if spins[ip1,j]!=s else 0
            lf=1 if spins[i,jm1]!=s else 0; rt=1 if spins[i,jp1]!=s else 0
            d=up+dn+lf+rt
            if d==2 and not ((up==1 and dn==1) or (lf==1 and rt==1)): J+=1
    return float(J)


@njit(cache=False)
def count_minority(spins):
    L2=spins.shape[0]; n=0
    for i in range(L2):
        for j in range(L2):
            if spins[i,j]==1: n+=1
    return float(min(n, L2*L2-n))


@njit(cache=False)
def phi_var8(spins):
    L2=spins.shape[0]; bs=8; nb=L2//bs; nblocks=nb*nb
    vals=np.empty(nblocks,np.float64); idx=0; area=float(bs*bs)
    for bi in range(nb):
        for bj in range(nb):
            s=0.0
            for di in range(bs):
                for dj in range(bs): s+=spins[bi*bs+di,bj*bs+dj]
            vals[idx]=abs(s/area); idx+=1
    mu=0.0
    for k in range(nblocks): mu+=vals[k]
    mu/=nblocks; var=0.0
    for k in range(nblocks): var+=(vals[k]-mu)**2
    return var/nblocks


@njit(cache=False)
def count_changed_sites(sf, s0):
    L2=sf.shape[0]; c=0
    for i in range(L2):
        for j in range(L2):
            if sf[i,j]!=s0[i,j]: c+=1
    return float(c)


@njit(cache=False)
def future_topshare(sf, s0):
    L2=sf.shape[0]; bs=8; nb=L2//bs; nblocks=nb*nb
    blocks=np.zeros(nblocks,np.float64); idx=0
    for bi in range(nb):
        for bj in range(nb):
            s=0.0
            for di in range(bs):
                for dj in range(bs):
                    i=bi*bs+di; j=bj*bs+dj
                    ip1=i+1 if i<L2-1 else 0; jp1=j+1 if j<L2-1 else 0
                    w0h=1 if s0[i,j]!=s0[i,jp1] else 0
                    w0v=1 if s0[i,j]!=s0[ip1,j] else 0
                    wfh=1 if sf[i,j]!=sf[i,jp1] else 0
                    wfv=1 if sf[i,j]!=sf[ip1,j] else 0
                    lost=(w0h-wfh)+(w0v-wfv)
                    if lost>0: s+=lost
            blocks[idx]=s; idx+=1
    total=0.0
    for k in range(nblocks): total+=blocks[k]
    if total<1e-6: return 0.0
    k10=max(1,nblocks//10); tmp=blocks.copy(); tmp.sort()
    thresh=tmp[nblocks-k10]; top=0.0
    for k in range(nblocks):
        if blocks[k]>=thresh: top+=blocks[k]
    return top/total


# =============================================================================
# 6. CAUSAL NUMBA CORE  (exact code from ising_sign_symmetry.py — earned 4/4)
# =============================================================================

@njit(cache=False)
def build_exposed_mask(spins):
    L2=spins.shape[0]; mask=np.zeros((L2,L2),dtype=np.int8)
    for i in range(L2):
        for j in range(L2):
            if local_dc(spins,i,j)==4: mask[i,j]=1
    return mask


@njit(cache=False)
def structure_aligned_sweep(spins, beta, eps, aligned):
    """
    Checkerboard Glauber with structure-aligned generator bias.
    DYNAMIC: exposed sites (d_c=4) identified each sweep from current spin state.
    d_c=4 ↔ s*h == -4 (all 4 neighbours unlike current spin).
    For these sites dW=-4 always; aligned promotes, misaligned suppresses.
    aligned=True:  p_flip += eps at each currently-exposed site
    aligned=False: p_flip -= eps at each currently-exposed site
    """
    L2=spins.shape[0]
    for parity in range(2):
        for i in range(L2):
            im1=i-1 if i>0 else L2-1; ip1=i+1 if i<L2-1 else 0
            start=parity if (i%2==0) else 1-parity
            for j in range(start, L2, 2):
                jm1=j-1 if j>0 else L2-1; jp1=j+1 if j<L2-1 else 0
                s=spins[i,j]
                h=(spins[im1,j]+spins[ip1,j]+spins[i,jm1]+spins[i,jp1])
                p=1.0/(1.0+math.exp(beta*(2.0*s*h)))
                if s*h==-4 and eps>0.0:
                    # d_c=4: dW=-4 always; aligned promotes, misaligned suppresses
                    if aligned: p+=eps
                    else:       p-=eps
                    if p<0.0: p=0.0
                    if p>1.0: p=1.0
                if np.random.random()<p: spins[i,j]=-s


@njit(cache=False)
def run_aligned(spins, beta, eps, aligned_bool, n):
    for _ in range(n): structure_aligned_sweep(spins, beta, eps, aligned_bool)


@njit(cache=False)
def random_targeted_sweep(spins, beta, eps, rho):
    """
    Budget-matched random-site control for Axis 4.
    Each site selected independently with probability rho (= E0/L^2 at t=0).
    Selected sites: promote wall-reducing flips (s*h<0), suppress wall-increasing (s*h>0).
    Same directional logic as structure_aligned_sweep but without d_c=4 class selection.
    Tests whether structural class identity (d_c=4) is causally necessary vs direction alone.
    """
    L2 = spins.shape[0]
    for parity in range(2):
        for i in range(L2):
            im1 = i-1 if i > 0 else L2-1; ip1 = i+1 if i < L2-1 else 0
            start = parity if (i % 2 == 0) else 1 - parity
            for j in range(start, L2, 2):
                jm1 = j-1 if j > 0 else L2-1; jp1 = j+1 if j < L2-1 else 0
                s = spins[i, j]
                h = (spins[im1,j] + spins[ip1,j] + spins[i,jm1] + spins[i,jp1])
                p = 1.0 / (1.0 + math.exp(beta * (2.0 * s * h)))
                if eps > 0.0 and np.random.random() < rho:
                    sh = s * h   # sh<0 means flip reduces walls; sh>0 means flip increases walls
                    if sh < 0:
                        p = p + eps
                        if p > 1.0: p = 1.0
                    elif sh > 0:
                        p = p - eps
                        if p < 0.0: p = 0.0
                if np.random.random() < p:
                    spins[i, j] = -s


@njit(cache=False)
def run_random_targeted(spins, beta, eps, rho, n):
    for _ in range(n):
        random_targeted_sweep(spins, beta, eps, rho)


@njit(cache=False)
def random_filtered_sweep(spins, beta, eps, rho, dc_filter):
    """
    Filtered random-site control for mechanism decomposition.
    Selects sites with prob rho per sweep (same as full random arm), then
    applies directional bias only to those passing the d_c filter:
      dc_filter == 0  : only d_c=0 sites (sh==4) — bulk-suppression mechanism
      dc_filter == 3  : only d_c>=3 sites (sh<=-2) — exposed-class mechanism
    """
    L2 = spins.shape[0]
    for parity in range(2):
        for i in range(L2):
            im1 = i-1 if i > 0 else L2-1; ip1 = i+1 if i < L2-1 else 0
            start = parity if (i % 2 == 0) else 1 - parity
            for j in range(start, L2, 2):
                jm1 = j-1 if j > 0 else L2-1; jp1 = j+1 if j < L2-1 else 0
                s = spins[i, j]
                h = (spins[im1,j] + spins[ip1,j] + spins[i,jm1] + spins[i,jp1])
                p = 1.0 / (1.0 + math.exp(beta * (2.0 * s * h)))
                if eps > 0.0 and np.random.random() < rho:
                    sh = s * h
                    pass_filter = False
                    if dc_filter == 0:
                        pass_filter = (sh == 4)      # d_c = 0
                    elif dc_filter == 3:
                        pass_filter = (sh <= -2)     # d_c >= 3
                    if pass_filter:
                        if sh < 0:
                            p = p + eps
                            if p > 1.0: p = 1.0
                        elif sh > 0:
                            p = p - eps
                            if p < 0.0: p = 0.0
                if np.random.random() < p:
                    spins[i, j] = -s


@njit(cache=False)
def run_random_filtered(spins, beta, eps, rho, dc_filter, n):
    for _ in range(n):
        random_filtered_sweep(spins, beta, eps, rho, dc_filter)


@njit(cache=False)
def mean_pflip_dc4(spins, beta):
    """
    Average baseline Glauber flip probability at d_c=4 sites.
    Used to compute eps_eff = (1 - p_flip_dc4) for the structural arm
    (since aligned promotion p -> p + eps is clipped at p=1).
    Returns NaN if no d_c=4 sites exist.
    """
    L2 = spins.shape[0]
    total_p = 0.0
    n_dc4 = 0
    for i in range(L2):
        im1 = i-1 if i > 0 else L2-1; ip1 = i+1 if i < L2-1 else 0
        for j in range(L2):
            jm1 = j-1 if j > 0 else L2-1; jp1 = j+1 if j < L2-1 else 0
            s = spins[i, j]
            h = (spins[im1,j] + spins[ip1,j] + spins[i,jm1] + spins[i,jp1])
            sh = s * h
            if sh == -4:  # d_c = 4
                p = 1.0 / (1.0 + math.exp(beta * (2.0 * sh)))
                total_p += p
                n_dc4 += 1
    if n_dc4 == 0:
        return -1.0
    return total_p / n_dc4


# =============================================================================
# 7. WARMUP
# =============================================================================

def warmup():
    np.random.seed(0)
    d=np.ones((16,16),dtype=np.int8)
    glauber_sweep(d,0.5); run_glauber(d,0.5,1); wall_count(d); abs_mag(d)
    local_dc(d,0,0); count_E(d); count_E_ge3(d); count_J(d)
    count_minority(d); phi_var8(d)
    d8=np.ones((8,8),dtype=np.int8)
    count_changed_sites(d8,d8); future_topshare(d8,d8)
    numba_seed(0)
    structure_aligned_sweep(d,0.5,0.05,True)
    structure_aligned_sweep(d,0.5,0.05,False)
    run_aligned(d,0.5,0.05,True,1); run_aligned(d,0.5,0.05,False,1)
    random_targeted_sweep(d,0.5,0.05,0.1)
    run_random_targeted(d,0.5,0.05,0.1,1)
    random_filtered_sweep(d,0.5,0.05,0.1,0)
    random_filtered_sweep(d,0.5,0.05,0.1,3)
    run_random_filtered(d,0.5,0.05,0.1,0,1)
    run_random_filtered(d,0.5,0.05,0.1,3,1)
    mean_pflip_dc4(d, 0.5)
    log("warmup OK")


# =============================================================================
# 8. WORLD SIMULATIONS
# =============================================================================

def sim_pred_world(args):
    seed, beta = args
    np.random.seed(seed)
    numba_seed(seed)
    spins=np.random.choice(np.array([-1,1],dtype=np.int8),size=(L,L))
    run_glauber(spins, beta, EQUIL)
    sp0=spins.copy(); W0=wall_count(sp0)
    feats = {
        'E':        count_E(sp0),
        'E_ge3':    count_E_ge3(sp0),
        'J':        count_J(sp0),
        'minority': count_minority(sp0),
        'phi8':     phi_var8(sp0),
        'W0':       W0,
        'Bmag':     abs_mag(sp0),
        # E_ge3_plus_J and E_plus_J are two-column design matrices,
        # built in the analysis section from raw E_ge3, E, J fields.
    }
    fut=sp0.copy(); run_glauber(fut, beta, K); W1=wall_count(fut); dW=W1-W0
    tgts = {
        'dW':                   dW,
        'dW_norm':              dW/(W0+1e-6),
        'future_topshare':      future_topshare(fut, sp0),
        'future_changed_sites': count_changed_sites(fut, sp0),
        'future_changed_frac':  count_changed_sites(fut, sp0)/(L*L),
    }
    return feats, tgts


def sim_causal_world(args):
    """
    Axis 4 PROTOCOL (correct):
    Columns 0-3: aligned/misaligned d_c=4-targeted arms vs baseline Glauber.
    Columns 4-5: budget-matched random-site aligned arm (class-level control).
    gap = baseline - arm; positive = arm accelerates coarsening.
    rho = E0/(L^2): density of d_c=4 sites at t=0, used for budget matching.
    """
    seed, eps_list = args
    np.random.seed(seed)
    numba_seed(seed)
    spins=np.random.choice(np.array([-1,1],dtype=np.int8),size=(L,L))
    run_glauber(spins, BETA_ORD, EQUIL)
    sp0=spins.copy(); W0=wall_count(sp0); M0=abs_mag(sp0)

    # Budget for random control: density of d_c=4 sites at t=0
    E0 = count_E(sp0)
    rho = float(E0) / float(L * L)

    # baseline: pure unperturbed Glauber
    base=sp0.copy(); run_glauber(base, BETA_ORD, K)
    base_dW=wall_count(base)-W0; base_dM=abs_mag(base)-M0

    results=np.zeros((len(eps_list),6),dtype=np.float64)
    for ei, eps in enumerate(eps_list):
        al=sp0.copy(); run_aligned(al, BETA_ORD, float(eps), True, K)
        al_dW=wall_count(al)-W0; al_dM=abs_mag(al)-M0

        mis=sp0.copy(); run_aligned(mis, BETA_ORD, float(eps), False, K)
        mis_dW=wall_count(mis)-W0; mis_dM=abs_mag(mis)-M0

        # Random-site aligned arm (budget-matched, same directional logic)
        rnd=sp0.copy(); run_random_targeted(rnd, BETA_ORD, float(eps), rho, K)
        rnd_dW=wall_count(rnd)-W0; rnd_dM=abs_mag(rnd)-M0

        results[ei,0]=base_dW-al_dW    # aligned gap dW     (want >0)
        results[ei,1]=base_dW-mis_dW   # misaligned gap dW  (want <0)
        results[ei,2]=base_dM-al_dM    # aligned gap dM     (want ~0)
        results[ei,3]=base_dM-mis_dM   # misaligned gap dM
        results[ei,4]=base_dW-rnd_dW   # random-aligned gap dW (class-level control)
        results[ei,5]=base_dM-rnd_dM   # random-aligned gap dM
    return results


def sim_pred_world_param(args):
    """Parameterized predictive world for near-Tc, finite-size, and horizon sub-experiments."""
    seed, beta, L_p, K_p, equil_p = args
    np.random.seed(seed)
    numba_seed(seed)
    spins = np.random.choice(np.array([-1,1], dtype=np.int8), size=(L_p, L_p))
    run_glauber(spins, beta, equil_p)
    sp0 = spins.copy(); W0 = wall_count(sp0)
    feats = {
        'E':     count_E(sp0),
        'E_ge3': count_E_ge3(sp0),
        'J':     count_J(sp0),
        'phi8':  phi_var8(sp0),
        'W0':    W0,
        'Bmag':  abs_mag(sp0),
    }
    fut = sp0.copy(); run_glauber(fut, beta, K_p); W1 = wall_count(fut)
    return feats, {'dW': W1 - W0}


def sim_causal_world_param(args):
    """
    Parameterized causal world: same protocol as sim_causal_world but at arbitrary beta.
    Returns 6 columns identical to sim_causal_world: [aligned_dW, misaligned_dW,
    aligned_dM, misaligned_dM, random_dW, random_dM].
    """
    seed, eps_list, beta_p, equil_p = args
    np.random.seed(seed)
    numba_seed(seed)
    spins = np.random.choice(np.array([-1,1], dtype=np.int8), size=(L,L))
    run_glauber(spins, beta_p, equil_p)
    sp0 = spins.copy(); W0 = wall_count(sp0); M0 = abs_mag(sp0)
    E0 = count_E(sp0)
    rho = float(E0) / float(L * L)

    base = sp0.copy(); run_glauber(base, beta_p, K)
    base_dW = wall_count(base) - W0; base_dM = abs_mag(base) - M0

    results = np.zeros((len(eps_list), 6), dtype=np.float64)
    for ei, eps in enumerate(eps_list):
        al = sp0.copy(); run_aligned(al, beta_p, float(eps), True, K)
        al_dW = wall_count(al) - W0; al_dM = abs_mag(al) - M0

        mis = sp0.copy(); run_aligned(mis, beta_p, float(eps), False, K)
        mis_dW = wall_count(mis) - W0; mis_dM = abs_mag(mis) - M0

        rnd = sp0.copy(); run_random_targeted(rnd, beta_p, float(eps), rho, K)
        rnd_dW = wall_count(rnd) - W0; rnd_dM = abs_mag(rnd) - M0

        results[ei,0] = base_dW - al_dW
        results[ei,1] = base_dW - mis_dW
        results[ei,2] = base_dM - al_dM
        results[ei,3] = base_dM - mis_dM
        results[ei,4] = base_dW - rnd_dW
        results[ei,5] = base_dM - rnd_dM
    return results


def sim_causal_filter_world(args):
    """
    Filtered random-site control for mechanism decomposition (eps=0.10 only).
    Tests whether the random-arm gap comes from d_c=0 sites (bulk-suppression
    hypothesis) or from d_c>=3 sites (exposed-class hypothesis).
    Same selection rate rho = E0/L^2 as the full random arm; the filter
    restricts WHICH selected sites get the directional bias.
    Returns: [dc0_only_dW_gap, dcge3_only_dW_gap, baseline_dW, full_random_dW_gap]
    """
    seed, eps_test = args
    np.random.seed(seed)
    numba_seed(seed)
    spins = np.random.choice(np.array([-1,1], dtype=np.int8), size=(L,L))
    run_glauber(spins, BETA_ORD, EQUIL)
    sp0 = spins.copy(); W0 = wall_count(sp0)
    E0 = count_E(sp0)
    rho = float(E0) / float(L * L)

    base = sp0.copy(); run_glauber(base, BETA_ORD, K)
    base_dW = wall_count(base) - W0

    dc0 = sp0.copy(); run_random_filtered(dc0, BETA_ORD, float(eps_test), rho, 0, K)
    dc0_dW = wall_count(dc0) - W0

    dcge3 = sp0.copy(); run_random_filtered(dcge3, BETA_ORD, float(eps_test), rho, 3, K)
    dcge3_dW = wall_count(dcge3) - W0

    full = sp0.copy(); run_random_targeted(full, BETA_ORD, float(eps_test), rho, K)
    full_dW = wall_count(full) - W0

    return np.array([base_dW - dc0_dW, base_dW - dcge3_dW, base_dW, base_dW - full_dW])


# =============================================================================
# 9. PARALLEL
# =============================================================================

def run_parallel(fn, jobs, desc):
    results=[None]*len(jobs)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fmap={ex.submit(fn,j):i for i,j in enumerate(jobs)}
        done=0; step=max(1,len(jobs)//4)
        for f in as_completed(fmap):
            results[fmap[f]]=f.result(); done+=1
            if done%step==0 or done==len(jobs):
                log(f"  {desc} {done}/{len(jobs)}")
    return results


# =============================================================================
# 10. FIGURES
# =============================================================================

def fig_schematic(sp0):
    c=L//2; hw=16
    patch=sp0[c-hw:c+hw, c-hw:c+hw].astype(int); L2=patch.shape[0]
    exp_y,exp_x=[],[]
    for i in range(L2):
        for j in range(L2):
            s=patch[i,j]; dc=0
            dc+=1 if patch[(i-1)%L2,j]!=s else 0
            dc+=1 if patch[(i+1)%L2,j]!=s else 0
            dc+=1 if patch[i,(j-1)%L2]!=s else 0
            dc+=1 if patch[i,(j+1)%L2]!=s else 0
            if dc==4: exp_y.append(i); exp_x.append(j)

    fig,ax=plt.subplots(figsize=(5.5,5.5))
    ax.imshow(patch,cmap="RdBu_r",vmin=-1.5,vmax=1.5,origin="upper",interpolation="nearest",alpha=0.65)
    for i in range(L2):
        for j in range(L2):
            ip1=(i+1)%L2; jp1=(j+1)%L2
            if patch[i,j]!=patch[i,jp1]:
                ax.plot([j+0.5,j+0.5],[i-0.5,i+0.5],color='#222222',lw=1.2,alpha=0.55)
            if patch[i,j]!=patch[ip1,j]:
                ax.plot([j-0.5,j+0.5],[i+0.5,i+0.5],color='#222222',lw=1.2,alpha=0.55)
    for yi,xi in zip(exp_y,exp_x):
        ax.add_patch(plt.Circle((xi,yi),0.42,fill=False,edgecolor='gold',lw=2.5,zorder=5))
    handles=[
        mpatches.Patch(facecolor="#5a9fd4",alpha=0.8,label="Spin $+1$"),
        mpatches.Patch(facecolor="#d73027",alpha=0.8,label="Spin $-1$"),
        plt.Line2D([0],[0],color='#222222',lw=1.5,label="Domain wall"),
        plt.Line2D([0],[0],marker='o',color='w',markeredgecolor='gold',
                   markersize=11,markeredgewidth=2.5,label="Exposed site ($E$, dc$=4$)"),
    ]
    ax.legend(handles=handles,loc="upper right",fontsize=9,framealpha=0.95)
    ax.set_title("Ising structural objects",fontsize=12,fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p=FIGS/"ising_schematic.png"; fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_axis1(org_results, ref_dr2=None):
    names=[r['name'] for r in org_results]
    means=[r['dr2'] for r in org_results]
    los=[r['ci_lo'] for r in org_results]
    his=[r['ci_hi'] for r in org_results]
    colors=[C_FINE if m>0 else C_COARSE for m in means]

    fig,ax=plt.subplots(figsize=(8.5,4.5))
    x=np.arange(len(names))
    ax.bar(x,means,color=colors,alpha=0.85,width=0.6,zorder=3)
    ax.errorbar(x,means,
                yerr=[[m-l for m,l in zip(means,los)],[h-m for m,h in zip(means,his)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.axhline(0,color='black',lw=0.9,ls='--',zorder=2)
    if ref_dr2 is not None:
        ax.axhline(ref_dr2, color='#888800', lw=1.4, ls=':', zorder=5,
                   label=r"$[W_0,B_{\rm mag}]$ vs $\phi_{\rm var8}$ threshold")
        ax.legend(fontsize=9, loc='lower right')
    best_i=int(np.argmax(means))
    ax.annotate("best",xy=(x[best_i],his[best_i]+0.005),ha='center',
                fontsize=9.5,color='#1a5276',fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha='right',fontsize=10)
    ax.set_ylabel(r"$\Delta R^2$ (fine organism $-$ $\phi_{\rm var8}$)",fontsize=10.5)
    ax.set_title(f"Axis 1 — Organism comparison  (target: $\\Delta W$, $T={T_ORD}$, $L={L}$, $K={K}$)",
                 fontsize=11)
    fig.tight_layout()
    p=FIGS/"ising_axis1_organism.png"; fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_axis2(tgt_results, best_org):
    names=[r['name'] for r in tgt_results]
    means=[r['dr2'] for r in tgt_results]
    los=[r['ci_lo'] for r in tgt_results]
    his=[r['ci_hi'] for r in tgt_results]

    fig,ax=plt.subplots(figsize=(8.5,4.5))
    x=np.arange(len(names))
    ax.bar(x,means,color=C_FINE,alpha=0.85,width=0.6,zorder=3)
    ax.errorbar(x,means,
                yerr=[[m-l for m,l in zip(means,los)],[h-m for m,h in zip(means,his)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.axhline(0,color='black',lw=0.9,ls='--',zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha='right',fontsize=10)
    ax.set_ylabel(r"$\Delta R^2$ (champion organism $-$ $\phi_{\rm var8}$)",fontsize=10.5)
    ax.set_title(f"Axis 2 — Target comparison: fine-over-coarse $\\Delta R^2$  ({best_org}, $T={T_ORD}$)",fontsize=11)
    fig.tight_layout()
    p=FIGS/"ising_axis2_target.png"; fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_axisB_sign_flip(causal_arr):
    """Main causal figure: aligned vs misaligned on dW across eps ladder."""
    al_m,al_lo,al_hi=[],[],[]
    mi_m,mi_lo,mi_hi=[],[],[]
    for ei in range(len(EPS_LIST)):
        m,lo,hi=mci(causal_arr[:,ei,0],seed=ei*10+1)
        al_m.append(m); al_lo.append(lo); al_hi.append(hi)
        m,lo,hi=mci(causal_arr[:,ei,1],seed=ei*10+2)
        mi_m.append(m); mi_lo.append(lo); mi_hi.append(hi)

    fig,ax=plt.subplots(figsize=(7.5,5.2))
    x=np.arange(len(EPS_LIST)); w=0.36
    ax.bar(x-w/2,al_m,w,color=C_ALIGN,alpha=0.85,label="Aligned (promotes wall reduction)",zorder=3)
    ax.bar(x+w/2,mi_m,w,color=C_MISALN,alpha=0.85,label="Misaligned (suppresses wall reduction)",zorder=3)
    ax.errorbar(x-w/2,al_m,
                yerr=[[m-l for m,l in zip(al_m,al_lo)],[h-m for m,h in zip(al_m,al_hi)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.errorbar(x+w/2,mi_m,
                yerr=[[m-l for m,l in zip(mi_m,mi_lo)],[h-m for m,h in zip(mi_m,mi_hi)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.axhline(0,color='black',lw=0.9,ls='--',zorder=2)
    y_check = max(al_hi) + 1.5          # consistent height in white space above all bars
    ax.set_ylim(top=y_check + 2.5)      # ensure axis has room
    for ei in range(len(EPS_LIST)):
        if al_lo[ei]>0 and mi_hi[ei]<0:
            ax.annotate(r"$\checkmark$",xy=(x[ei],y_check),ha='center',
                        fontsize=14,color='#1a7a1a',fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['0 (null)' if e==0 else str(e) for e in EPS_LIST],fontsize=11)
    ax.set_xlabel(r"Generator bias $\varepsilon$",fontsize=12)
    ax.set_ylabel(r"Gap vs baseline $(\Delta W_{\rm base} - \Delta W_{\rm arm})$",fontsize=11)
    ax.set_title("Axis 4 — Direction-specific generator control\n"
                 r"Aligned $>0$, misaligned $<0$: both CIs exclude zero ($\checkmark$ = sign flip)",
                 fontsize=11)
    ax.legend(fontsize=9.5,loc='upper center',
              bbox_to_anchor=(0.5,-0.14),ncol=2,borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    p=FIGS/"ising_axisB_aligned_vs_misaligned.png"
    fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_axisB_target_specificity(causal_arr):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    specs=[(axes[0],r"$\Delta W$ (wall change)",0,1),
           (axes[1],r"$\Delta M$ (magnetisation change)",2,3)]
    for ax,title,ac,mc in specs:
        al_m,al_lo,al_hi=[],[],[]; mi_m,mi_lo,mi_hi=[],[],[]
        for ei in range(len(EPS_LIST)):
            m,lo,hi=mci(causal_arr[:,ei,ac],seed=ei*10+ac*5+1)
            al_m.append(m); al_lo.append(lo); al_hi.append(hi)
            m,lo,hi=mci(causal_arr[:,ei,mc],seed=ei*10+mc*5+2)
            mi_m.append(m); mi_lo.append(lo); mi_hi.append(hi)
        x=np.arange(len(EPS_LIST)); w=0.36
        ax.bar(x-w/2,al_m,w,color=C_ALIGN,alpha=0.85,label="Aligned",zorder=3)
        ax.bar(x+w/2,mi_m,w,color=C_MISALN,alpha=0.85,label="Misaligned",zorder=3)
        ax.errorbar(x-w/2,al_m,
                    yerr=[[m-l for m,l in zip(al_m,al_lo)],[h-m for m,h in zip(al_m,al_hi)]],
                    fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
        ax.errorbar(x+w/2,mi_m,
                    yerr=[[m-l for m,l in zip(mi_m,mi_lo)],[h-m for m,h in zip(mi_m,mi_hi)]],
                    fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
        ax.axhline(0,color='black',lw=0.9,ls='--',zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(['0 (null)' if e==0 else str(e) for e in EPS_LIST])
        ax.set_xlabel(r"$\varepsilon$",fontsize=12)
        ax.set_ylabel("Gap vs baseline",fontsize=11)
        ax.set_title(f"Target: {title}",fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(r"Axis 4 — Target specificity: large $\Delta W$ effect, near-zero $\Delta M$",
                 fontsize=12,fontweight='bold',y=1.01)
    fig.tight_layout()
    p=FIGS/"ising_axisB_target_specificity.png"
    fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_axis4_random_control(causal_arr):
    """Class-level control: d_c=4 structural aligned arm vs budget-matched random-site aligned arm."""
    al_m,al_lo,al_hi=[],[],[]
    rn_m,rn_lo,rn_hi=[],[],[]
    for ei in range(len(EPS_LIST)):
        m,lo,hi=mci(causal_arr[:,ei,0],seed=ei*10+1)
        al_m.append(m); al_lo.append(lo); al_hi.append(hi)
        m,lo,hi=mci(causal_arr[:,ei,4],seed=ei*10+5)
        rn_m.append(m); rn_lo.append(lo); rn_hi.append(hi)

    fig,ax=plt.subplots(figsize=(7.5,5.2))
    x=np.arange(len(EPS_LIST)); w=0.36
    ax.bar(x-w/2,al_m,w,color=C_ALIGN,alpha=0.85,label=r"Structural aligned ($d_c=4$ sites)",zorder=3)
    ax.bar(x+w/2,rn_m,w,color=C_NEUTRAL,alpha=0.85,label="Random-site aligned (budget-matched)",zorder=3)
    ax.errorbar(x-w/2,al_m,
                yerr=[[m-l for m,l in zip(al_m,al_lo)],[h-m for m,h in zip(al_m,al_hi)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.errorbar(x+w/2,rn_m,
                yerr=[[m-l for m,l in zip(rn_m,rn_lo)],[h-m for m,h in zip(rn_m,rn_hi)]],
                fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax.axhline(0,color='black',lw=0.9,ls='--',zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(['0 (null)' if e==0 else str(e) for e in EPS_LIST],fontsize=11)
    ax.set_xlabel(r"Generator bias $\varepsilon$",fontsize=12)
    ax.set_ylabel(r"Gap vs baseline $(\Delta W_{\rm base} - \Delta W_{\rm arm})$",fontsize=11)
    ax.set_title("Axis 4 — Class-level control: structural vs random-site targeting\n"
                 "Random arm uses same budget ($\\rho = E_0/L^2$) and same alignment logic",
                 fontsize=10.5)
    ax.legend(fontsize=9.5,loc='upper center',
              bbox_to_anchor=(0.5,-0.14),ncol=2,borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    p=FIGS/"ising_axis4_random_control.png"
    fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


def fig_summary(org_results, dr2_ord, dr2_dis, causal_arr, tgt_results):
    fig=plt.figure(figsize=(19,4.8))

    # Panel A: Axis 1 — organism ΔR²
    ax1=fig.add_subplot(141)
    names=[r['name'] for r in org_results]
    means=[r['dr2'] for r in org_results]
    los=[r['ci_lo'] for r in org_results]; his=[r['ci_hi'] for r in org_results]
    x=np.arange(len(names)); cols=[C_FINE if m>0 else C_COARSE for m in means]
    ax1.bar(x,means,color=cols,alpha=0.85,width=0.6,zorder=3)
    ax1.errorbar(x,means,
                 yerr=[[m-l for m,l in zip(means,los)],[h-m for m,h in zip(means,his)]],
                 fmt='none',color='#222222',capsize=3,lw=1.5,zorder=4)
    ax1.axhline(0,color='black',lw=0.8,ls='--')
    ax1.set_xticks(x); ax1.set_xticklabels(names,rotation=30,ha='right',fontsize=8)
    ax1.set_ylabel(r"$\Delta R^2$",fontsize=10)
    ax1.set_title(f"A. Organism ($T={T_ORD}$)",fontsize=10.5,fontweight='bold')

    # Panel B: Axis A — regime crossover
    ax2=fig.add_subplot(142)
    vals=[dr2_ord[0],dr2_dis[0]]; los2=[dr2_ord[1],dr2_dis[1]]; his2=[dr2_ord[2],dr2_dis[2]]
    lbls=[f"Ordered\n$T={T_ORD}$",f"Disordered\n$T={T_DIS}$"]
    cols2=[C_FINE if v>0 else C_COARSE for v in vals]
    ax2.bar([0,1],vals,color=cols2,alpha=0.85,width=0.5,zorder=3)
    ax2.errorbar([0,1],vals,
                 yerr=[[v-l for v,l in zip(vals,los2)],[h-v for v,h in zip(vals,his2)]],
                 fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax2.axhline(0,color='black',lw=0.8,ls='--')
    ax2.set_xticks([0,1]); ax2.set_xticklabels(lbls,fontsize=9.5)
    ax2.set_ylabel(r"$\Delta R^2$ ($E$ vs $\phi_{\rm var8}$)",fontsize=10)
    ax2.set_title("B. Regime crossover (Axis 2)",fontsize=10.5,fontweight='bold')
    sr=(dr2_ord[1]>0)and(dr2_dis[2]<0)
    ax2.annotate("Sign flip: "+("PASS" if sr else "FAIL"),
                 xy=(0.5,0.92),xycoords='axes fraction',ha='center',
                 fontsize=9,color='#1a7a1a' if sr else '#cc0000',fontweight='bold')

    # Panel C: Axis 2 — target ΔR² (fine-over-coarse)
    ax3=fig.add_subplot(143)
    tnames=[r['name'].replace('future_','') for r in tgt_results]
    tmeans=[r['dr2'] for r in tgt_results]
    tlos=[r['ci_lo'] for r in tgt_results]; this=[r['ci_hi'] for r in tgt_results]
    xt=np.arange(len(tnames)); tcols=[C_FINE if m>0 else C_COARSE for m in tmeans]
    ax3.bar(xt,tmeans,color=tcols,alpha=0.85,width=0.6,zorder=3)
    ax3.errorbar(xt,tmeans,
                 yerr=[[m-l for m,l in zip(tmeans,tlos)],[h-m for m,h in zip(tmeans,this)]],
                 fmt='none',color='#222222',capsize=3,lw=1.5,zorder=4)
    ax3.axhline(0,color='black',lw=0.8,ls='--')
    ax3.set_xticks(xt); ax3.set_xticklabels(tnames,rotation=30,ha='right',fontsize=8)
    ax3.set_ylabel(r"$\Delta R^2$ (fine $-$ coarse)",fontsize=10)
    ax3.set_title("C. Target (Axis 2, fine-over-coarse)",fontsize=10.5,fontweight='bold')

    # Panel D: Axis B — causal sign flip at eps=0.10
    ax4=fig.add_subplot(144)
    ei10=EPS_LIST.index(0.10)
    aw_m,aw_lo,aw_hi=mci(causal_arr[:,ei10,0],seed=301)
    mw_m,mw_lo,mw_hi=mci(causal_arr[:,ei10,1],seed=302)
    am_m,am_lo,am_hi=mci(causal_arr[:,ei10,2],seed=303)
    mm_m,mm_lo,mm_hi=mci(causal_arr[:,ei10,3],seed=304)
    vals4=[aw_m,mw_m,am_m,mm_m]
    los4=[aw_lo,mw_lo,am_lo,mm_lo]; his4=[aw_hi,mw_hi,am_hi,mm_hi]
    lbls4=["Al.\n$\\Delta W$","Mi.\n$\\Delta W$","Al.\n$\\Delta M$","Mi.\n$\\Delta M$"]
    cols4=[C_ALIGN,C_MISALN,C_NEUTRAL,C_NEUTRAL]
    ax4.bar([0,1,2,3],vals4,color=cols4,alpha=0.85,width=0.6,zorder=3)
    ax4.errorbar([0,1,2,3],vals4,
                 yerr=[[v-l for v,l in zip(vals4,los4)],[h-v for v,h in zip(vals4,his4)]],
                 fmt='none',color='#222222',capsize=4,lw=1.8,zorder=4)
    ax4.axhline(0,color='black',lw=0.8,ls='--')
    ax4.set_xticks([0,1,2,3]); ax4.set_xticklabels(lbls4,fontsize=9.5)
    ax4.set_ylabel("Gap vs baseline",fontsize=10)
    ax4.set_title(f"D. Generator causal ($\\varepsilon=0.10$, Axis 4)",fontsize=10.5,fontweight='bold')
    flip=(aw_lo>0)and(mw_hi<0)
    ax4.annotate("Sign flip: "+("PASS" if flip else "FAIL"),
                 xy=(0.5,0.92),xycoords='axes fraction',ha='center',
                 fontsize=9,color='#1a7a1a' if flip else '#cc0000',fontweight='bold')

    fig.suptitle("Ising four-axis summary: organism · crossover · target · generator causal",
                 fontsize=11,fontweight='bold')
    fig.tight_layout()
    p=FIGS/"ising_main_summary_panel.png"
    fig.savefig(p,dpi=160,bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    log(f"  fig: {p.name}")


# =============================================================================
# 11. TABLES + MACROS + JSON
# =============================================================================

def save_outputs(org_results, tgt_results, causal_arr,
                 dr2_ord, dr2_dis, r2_wb, r2_e,
                 dr2_vs_wb=0.0, lo_vs_wb=0.0, hi_vs_wb=0.0,
                 dr2_ch_ord=None, dr2_ch_dis=None, asym_at_10=float('nan'),
                 rand_gap_10=(float('nan'),float('nan'),float('nan')),
                 dr2_crit=None, T_CRIT=2.20,
                 fss_results=None, mean_E_64=float('nan'),
                 hor_results=None,
                 filt_dc0=(float('nan'),float('nan'),float('nan')),
                 filt_dcge3=(float('nan'),float('nan'),float('nan')),
                 filt_full=(float('nan'),float('nan'),float('nan')),
                 mean_pflip_dc4=float('nan'), eps_eff_max=float('nan'),
                 causal_dis_eps10_al=(float('nan'),float('nan'),float('nan')),
                 causal_dis_eps10_mi=(float('nan'),float('nan'),float('nan')),
                 causal_dis_eps10_rn=(float('nan'),float('nan'),float('nan')),
                 causal_dis_n_flip=0, causal_dis_n_eps=0,
                 crossover_results=None,
                 r2_ood=float('nan'), r2_indomain=float('nan')):

    # tables
    save_csv(pd.DataFrame(org_results), TABS/"axis1_organisms.csv")
    save_csv(pd.DataFrame(tgt_results), TABS/"axis2_targets.csv")

    rows=[]
    for ei,eps in enumerate(EPS_LIST):
        al_m,al_lo,al_hi=mci(causal_arr[:,ei,0],seed=ei*10+1)
        mi_m,mi_lo,mi_hi=mci(causal_arr[:,ei,1],seed=ei*10+2)
        dm_m,dm_lo,dm_hi=mci(causal_arr[:,ei,2],seed=ei*10+3)
        rows.append({'eps':eps,
                     'aligned_gap':al_m,'aligned_lo':al_lo,'aligned_hi':al_hi,
                     'misaligned_gap':mi_m,'misaligned_lo':mi_lo,'misaligned_hi':mi_hi,
                     'dM_aligned':dm_m,'dM_lo':dm_lo,'dM_hi':dm_hi,
                     'sign_flip':(al_lo>0)and(mi_hi<0)})
    df_c=pd.DataFrame(rows)
    save_csv(df_c, TABS/"axisB_causal.csv")

    best_org=max(org_results,key=lambda r:r['dr2'])
    best_tgt=max(tgt_results,key=lambda r:r['dr2'])
    n_flip=int(df_c[df_c['eps']>0]['sign_flip'].sum())
    n_eps_nz=int((df_c['eps']>0).sum())
    row10=df_c[df_c['eps']==0.10].iloc[0]

    vals={
        "N_PRED":N_PRED,"N_CAUSAL":N_CAUSAL,"L":L,"K":K,
        "T_ORD":T_ORD,"T_DIS":T_DIS,"EQUIL":EQUIL,
        "best_organism":best_org['name'],
        "axis1_best_dr2":best_org['dr2'],
        "axis1_ci_lo":best_org['ci_lo'],"axis1_ci_hi":best_org['ci_hi'],
        "axis_A_ord_dr2":dr2_ord[0],"axis_A_ord_lo":dr2_ord[1],"axis_A_ord_hi":dr2_ord[2],
        "axis_A_dis_dr2":dr2_dis[0],"axis_A_dis_lo":dr2_dis[1],"axis_A_dis_hi":dr2_dis[2],
        "sign_reversal":bool(dr2_ord[1]>0 and dr2_dis[2]<0),
        "best_target":best_tgt['name'],
        "axis2_best_dr2":best_tgt['dr2'],
        "axis2_ci_lo":best_tgt['ci_lo'],"axis2_ci_hi":best_tgt['ci_hi'],
        "axis2_best_r2_fine":best_tgt['r2_fine'],
        "axis2_best_r2_coarse":best_tgt['r2_coarse'],
        "axisC_r2_wb":r2_wb,"axisC_r2_e":r2_e,"axisC_gap":r2_wb-r2_e,
        "axisB_n_sign_flip":n_flip,"axisB_n_eps":n_eps_nz,
        "axisB_all_pass":n_flip==n_eps_nz,
        "axisB_aligned_eps10":float(row10['aligned_gap']),
        "axisB_aligned_lo10":float(row10['aligned_lo']),
        "axisB_aligned_hi10":float(row10['aligned_hi']),
        "axisB_misaln_eps10":float(row10['misaligned_gap']),
        "axisB_misaln_lo10":float(row10['misaligned_lo']),
        "axisB_misaln_hi10":float(row10['misaligned_hi']),
        "axisB_support":n_flip==n_eps_nz,
        "axisB_read":("earned" if n_flip==n_eps_nz
                      else "earned_narrow" if n_flip>=3
                      else "not_earned"),
    }
    save_json(vals, DATA/"paper_values.json")

    def cmd(name,val): return f"\\newcommand{{\\{name}}}{{{val}}}\n"
    lines=["% Auto-generated by ising_paper.py\n",
        cmd("IsingSamplePred",str(N_PRED)),
        cmd("IsingSampleCausal",str(N_CAUSAL)),
        cmd("IsingL",str(L)), cmd("IsingK",str(K)),
        cmd("IsingTOrd",tf(T_ORD,2)), cmd("IsingTDis",tf(T_DIS,2)),
        cmd("IsingEquil",str(EQUIL)),
        cmd("BestOrganism",{
            "E_ge3_plus_J": r"$E_{\geq 3}+J$",
            "E_plus_J":     r"$E+J$",
            "E_ge3":        r"$E_{\geq 3}$",
            "E":            r"$E$",
            "J":            r"$J$",
            "minority":     r"minority",
        }.get(best_org['name'], best_org['name'].replace("_",r"\_"))),
        cmd("AxisOneDR",tf(best_org['dr2'])),
        cmd("AxisOneDRlo",tf(best_org['ci_lo'])),
        cmd("AxisOneDRhi",tf(best_org['ci_hi'])),
        cmd("AxisAOrdDR",tf(dr2_ord[0])),
        cmd("AxisAOrdlo",tf(dr2_ord[1])),
        cmd("AxisAOrdhi",tf(dr2_ord[2])),
        cmd("AxisADisDR",tf(dr2_dis[0])),
        cmd("AxisADislo",tf(dr2_dis[1])),
        cmd("AxisADishi",tf(dr2_dis[2])),
        cmd("SignReversal","Yes" if vals['sign_reversal'] else "No"),
        cmd("BestTarget",best_tgt['name'].replace("_",r"\_")),
        cmd("AxisTwoDR",tf(best_tgt['dr2'])),
        cmd("AxisTwoRFine",tf(best_tgt['r2_fine'])),
        cmd("AxisTwoRCoarse",tf(best_tgt['r2_coarse'])),
        cmd("AxisCRWB",tf(r2_wb)), cmd("AxisCRE",tf(r2_e)),
        cmd("AxisCGap",tf(r2_wb-r2_e)),
        cmd("AxisBSignFlips",f"{n_flip}/{n_eps_nz}"),
        cmd("AxisBAlignedTen",tf(float(row10['aligned_gap']),2)),
        cmd("AxisBMisalnTen",tf(float(row10['misaligned_gap']),2)),
        cmd("AxisBSupport",vals['axisB_read'].upper().replace("_"," ")),
        cmd("AxisOneVsWB",tf(dr2_vs_wb)),
        cmd("AxisOneVsWBlo",tf(lo_vs_wb)),
        cmd("AxisOneVsWBhi",tf(hi_vs_wb)),
        cmd("AxisBAsymRatio",tf(asym_at_10,1)),
        # Random-site control (class-level, Axis 4 supplement)
        cmd("RandGapTen",    tf(float(rand_gap_10[0]),2)),
        cmd("RandGapTenLo",  tf(float(rand_gap_10[1]),2)),
        cmd("RandGapTenHi",  tf(float(rand_gap_10[2]),2)),
        cmd("MeanELSixtyFour", tf(float(mean_E_64),1)),
        # Near-Tc crossover (T=2.20)
        cmd("NearTcTemp",    tf(float(T_CRIT),2)),
        cmd("NearTcDR",      tf(float(dr2_crit[0]) if dr2_crit is not None else float('nan'))),
        cmd("NearTcDRlo",    tf(float(dr2_crit[1]) if dr2_crit is not None else float('nan'))),
        cmd("NearTcDRhi",    tf(float(dr2_crit[2]) if dr2_crit is not None else float('nan'))),
        # Finite-size (L=32, L=128)
        cmd("FSSLThirtyTwo", tf(float(fss_results[32]['dr2']) if fss_results else float('nan'))),
        cmd("FSSLThirtyTwoLo", tf(float(fss_results[32]['lo']) if fss_results else float('nan'))),
        cmd("FSSLThirtyTwoHi", tf(float(fss_results[32]['hi']) if fss_results else float('nan'))),
        cmd("FSSMeanEThirtyTwo", tf(float(fss_results[32]['mean_E']) if fss_results else float('nan'),1)),
        cmd("FSSLOneTwentyEight", tf(float(fss_results[128]['dr2']) if fss_results else float('nan'))),
        cmd("FSSLOneTwentyEightLo", tf(float(fss_results[128]['lo']) if fss_results else float('nan'))),
        cmd("FSSLOneTwentyEightHi", tf(float(fss_results[128]['hi']) if fss_results else float('nan'))),
        cmd("FSSMeanEOneTwentyEight", tf(float(fss_results[128]['mean_E']) if fss_results else float('nan'),1)),
        # Horizon sensitivity (K=10, K=50)
        cmd("HorKTenDR",    tf(float(hor_results[10]['dr2']) if hor_results else float('nan'))),
        cmd("HorKTenDRlo",  tf(float(hor_results[10]['lo'])  if hor_results else float('nan'))),
        cmd("HorKTenDRhi",  tf(float(hor_results[10]['hi'])  if hor_results else float('nan'))),
        cmd("HorKFiftyDR",  tf(float(hor_results[50]['dr2']) if hor_results else float('nan'))),
        cmd("HorKFiftyDRlo",tf(float(hor_results[50]['lo'])  if hor_results else float('nan'))),
        cmd("HorKFiftyDRhi",tf(float(hor_results[50]['hi'])  if hor_results else float('nan'))),
        # Ridge penalty
        cmd("RidgeAlpha",   tf(float(ALPHA),1)),
        # Random-arm filter decomposition (mechanism test)
        cmd("FiltDcZero",   tf(float(filt_dc0[0]),2)),
        cmd("FiltDcZeroLo", tf(float(filt_dc0[1]),2)),
        cmd("FiltDcZeroHi", tf(float(filt_dc0[2]),2)),
        cmd("FiltDcGeThree",   tf(float(filt_dcge3[0]),2)),
        cmd("FiltDcGeThreeLo", tf(float(filt_dcge3[1]),2)),
        cmd("FiltDcGeThreeHi", tf(float(filt_dcge3[2]),2)),
        cmd("FiltFull",     tf(float(filt_full[0]),2)),
        cmd("FiltFullLo",   tf(float(filt_full[1]),2)),
        cmd("FiltFullHi",   tf(float(filt_full[2]),2)),
        # eps_eff ceiling for structural arm
        cmd("MeanPflipDcFour", tf(float(mean_pflip_dc4),3)),
        cmd("EpsEffMax",       tf(float(eps_eff_max),4)),
        # Causal at T=2.50 (disordered phase)
        cmd("CausalDisAlignedTen",  tf(float(causal_dis_eps10_al[0]),2)),
        cmd("CausalDisMisalnTen",   tf(float(causal_dis_eps10_mi[0]),2)),
        cmd("CausalDisRandTen",     tf(float(causal_dis_eps10_rn[0]),2)),
        cmd("CausalDisAlignedTenLo", tf(float(causal_dis_eps10_al[1]),2)),
        cmd("CausalDisAlignedTenHi", tf(float(causal_dis_eps10_al[2]),2)),
        cmd("CausalDisMisalnTenLo",  tf(float(causal_dis_eps10_mi[1]),2)),
        cmd("CausalDisMisalnTenHi",  tf(float(causal_dis_eps10_mi[2]),2)),
        cmd("CausalDisSignFlips", f"{int(causal_dis_n_flip)}/{int(causal_dis_n_eps)}"),
        # Crossover fill-in (T=2.0 and T=2.35)
        cmd("CrossTwoZeroDR",   tf(float(crossover_results[2.00]['dr2']) if crossover_results else float('nan'))),
        cmd("CrossTwoZeroLo",   tf(float(crossover_results[2.00]['lo'])  if crossover_results else float('nan'))),
        cmd("CrossTwoZeroHi",   tf(float(crossover_results[2.00]['hi'])  if crossover_results else float('nan'))),
        cmd("CrossTwoThirtyFiveDR", tf(float(crossover_results[2.35]['dr2']) if crossover_results else float('nan'))),
        cmd("CrossTwoThirtyFiveLo", tf(float(crossover_results[2.35]['lo'])  if crossover_results else float('nan'))),
        cmd("CrossTwoThirtyFiveHi", tf(float(crossover_results[2.35]['hi'])  if crossover_results else float('nan'))),
        # Cross-lattice weight transfer (train L=64, apply at L=128)
        cmd("TransferCrossR",    tf(float(r2_ood),3)),
        cmd("TransferInSampleR", tf(float(r2_indomain),3)),
    ]
    with open(DATA/"paper_macros.tex","w") as f: f.writelines(lines)
    log("  tables + JSON + macros saved")
    return vals, df_c


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    t0=time.time()
    make_dirs()
    log("="*70)
    log(f"ising_paper.py v2  FULL_RUN={FULL_RUN}  N_PRED={N_PRED}  N_CAUSAL={N_CAUSAL}")
    log(f"L={L}  K={K}  T_ORD={T_ORD}  T_DIS={T_DIS}  EQUIL={EQUIL}")
    log("Axis 4 protocol: aligned-vs-misaligned generator + random-site control, sign-flip discriminator")
    log("="*70)

    warmup()

    # predictive battery (ordered + disordered)
    log("\n--- Predictive battery (ordered T=1.80) ---")
    out_o=run_parallel(sim_pred_world,[(SEED+10000+s,BETA_ORD) for s in range(N_PRED)],"ordered")
    log("\n--- Predictive battery (disordered T=2.50) ---")
    out_d=run_parallel(sim_pred_world,[(SEED+20000+s,BETA_DIS) for s in range(N_PRED)],"disordered")

    feats_o=[x[0] for x in out_o]; tgts_o=[x[1] for x in out_o]
    feats_d=[x[0] for x in out_d]
    dW_o=np.array([t['dW'] for t in tgts_o])
    dW_d=np.array([x[1]['dW'] for x in out_d])
    Xphi_o=np.array([f['phi8'] for f in feats_o]).reshape(-1,1)
    Xphi_d=np.array([f['phi8'] for f in feats_d]).reshape(-1,1)

    # AXIS 1 — organism comparison
    log("\n--- Axis 1: organism comparison ---")
    def design_matrix(key, feats):
        """
        Return design matrix for organism key.
        Composite organisms use TWO COLUMNS (not a summed scalar).
        E_ge3_plus_J -> [E_ge3, J]   (two independent fine features)
        E_plus_J     -> [E, J]
        All others   -> single column.
        """
        if key == 'E_ge3_plus_J':
            return np.column_stack([
                np.array([f['E_ge3'] for f in feats]),
                np.array([f['J']     for f in feats]),
            ])
        elif key == 'E_plus_J':
            return np.column_stack([
                np.array([f['E'] for f in feats]),
                np.array([f['J'] for f in feats]),
            ])
        else:
            return np.array([f[key] for f in feats]).reshape(-1, 1)

    organism_keys=['E','E_ge3','J','E_ge3_plus_J','E_plus_J','minority']
    org_results=[]
    for key in organism_keys:
        Xf=design_matrix(key, feats_o)
        dr2,lo,hi=dr2_boot(Xf,Xphi_o,dW_o,nb=N_BOOT//2,seed=abs(hash(key))%9999)
        r2v=ridge_cv(Xf,dW_o,seed=abs(hash(key))%9999+1)
        org_results.append({'name':key,'dr2':dr2,'ci_lo':lo,'ci_hi':hi,'r2':r2v})
        log(f"  {key:<22}: ΔR²={dr2:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  R²={r2v:.4f}")

    best_org=max(org_results,key=lambda r:r['dr2'])['name']
    log(f"\n  Champion organism: {best_org}")

    # AXIS A — sign reversal (E organism at both temperatures)
    log("\n--- Axis A: regime crossover (E vs phi8) ---")
    XE_o=np.array([f['E'] for f in feats_o]).reshape(-1,1)
    XE_d=np.array([f['E'] for f in feats_d]).reshape(-1,1)
    dr2_ord=dr2_boot(XE_o,Xphi_o,dW_o,nb=N_BOOT,seed=1)
    dr2_dis=dr2_boot(XE_d,Xphi_d,dW_d,nb=N_BOOT,seed=2)
    log(f"  Ordered   T={T_ORD}: ΔR²={dr2_ord[0]:+.4f}  CI=[{dr2_ord[1]:+.4f},{dr2_ord[2]:+.4f}]")
    log(f"  Disordered T={T_DIS}: ΔR²={dr2_dis[0]:+.4f}  CI=[{dr2_dis[1]:+.4f},{dr2_dis[2]:+.4f}]")
    log(f"  Sign reversal: {dr2_ord[1]>0 and dr2_dis[2]<0}")

    # Also verify crossover holds for champion E_ge3+J
    XCh_o=design_matrix(best_org, feats_o)
    XCh_d=design_matrix(best_org, feats_d)
    dr2_ch_ord=dr2_boot(XCh_o,Xphi_o,dW_o,nb=N_BOOT//2,seed=11)
    dr2_ch_dis=dr2_boot(XCh_d,Xphi_d,dW_d,nb=N_BOOT//2,seed=12)
    log(f"  Champion ({best_org}) ordered:    ΔR²={dr2_ch_ord[0]:+.4f}  CI=[{dr2_ch_ord[1]:+.4f},{dr2_ch_ord[2]:+.4f}]")
    log(f"  Champion ({best_org}) disordered: ΔR²={dr2_ch_dis[0]:+.4f}  CI=[{dr2_ch_dis[1]:+.4f},{dr2_ch_dis[2]:+.4f}]")
    cham_reversal=(dr2_ch_ord[1]>0 and dr2_ch_dis[2]<0)
    log(f"  Champion sign reversal: {cham_reversal}")

    # AXIS C — predictor/handle dissociation
    log("\n--- Axis C: predictor/handle gap ---")
    XW0=np.array([f['W0'] for f in feats_o]).reshape(-1,1)
    XBm=np.array([f['Bmag'] for f in feats_o]).reshape(-1,1)
    Xcf=np.column_stack([XW0,XBm])
    XE_c=np.array([f['E'] for f in feats_o]).reshape(-1,1)
    r2_wb=ridge_cv(Xcf,dW_o,seed=50)
    r2_e=ridge_cv(XE_c,dW_o,seed=51)
    log(f"  R²([W0,Bmag]): {r2_wb:.4f}  R²(E): {r2_e:.4f}  gap: {r2_wb-r2_e:+.4f}")

    # Compare champion against [W0,Bmag] — the meaningful coarse rival
    log("\n  Champion vs [W0,Bmag] (meaningful 2-feature coarse rival):")
    Xbest_for_ref=design_matrix(best_org, feats_o)
    dr2_vs_wb,lo_vs_wb,hi_vs_wb=dr2_boot(Xbest_for_ref,Xcf,dW_o,nb=N_BOOT,seed=9901)
    r2_wb_pt=ridge_cv(Xcf,dW_o,seed=9902)
    r2_phi_pt=ridge_cv(Xphi_o,dW_o,seed=9903)
    ref_dr2_for_fig=r2_wb_pt-r2_phi_pt   # [W0,Bmag] advantage over phi_var8
    log(f"  ΔR²({best_org} vs [W0,Bmag])={dr2_vs_wb:+.4f}  CI=[{lo_vs_wb:+.4f},{hi_vs_wb:+.4f}]")
    log(f"  R²([W0,Bmag])={r2_wb_pt:.4f}  R²(phi8)={r2_phi_pt:.4f}  ref_dr2={ref_dr2_for_fig:+.4f}")

    # Alpha sensitivity for champion
    log("\n  Ridge α sensitivity (champion organism, target dW):")
    for a_test in [0.01, 0.1, 1.0, 10.0]:
        r2a=ridge_cv(design_matrix(best_org,feats_o),dW_o,alpha=a_test,seed=42)
        log(f"  α={a_test:.2f}: R²={r2a:.4f}")

    # AXIS 2 — target comparison
    log(f"\n--- Axis 2: target comparison (ΔR² fine-over-coarse, predictor={best_org}) ---")
    log("  Criterion: ΔR² = R²(champion organism) - R²(phi_var8) on each target")
    log("  This identifies the target most specifically favoured by the fine organism,")
    log("  not merely the most predictable target overall.")
    Xbest=design_matrix(best_org, feats_o)
    tgt_keys=['dW','dW_norm','future_topshare','future_changed_sites','future_changed_frac']
    tgt_results=[]
    for key in tgt_keys:
        y=np.array([t[key] for t in tgts_o])
        # ΔR² = R²(champion fine organism) - R²(coarse phi_var8)
        dr2,lo,hi=dr2_boot(Xbest,Xphi_o,y,nb=N_BOOT//2,seed=abs(hash(key))%9999)
        r2_fine=ridge_cv(Xbest,y,seed=abs(hash(key))%9999+1)
        r2_coarse=ridge_cv(Xphi_o,y,seed=abs(hash(key))%9999+2)
        tgt_results.append({'name':key,'dr2':dr2,'ci_lo':lo,'ci_hi':hi,
                            'r2_fine':r2_fine,'r2_coarse':r2_coarse})
        log(f"  {key:<30}: ΔR²={dr2:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
            f"R²_fine={r2_fine:.4f}  R²_coarse={r2_coarse:.4f}")

    # AXIS 4 — aligned vs misaligned generator (THE CORRECT CAUSAL TEST)
    log("\n--- Axis 4: aligned vs misaligned generator + random-site control ---")
    log("  Protocol: d_c=4-targeted aligned/misaligned arms + budget-matched random arm vs baseline")
    log("  Earn condition: aligned>0 AND misaligned<0, both CIs exclude zero")
    log("  Class-level control: random-site aligned gap should be much smaller than structural gap")
    out_c=run_parallel(sim_causal_world,
                       [(SEED+90000+s,EPS_LIST) for s in range(N_CAUSAL)],"causal")
    causal_arr=np.array(out_c)  # (N_CAUSAL, n_eps, 6)

    log("\n  eps   | aligned dW (struct)     | misaligned dW           | random-aligned dW       | sign_flip")
    log("  " + "-" * 110)
    asym_at_10 = float('nan')
    rand_gap_10_m = float('nan'); rand_gap_10_lo = float('nan'); rand_gap_10_hi = float('nan')
    for ei, eps in enumerate(EPS_LIST):
        al_m,al_lo,al_hi = mci(causal_arr[:,ei,0], seed=ei*10+1)
        mi_m,mi_lo,mi_hi = mci(causal_arr[:,ei,1], seed=ei*10+2)
        dm_m,dm_lo,dm_hi = mci(causal_arr[:,ei,2], seed=ei*10+3)
        rn_m,rn_lo,rn_hi = mci(causal_arr[:,ei,4], seed=ei*10+5)
        flip = (al_lo > 0) and (mi_hi < 0)
        dW_scale = abs(al_m) if abs(al_m) > 1e-6 else 1.0
        dm_ratio = abs(dm_m) / dW_scale
        log(f"  {eps:.2f}  | {al_m:+.3f} [{al_lo:+.3f},{al_hi:+.3f}] | "
            f"{mi_m:+.3f} [{mi_lo:+.3f},{mi_hi:+.3f}] | "
            f"{rn_m:+.3f} [{rn_lo:+.3f},{rn_hi:+.3f}] | "
            f"{'PASS' if flip else 'FAIL':>9}")
        if eps > 0:
            asym = abs(mi_m)/al_m if al_m > 1e-6 else float('nan')
            log(f"    gap asymmetry |misaligned|/aligned = {asym:.2f}x  |  random/structural = {abs(rn_m)/max(abs(al_m),1e-6):.2f}x")
            if abs(eps - 0.10) < 1e-9:
                asym_at_10 = asym
                rand_gap_10_m = rn_m; rand_gap_10_lo = rn_lo; rand_gap_10_hi = rn_hi

    log("  NOTE on dM: dominant effect is on ΔW; ΔM coupling is comparatively small.")
    log(f"  Gap asymmetry at ε=0.10: {asym_at_10:.2f}x")
    log(f"  Random-site aligned gap at ε=0.10: {rand_gap_10_m:+.3f} [{rand_gap_10_lo:+.3f},{rand_gap_10_hi:+.3f}]")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT A: Near-critical temperature T=2.20
    # -------------------------------------------------------------------------
    T_CRIT = 2.20; BETA_CRIT = 1.0/T_CRIT
    N_CRIT = N_PRED; EQUIL_CRIT = 2000  # full N for tight CI, longer equilibration near Tc
    log(f"\n--- Sub-experiment: near-Tc crossover at T={T_CRIT} (N={N_CRIT}, EQUIL={EQUIL_CRIT}) ---")
    out_crit=run_parallel(sim_pred_world_param,
                          [(SEED+30000+s,BETA_CRIT,L,K,EQUIL_CRIT) for s in range(N_CRIT)],
                          "near-Tc")
    feats_crit=[x[0] for x in out_crit]; tgts_crit=[x[1] for x in out_crit]
    dW_crit=np.array([t['dW'] for t in tgts_crit])
    XE_crit=np.array([f['E'] for f in feats_crit]).reshape(-1,1)
    Xphi_crit=np.array([f['phi8'] for f in feats_crit]).reshape(-1,1)
    dr2_crit=dr2_boot(XE_crit,Xphi_crit,dW_crit,nb=N_BOOT//2,seed=301)
    r2_E_crit = ridge_cv(XE_crit,dW_crit,seed=302)
    r2_phi_crit = ridge_cv(Xphi_crit,dW_crit,seed=303)
    log(f"  T={T_CRIT}: ΔR²(E vs phi8)={dr2_crit[0]:+.4f}  CI=[{dr2_crit[1]:+.4f},{dr2_crit[2]:+.4f}]")
    log(f"  R²(E)={r2_E_crit:.4f}  R²(phi8)={r2_phi_crit:.4f}")
    log(f"  Crossover summary: T=1.80: ΔR²=+{dr2_ord[0]:.3f} | T=2.20: ΔR²={dr2_crit[0]:+.3f} | T=2.50: ΔR²={dr2_dis[0]:+.3f}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT B: Finite-size check L=32 and L=128
    # -------------------------------------------------------------------------
    N_FSS = max(500, N_PRED//6)
    log(f"\n--- Sub-experiment: finite-size check (N={N_FSS} each, T={T_ORD}) ---")
    fss_results = {}
    for L_fss, equil_fss in [(32, 500), (128, 2000)]:
        out_fss=run_parallel(sim_pred_world_param,
                             [(SEED+40000+s,BETA_ORD,L_fss,K,equil_fss) for s in range(N_FSS)],
                             f"L={L_fss}")
        feats_fss=[x[0] for x in out_fss]; tgts_fss=[x[1] for x in out_fss]
        dW_fss=np.array([t['dW'] for t in tgts_fss])
        XE_fss=np.array([f['E'] for f in feats_fss]).reshape(-1,1)
        Xphi_fss=np.array([f['phi8'] for f in feats_fss]).reshape(-1,1)
        dr2_fss=dr2_boot(XE_fss,Xphi_fss,dW_fss,nb=N_BOOT//3,seed=400+L_fss)
        mean_E_fss = float(np.mean([f['E'] for f in feats_fss]))
        fss_results[L_fss] = {'dr2': dr2_fss[0], 'lo': dr2_fss[1], 'hi': dr2_fss[2],
                               'mean_E': mean_E_fss}
        log(f"  L={L_fss}: ΔR²(E vs phi8)={dr2_fss[0]:+.4f}  CI=[{dr2_fss[1]:+.4f},{dr2_fss[2]:+.4f}]  mean_E={mean_E_fss:.1f}")
    # L=64 reference
    mean_E_64 = float(np.mean([f['E'] for f in feats_o]))
    log(f"  L=64 (main): ΔR²={dr2_ord[0]:+.4f}  mean_E={mean_E_64:.1f}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT C: Horizon sensitivity K=10 and K=50
    # -------------------------------------------------------------------------
    N_HOR = max(500, N_PRED//6)
    log(f"\n--- Sub-experiment: horizon sensitivity (N={N_HOR} each, T={T_ORD}, L={L}) ---")
    hor_results = {}
    for K_hor in [10, 50]:
        out_hor=run_parallel(sim_pred_world_param,
                             [(SEED+50000+s,BETA_ORD,L,K_hor,EQUIL) for s in range(N_HOR)],
                             f"K={K_hor}")
        feats_hor=[x[0] for x in out_hor]; tgts_hor=[x[1] for x in out_hor]
        dW_hor=np.array([t['dW'] for t in tgts_hor])
        XEg_hor=np.column_stack([np.array([f['E_ge3'] for f in feats_hor]),
                                  np.array([f['J']     for f in feats_hor])])
        Xphi_hor=np.array([f['phi8'] for f in feats_hor]).reshape(-1,1)
        dr2_hor=dr2_boot(XEg_hor,Xphi_hor,dW_hor,nb=N_BOOT//3,seed=500+K_hor)
        hor_results[K_hor] = {'dr2': dr2_hor[0], 'lo': dr2_hor[1], 'hi': dr2_hor[2]}
        log(f"  K={K_hor}: ΔR²(E_ge3+J vs phi8)={dr2_hor[0]:+.4f}  CI=[{dr2_hor[1]:+.4f},{dr2_hor[2]:+.4f}]")
    # K=20 reference from best_org champion (same statistic)
    log(f"  K=20 (main): ΔR²={max(org_results,key=lambda r:r['dr2'])['dr2']:+.4f}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT D: Filter decomposition of random-site arm (eps=0.10)
    # Tests bulk-suppression hypothesis: at T=1.80 most random sites have d_c=0,
    # so "aligned" logic SUPPRESSES their flips. If random-arm gap is dominated
    # by d_c=0 subset, that confirms the bulk-suppression mechanism.
    # -------------------------------------------------------------------------
    N_FILT = min(N_CAUSAL, 1500)
    EPS_TEST = 0.10
    log(f"\n--- Sub-experiment: random-arm decomposition by d_c (N={N_FILT}, eps={EPS_TEST}) ---")
    log("  Test: which sites drive the random-arm gap?")
    log("    d_c=0 only filter: keeps the bulk-suppression mechanism")
    log("    d_c>=3 only filter: keeps the exposed-class subset")
    out_filt = run_parallel(sim_causal_filter_world,
                            [(SEED+60000+s, EPS_TEST) for s in range(N_FILT)], "filter")
    filt_arr = np.array(out_filt)  # (N, 4): [dc0_gap, dcge3_gap, base_dW, full_random_gap]
    dc0_m, dc0_lo, dc0_hi = mci(filt_arr[:,0], seed=601)
    dcge3_m, dcge3_lo, dcge3_hi = mci(filt_arr[:,1], seed=602)
    full_m, full_lo, full_hi = mci(filt_arr[:,3], seed=603)
    log(f"  d_c=0 only:    gap = {dc0_m:+.3f} [{dc0_lo:+.3f}, {dc0_hi:+.3f}]")
    log(f"  d_c>=3 only:   gap = {dcge3_m:+.3f} [{dcge3_lo:+.3f}, {dcge3_hi:+.3f}]")
    log(f"  full random:   gap = {full_m:+.3f} [{full_lo:+.3f}, {full_hi:+.3f}]  (this run)")
    if abs(full_m) > 1e-6:
        log(f"  d_c=0 share of full-random gap: {dc0_m/full_m:.2%}")
        log(f"  d_c>=3 share of full-random gap: {dcge3_m/full_m:.2%}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT E: Structural-arm eps_eff (ceiling effect)
    # Computes mean baseline P_flip at d_c=4 sites; eps_eff = 1 - <P_flip>.
    # -------------------------------------------------------------------------
    log(f"\n--- Sub-experiment: structural-arm eps_eff (ceiling effect) ---")
    np.random.seed(SEED + 70000)
    p_flips = []
    n_subset = min(500, N_PRED)
    for s in range(n_subset):
        np.random.seed(SEED + 70000 + s); numba_seed(SEED + 70000 + s)
        spins = np.random.choice(np.array([-1,1], dtype=np.int8), size=(L,L))
        run_glauber(spins, BETA_ORD, EQUIL)
        pp = mean_pflip_dc4(spins, BETA_ORD)
        if pp > 0:
            p_flips.append(pp)
    mean_pflip = float(np.mean(p_flips)) if p_flips else float('nan')
    eps_eff_max = 1.0 - mean_pflip
    log(f"  Baseline P_flip at d_c=4 (T={T_ORD}): {mean_pflip:.4f}  (n_samples={len(p_flips)})")
    log(f"  Max effective epsilon (structural arm, ceiling): {eps_eff_max:.4f}")
    log(f"  Nominal eps=0.10 → eps_eff ~ {min(0.10, eps_eff_max):.4f} at d_c=4 sites")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT F: Causal experiment at T=2.50 (disordered phase)
    # Tests whether direction-specific causal sign-flip survives into the
    # disordered phase (where E predictively reverses sign).
    # -------------------------------------------------------------------------
    N_CAUS_DIS = min(N_CAUSAL, 1500)
    log(f"\n--- Sub-experiment: causal protocol at T={T_DIS} (N={N_CAUS_DIS}) ---")
    out_cdis = run_parallel(sim_causal_world_param,
                            [(SEED+80000+s, EPS_LIST, BETA_DIS, EQUIL)
                             for s in range(N_CAUS_DIS)], "causal-dis")
    causal_dis_arr = np.array(out_cdis)
    log("  eps   | aligned dW             | misaligned dW          | random-aligned dW       | sign_flip")
    log("  " + "-"*110)
    n_flip_dis = 0; n_eps_dis = 0
    cd_al10_m = float('nan'); cd_mi10_m = float('nan'); cd_rn10_m = float('nan')
    cd_al10_lo = cd_al10_hi = cd_mi10_lo = cd_mi10_hi = float('nan')
    cd_rn10_lo = cd_rn10_hi = float('nan')
    for ei, eps in enumerate(EPS_LIST):
        al_m, al_lo, al_hi = mci(causal_dis_arr[:,ei,0], seed=ei*10+701)
        mi_m, mi_lo, mi_hi = mci(causal_dis_arr[:,ei,1], seed=ei*10+702)
        rn_m, rn_lo, rn_hi = mci(causal_dis_arr[:,ei,4], seed=ei*10+703)
        flip = (al_lo > 0) and (mi_hi < 0)
        if eps > 0:
            n_eps_dis += 1
            if flip: n_flip_dis += 1
        log(f"  {eps:.2f}  | {al_m:+.3f} [{al_lo:+.3f},{al_hi:+.3f}] | "
            f"{mi_m:+.3f} [{mi_lo:+.3f},{mi_hi:+.3f}] | "
            f"{rn_m:+.3f} [{rn_lo:+.3f},{rn_hi:+.3f}] | "
            f"{'PASS' if flip else 'FAIL':>9}")
        if abs(eps - 0.10) < 1e-9:
            cd_al10_m, cd_al10_lo, cd_al10_hi = al_m, al_lo, al_hi
            cd_mi10_m, cd_mi10_lo, cd_mi10_hi = mi_m, mi_lo, mi_hi
            cd_rn10_m, cd_rn10_lo, cd_rn10_hi = rn_m, rn_lo, rn_hi
    log(f"  T={T_DIS} sign-flip count: {n_flip_dis}/{n_eps_dis}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT G: Crossover fill-in (T=2.0 and T=2.35)
    # -------------------------------------------------------------------------
    N_CROSS = max(1000, N_PRED//3)
    crossover_results = {}
    log(f"\n--- Sub-experiment: crossover fill-in T in [2.0, 2.35] (N={N_CROSS}) ---")
    for T_x in [2.00, 2.35]:
        beta_x = 1.0 / T_x
        equil_x = 2000 if T_x >= 2.20 else EQUIL
        out_x = run_parallel(sim_pred_world_param,
                             [(SEED+31000+int(T_x*100)*1000+s, beta_x, L, K, equil_x)
                              for s in range(N_CROSS)], f"T={T_x}")
        feats_x = [r[0] for r in out_x]; tgts_x = [r[1] for r in out_x]
        dW_x = np.array([t['dW'] for t in tgts_x])
        XE_x = np.array([f['E'] for f in feats_x]).reshape(-1,1)
        Xphi_x = np.array([f['phi8'] for f in feats_x]).reshape(-1,1)
        dr2_x = dr2_boot(XE_x, Xphi_x, dW_x, nb=N_BOOT//3, seed=int(T_x*100))
        crossover_results[T_x] = {'dr2': dr2_x[0], 'lo': dr2_x[1], 'hi': dr2_x[2]}
        log(f"  T={T_x:.2f}: ΔR²(E vs phi8)={dr2_x[0]:+.4f}  CI=[{dr2_x[1]:+.4f},{dr2_x[2]:+.4f}]")
    log(f"  Full crossover: T=1.80: {dr2_ord[0]:+.3f} | T=2.00: {crossover_results[2.00]['dr2']:+.3f} | "
        f"T=2.20: {dr2_crit[0]:+.3f} | T=2.35: {crossover_results[2.35]['dr2']:+.3f} | T=2.50: {dr2_dis[0]:+.3f}")

    # -------------------------------------------------------------------------
    # SUB-EXPERIMENT H: Out-of-distribution test (train L=64, predict L=128)
    # Uses normalized per-site descriptors and dW/(W0+1e-6) target so the
    # mapping is dimensionless and L-independent.
    # -------------------------------------------------------------------------
    log(f"\n--- Sub-experiment: OOD generalization (train L=64 → predict L=128) ---")
    # Recompute L=128 features (we already have feats_fss-like data only via fss path)
    # For OOD we need feats matching L=128; reuse the most recent fss run by re-querying.
    # Easier: rerun a small L=128 batch dedicated to OOD.
    N_OOD = 500
    out_ood = run_parallel(sim_pred_world_param,
                           [(SEED+95000+s, BETA_ORD, 128, K, 2000) for s in range(N_OOD)], "L=128 OOD")
    feats_ood = [r[0] for r in out_ood]; tgts_ood = [r[1] for r in out_ood]
    L64sq = float(L*L); L128sq = 128.0*128.0
    # Train features (L=64): normalize per-site, target = dW/(W0+1e-6)
    Xtr = np.column_stack([
        np.array([f['E_ge3'] for f in feats_o]) / L64sq,
        np.array([f['J']     for f in feats_o]) / L64sq,
    ])
    ytr = np.array([t['dW'] for t in tgts_o]) / (np.array([f['W0'] for f in feats_o]) + 1e-6)
    Xte = np.column_stack([
        np.array([f['E_ge3'] for f in feats_ood]) / L128sq,
        np.array([f['J']     for f in feats_ood]) / L128sq,
    ])
    yte = np.array([t['dW'] for t in tgts_ood]) / (np.array([f['W0'] for f in feats_ood]) + 1e-6)
    # Fit Ridge on L=64, predict on L=128
    from sklearn.linear_model import Ridge
    rg = Ridge(alpha=ALPHA, fit_intercept=True).fit(Xtr, ytr)
    yhat = rg.predict(Xte)
    r2_ood = r2_oos(yte, yhat)
    # In-domain reference (L=64 self-CV with same normalized target)
    r2_indomain = ridge_cv(Xtr, ytr, seed=901)
    log(f"  In-domain L=64 R² (normalized features, target dW/W0): {r2_indomain:.4f}")
    log(f"  OOD L=64→L=128 R²:                                     {r2_ood:.4f}")
    log(f"  Generalization retention: {r2_ood/max(r2_indomain,1e-6):.2%}")

    # figures
    log("\n--- Generating figures ---")
    np.random.seed(SEED+77)
    sp_s=np.random.choice(np.array([-1,1],dtype=np.int8),size=(L,L))
    run_glauber(sp_s,BETA_ORD,EQUIL)
    fig_schematic(sp_s)
    fig_axis1(org_results, ref_dr2_for_fig)
    fig_axis2(tgt_results,best_org)
    fig_axisB_sign_flip(causal_arr)
    fig_axisB_target_specificity(causal_arr)
    fig_axis4_random_control(causal_arr)
    fig_summary(org_results,dr2_ord,dr2_dis,causal_arr,tgt_results)

    # tables + macros
    log("\n--- Saving outputs ---")
    vals,df_c=save_outputs(
        org_results,tgt_results,causal_arr,
        dr2_ord,dr2_dis,r2_wb,r2_e,
        dr2_vs_wb=dr2_vs_wb,lo_vs_wb=lo_vs_wb,hi_vs_wb=hi_vs_wb,
        dr2_ch_ord=dr2_ch_ord,dr2_ch_dis=dr2_ch_dis,
        asym_at_10=asym_at_10,
        rand_gap_10=(rand_gap_10_m,rand_gap_10_lo,rand_gap_10_hi),
        dr2_crit=dr2_crit, T_CRIT=T_CRIT,
        fss_results=fss_results, mean_E_64=mean_E_64,
        hor_results=hor_results,
        filt_dc0=(dc0_m,dc0_lo,dc0_hi),
        filt_dcge3=(dcge3_m,dcge3_lo,dcge3_hi),
        filt_full=(full_m,full_lo,full_hi),
        mean_pflip_dc4=mean_pflip, eps_eff_max=eps_eff_max,
        causal_dis_eps10_al=(cd_al10_m,cd_al10_lo,cd_al10_hi),
        causal_dis_eps10_mi=(cd_mi10_m,cd_mi10_lo,cd_mi10_hi),
        causal_dis_eps10_rn=(cd_rn10_m,cd_rn10_lo,cd_rn10_hi),
        causal_dis_n_flip=n_flip_dis, causal_dis_n_eps=n_eps_dis,
        crossover_results=crossover_results,
        r2_ood=r2_ood, r2_indomain=r2_indomain,
    )

    # final report
    log("\n"+"="*70)
    log("FINAL REPORT")
    log("="*70)
    log(f"  Axis 1  champion:  {vals['best_organism']}  ΔR²={vals['axis1_best_dr2']:+.4f}")
    log(f"  Axis 2  crossover: T={T_ORD} ΔR²={vals['axis_A_ord_dr2']:+.4f}  "
        f"T={T_DIS} ΔR²={vals['axis_A_dis_dr2']:+.4f}  "
        f"sign_reversal={vals['sign_reversal']}")
    log(f"  Near-Tc: T={T_CRIT} ΔR²={dr2_crit[0]:+.4f}")
    log(f"  Axis 3  champion target: {vals['best_target']}  ΔR²={vals['axis2_best_dr2']:+.4f}")
    log(f"  Axis C  gap:       R²([W0,Bmag])={r2_wb:.4f}  R²(E)={r2_e:.4f}  "
        f"gap={r2_wb-r2_e:+.4f}")
    log(f"  Axis 4  sign_flips: {vals['axisB_n_sign_flip']}/{vals['axisB_n_eps']}  support={vals['axisB_support']}")
    log(f"  Random-control gap (ε=0.10): {rand_gap_10_m:+.3f} vs structural {vals['axisB_aligned_eps10']:+.3f}")
    log(f"  Finite-size:  L=32 ΔR²={fss_results[32]['dr2']:+.4f}  "
        f"L=128 ΔR²={fss_results[128]['dr2']:+.4f}  L=64 ref={dr2_ord[0]:+.4f}")
    log(f"  Horizon:  K=10 ΔR²={hor_results[10]['dr2']:+.4f}  "
        f"K=50 ΔR²={hor_results[50]['dr2']:+.4f}  K=20 ref={max(org_results,key=lambda r:r['dr2'])['dr2']:+.4f}")
    log(f"\n  All outputs: {OUT}/")
    log(f"  Runtime: {time.time()-t0:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
