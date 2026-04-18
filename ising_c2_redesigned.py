"""
ising_c2_redesigned.py
=======================
C2 Layer 2 Budget-Normalization Probe -- Ising (Redesigned)

FIXES FROM PREVIOUS ATTEMPT:
  Old: coarse arm = sites in most-deviant phi_var8 blocks (magnetization proxy)
  New: coarse arm = sites in highest-wall-density blocks (directly tied to
       the causal target dW = future wall count change)

DESIGN:
  Predictive winner in Ising (T=1.80): E-organism (exposed sites, dc=4)
  Causal target: dW = W(t+K) - W(t_0)

  Three matched-budget aligned generator arms:
    (1) E-targeted:     apply aligned bias at exposed sites (dc=4)
    (2) Random-targeted: apply aligned bias at random sites, count = n_E
    (3) Wall-density-targeted: apply aligned bias at sites in the
        highest-wall-density 8x8 blocks, count = n_E

  Aligned direction = promote wall-reducing flips (same as original Axis B).
  All arms use the same eps. Matched site count = matched perturbation mass.

  Budget metric: eps x n_sites (same eps across arms, site count matched
  to n_E per world). This is matched perturbation mass with fixed eps.

FOUR-FIELD DEFINITION:
  Structural object: Glauber flip-rate modification at eps.
  Causal location class:
    Fine: exposed sites (dc=4) -- fine structural identifier
    Coarse: sites in highest-wall-density blocks -- tied to dW target
    Random: random sites, count-matched
  Fixable background: beta, lattice, bulk Glauber, non-selected rates, dW def.
  Budget: eps x n_E (same eps, matched site count).

PRIMARY VERDICT:   delta_coarse_minus_E (coarse > E => C2 supported in Ising)
SECONDARY VERDICT: delta_E_minus_rnd    (E site-selection adds causal value?)

INTERPRETATION MATRIX:
  Case A: coarse > E (AND E >= rnd or E approx rnd)
          Coarse causal privilege survives. C2 SUPPORTED in Ising.
  Case B: E > rnd, coarse approx E
          Fine site-selection adds value vs random; coarse not clearly stronger.
          Direction-only dissociation. PARTIAL C2.
  Case C: E approx rnd, coarse approx E
          Alignment direction dominates; site-selection weak. WEAK C2.
  Case D: E > coarse AND E > rnd
          Fine predictive sites also strongest causal. Dissociation NARROWED.

Standalone -- no import from ising.py required.
Output: outputs/ising_c2_redesigned/
"""

from __future__ import annotations
import math, json, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    raise ImportError("pip install numba")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FULL_RUN    = True  # set True for Kaggle
MAX_WORKERS = 8

if FULL_RUN:
    N_CAUSAL = 2000
    EQUIL    = 1000
    N_BOOT   = 1000
else:
    N_CAUSAL = 400
    EQUIL    = 500
    N_BOOT   = 400

L        = 64
K        = 20
T_ORD    = 1.80
BETA_ORD = 1.0 / T_ORD
EPS_LIST = [0.02, 0.05, 0.10, 0.20]
BLOCK    = 8          # block size for coarse description
SEED     = 20260405

OUT = Path("outputs") / "ising_c2_redesigned"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# NUMBA ENGINE
# ---------------------------------------------------------------------------

@njit(cache=False)
def glauber_sweep(spins, beta):
    L2 = spins.shape[0]
    for parity in range(2):
        for i in range(L2):
            im1=i-1 if i>0 else L2-1; ip1=i+1 if i<L2-1 else 0
            start=parity if (i%2==0) else 1-parity
            for j in range(start, L2, 2):
                jm1=j-1 if j>0 else L2-1; jp1=j+1 if j<L2-1 else 0
                h=spins[im1,j]+spins[ip1,j]+spins[i,jm1]+spins[i,jp1]
                p=1.0/(1.0+math.exp(-2.0*beta*h))
                spins[i,j]=1 if np.random.random()<p else -1


@njit(cache=False)
def run_glauber(spins, beta, n):
    for _ in range(n): glauber_sweep(spins, beta)


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
def local_dc(spins, i, j):
    L2=spins.shape[0]; s=spins[i,j]; c=0
    if spins[i-1 if i>0 else L2-1, j]!=s: c+=1
    if spins[i+1 if i<L2-1 else 0,  j]!=s: c+=1
    if spins[i, j-1 if j>0 else L2-1]!=s: c+=1
    if spins[i, j+1 if j<L2-1 else 0]!=s: c+=1
    return c


@njit(cache=False)
def build_exposed_mask(spins):
    """E mask: exposed sites (dc=4)."""
    L2=spins.shape[0]; mask=np.zeros((L2,L2),dtype=np.int8)
    for i in range(L2):
        for j in range(L2):
            if local_dc(spins,i,j)==4: mask[i,j]=1
    return mask


@njit(cache=False)
def local_wall_density(spins, i, j, bs):
    """Count walls in the bs x bs block containing site (i,j)."""
    L2=spins.shape[0]
    bi=(i//bs)*bs; bj=(j//bs)*bs; W=0
    for di in range(bs):
        for dj in range(bs):
            r=bi+di; c=bj+dj
            rp1=(r+1)%L2; cp1=(c+1)%L2
            if spins[r,c]!=spins[rp1,c]: W+=1
            if spins[r,c]!=spins[r,cp1]: W+=1
    return W


@njit(cache=False)
def structure_aligned_sweep(spins, beta, mask, eps, aligned):
    """
    Checkerboard Glauber with aligned generator bias at masked sites.
    aligned=True:  promote wall-reducing flips (p += eps if dW<0)
    aligned=False: suppress wall-reducing flips (p -= eps if dW<0)
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
                if mask[i,j]==1 and eps>0.0:
                    dW=0
                    b=1 if spins[im1,j]!=s else 0; a=1 if spins[im1,j]!=-s else 0; dW+=a-b
                    b=1 if spins[ip1,j]!=s else 0; a=1 if spins[ip1,j]!=-s else 0; dW+=a-b
                    b=1 if spins[i,jm1]!=s else 0; a=1 if spins[i,jm1]!=-s else 0; dW+=a-b
                    b=1 if spins[i,jp1]!=s else 0; a=1 if spins[i,jp1]!=-s else 0; dW+=a-b
                    if aligned:
                        if dW<0: p+=eps
                        elif dW>0: p-=eps
                    else:
                        if dW<0: p-=eps
                        elif dW>0: p+=eps
                    if p<0.0: p=0.0
                    if p>1.0: p=1.0
                if np.random.random()<p: spins[i,j]=-s


@njit(cache=False)
def run_aligned(spins, beta, mask, eps, aligned_bool, n):
    for _ in range(n): structure_aligned_sweep(spins, beta, mask, eps, aligned_bool)


# ---------------------------------------------------------------------------
# MASK BUILDERS (numpy -- called before parallel dispatch)
# ---------------------------------------------------------------------------

def build_random_mask(L2, n_sites, rng):
    flat = np.zeros(L2*L2, dtype=np.int8)
    chosen = rng.choice(L2*L2, size=min(n_sites, L2*L2), replace=False)
    flat[chosen] = 1
    return flat.reshape(L2, L2)


def build_wall_density_mask(spins, n_sites):
    """
    Coarse arm: select n_sites from the highest-wall-density 8x8 blocks.

    Wall density per block = count of domain walls within that block.
    Directly tied to the causal target dW = future wall count change.
    Blocks with most walls are the natural intervention support for a
    coarse description that tracks wall-based structure.

    This is the correct operationalization for the redesigned probe:
    it targets the coarse region most relevant to the dW causal target,
    not a magnetization-variance proxy.
    """
    L2=spins.shape[0]; bs=BLOCK; nb=L2//bs
    block_walls = np.zeros((nb, nb), dtype=float)
    for bi in range(nb):
        for bj in range(nb):
            W = 0
            for di in range(bs):
                for dj in range(bs):
                    r=bi*bs+di; c=bj*bs+dj
                    rp1=(r+1)%L2; cp1=(c+1)%L2
                    if spins[r,c]!=spins[rp1,c]: W+=1
                    if spins[r,c]!=spins[r,cp1]: W+=1
            block_walls[bi,bj] = W

    order = np.dstack(np.unravel_index(
        np.argsort(block_walls.ravel())[::-1], (nb,nb)))[0]
    mask = np.zeros((L2,L2), dtype=np.int8)
    count = 0
    for bi,bj in order:
        for di in range(bs):
            for dj in range(bs):
                if count>=n_sites: break
                mask[bi*bs+di, bj*bs+dj]=1; count+=1
            if count>=n_sites: break
        if count>=n_sites: break
    return mask


# ---------------------------------------------------------------------------
# WARMUP
# ---------------------------------------------------------------------------

def warmup():
    np.random.seed(0)
    d=np.ones((16,16),dtype=np.int8)
    run_glauber(d,0.5,1); wall_count(d); local_dc(d,0,0)
    m=build_exposed_mask(d)
    run_aligned(d,0.5,m,0.05,True,1)
    print("warmup OK")


# ---------------------------------------------------------------------------
# PER-WORLD SIMULATION
# ---------------------------------------------------------------------------

def sim_world(args):
    """
    Returns array of shape (n_eps, 3):
      col 0: E-targeted aligned gap     (baseline_dW - E_arm_dW; want > 0)
      col 1: random-targeted aligned gap
      col 2: wall-density-targeted aligned gap

    gap = baseline_dW - arm_dW
    Positive = arm achieved less wall growth (or more wall reduction) than baseline.
    Primary interest: which arm produces the largest positive gap?
    """
    seed, eps_list = args
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2**31))

    spins = np.random.choice(np.array([-1,1],dtype=np.int8), size=(L,L))
    run_glauber(spins, BETA_ORD, EQUIL)
    sp0   = spins.copy()
    W0    = wall_count(sp0)

    # build masks
    e_mask   = build_exposed_mask(sp0)
    n_E      = int(e_mask.sum())
    rnd_mask = build_random_mask(L, n_E, rng)
    wd_mask  = build_wall_density_mask(sp0, n_E)

    # baseline: pure Glauber
    base = sp0.copy(); run_glauber(base, BETA_ORD, K)
    base_dW = wall_count(base) - W0

    results = np.zeros((len(eps_list), 3), dtype=np.float64)
    for ei, eps in enumerate(eps_list):
        al_e  = sp0.copy(); run_aligned(al_e,  BETA_ORD, e_mask,   float(eps), True, K)
        al_r  = sp0.copy(); run_aligned(al_r,  BETA_ORD, rnd_mask, float(eps), True, K)
        al_wd = sp0.copy(); run_aligned(al_wd, BETA_ORD, wd_mask,  float(eps), True, K)

        results[ei,0] = base_dW - (wall_count(al_e)  - W0)
        results[ei,1] = base_dW - (wall_count(al_r)  - W0)
        results[ei,2] = base_dW - (wall_count(al_wd) - W0)

    return results, n_E


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------

def mci(vals, nb=400, seed=0):
    v=np.array(vals,float); rng=np.random.default_rng(seed)
    b=np.array([np.mean(v[rng.integers(0,len(v),len(v))]) for _ in range(nb)])
    return float(np.mean(v)),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))


def delta_ci(a, b_arr, nb=400, seed=0):
    a=np.array(a,float); b_arr=np.array(b_arr,float)
    rng=np.random.default_rng(seed); n=len(a)
    diffs=np.array([
        np.mean(a[rng.integers(0,n,n)])-np.mean(b_arr[rng.integers(0,n,n)])
        for _ in range(nb)
    ])
    return float(np.mean(a-b_arr)),float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))


def classify(wd_beats_e, e_beats_rnd):
    if wd_beats_e:
        return "A", "Wall-density coarse arm beats E. C2 SUPPORTED in Ising."
    if e_beats_rnd:
        return "B", ("E site-selection adds causal value vs random; "
                     "wall-density not clearly stronger. PARTIAL C2.")
    return "C/D", ("Alignment direction dominates; site-selection weak. "
                   "Dissociation weak or absent at this budget.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0=time.time()
    print("Ising C2 Budget-Normalization Probe (Redesigned)")
    print(f"T={T_ORD}  L={L}  K={K}  N={N_CAUSAL}  EQUIL={EQUIL}  FULL={FULL_RUN}")
    print(f"EPS_LIST={EPS_LIST}")
    print()
    print("Coarse arm: highest-wall-density 8x8 blocks (redesigned from phi_var8 proxy)")
    print("PRIMARY:   delta_walldens_minus_E (coarse > E => C2 supported)")
    print("SECONDARY: delta_E_minus_rnd      (E adds causal value over random?)")
    print()

    warmup()

    jobs = [(SEED+s, EPS_LIST) for s in range(N_CAUSAL)]
    results = [None]*N_CAUSAL; n_E_vals = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fmap={ex.submit(sim_world,j):i for i,j in enumerate(jobs)}
        done=0; step=max(1,N_CAUSAL//8)
        for f in as_completed(fmap):
            res,nE=f.result(); results[fmap[f]]=res; n_E_vals.append(nE)
            done+=1
            if done%step==0 or done==N_CAUSAL:
                print(f"  {done}/{N_CAUSAL}", flush=True)

    arr = np.stack(results)  # (N_CAUSAL, n_eps, 3)

    print()
    print("="*90)
    print(f"{'eps':>5}  "
          f"{'E-gap':>7} {'[lo,hi]':>14}  "
          f"{'Rnd-gap':>7} {'[lo,hi]':>14}  "
          f"{'WD-gap':>7} {'[lo,hi]':>14}  "
          f"{'WD>E?':>6}  {'E>Rnd?':>7}")
    print("="*90)

    rows=[]; wd_beats_e_any=False; e_beats_rnd_any=False

    for ei,eps in enumerate(EPS_LIST):
        ge=arr[:,ei,0]; gr=arr[:,ei,1]; gw=arr[:,ei,2]

        em,elo,ehi=mci(ge,N_BOOT,seed=ei*10+1)
        rm,rlo,rhi=mci(gr,N_BOOT,seed=ei*10+2)
        wm,wlo,whi=mci(gw,N_BOOT,seed=ei*10+3)

        d_we,d_we_lo,d_we_hi=delta_ci(gw,ge,N_BOOT,seed=ei*10+4)  # WD - E
        d_er,d_er_lo,d_er_hi=delta_ci(ge,gr,N_BOOT,seed=ei*10+5)  # E - rnd

        wd_beats_e=d_we_lo>0; e_beats_rnd=d_er_lo>0
        if wd_beats_e: wd_beats_e_any=True
        if e_beats_rnd: e_beats_rnd_any=True

        print(f"{eps:>5.2f}  "
              f"{em:>+7.4f} [{elo:+.3f},{ehi:+.3f}]  "
              f"{rm:>+7.4f} [{rlo:+.3f},{rhi:+.3f}]  "
              f"{wm:>+7.4f} [{wlo:+.3f},{whi:+.3f}]  "
              f"{'YES' if wd_beats_e else 'no':>6}  "
              f"{'YES' if e_beats_rnd else 'no':>7}")

        rows.append({
            "eps":eps,
            "E_gap":em,"E_lo":elo,"E_hi":ehi,
            "rnd_gap":rm,"rnd_lo":rlo,"rnd_hi":rhi,
            "wd_gap":wm,"wd_lo":wlo,"wd_hi":whi,
            "delta_wd_minus_E":d_we,"d_we_lo":d_we_lo,"d_we_hi":d_we_hi,
            "delta_E_minus_rnd":d_er,"d_er_lo":d_er_lo,"d_er_hi":d_er_hi,
            "wd_beats_E_CI":wd_beats_e,"E_beats_rnd_CI":e_beats_rnd,
        })

    print("="*90)

    print()
    print("PRIMARY VERDICT (wall-density vs E) -- per eps:")
    for r in rows:
        tag="BEATS E" if r["wd_beats_E_CI"] else "no sig."
        print(f"  eps={r['eps']:.2f}: delta_wd_minus_E={r['delta_wd_minus_E']:+.4f} "
              f"CI=[{r['d_we_lo']:+.4f},{r['d_we_hi']:+.4f}]  {tag}")
    print(f"  WD beats E at any eps: {wd_beats_e_any}")

    print()
    print("SECONDARY VERDICT (E vs random) -- per eps:")
    for r in rows:
        tag="E BEATS RND" if r["E_beats_rnd_CI"] else "no sig."
        print(f"  eps={r['eps']:.2f}: delta_E_minus_rnd={r['delta_E_minus_rnd']:+.4f} "
              f"CI=[{r['d_er_lo']:+.4f},{r['d_er_hi']:+.4f}]  {tag}")
    print(f"  E beats rnd at any eps: {e_beats_rnd_any}")

    case,reading=classify(wd_beats_e_any,e_beats_rnd_any)
    print()
    print(f"CASE: {case}")
    print(f"C2 READING: {reading}")

    print()
    print(f"Mean n_E per world: {np.mean(n_E_vals):.1f}  "
          f"(budget = {np.mean(n_E_vals):.0f} sites x eps per arm)")
    print()
    print("COARSE ARM NOTE:")
    print("  Wall-density blocks = 8x8 blocks ranked by local domain wall count.")
    print("  Directly tied to causal target dW. Not a phi_var8 magnetization proxy.")
    print("  Natural but not uniquely forced operationalization of coarse description.")

    df=pd.DataFrame(rows)
    df.to_csv(OUT/"ising_c2_redesigned_summary.csv",index=False)
    with open(OUT/"ising_c2_redesigned_stats.json","w") as f:
        json.dump({
            "config":{"N_CAUSAL":N_CAUSAL,"L":L,"K":K,"T_ORD":T_ORD,
                      "EQUIL":EQUIL,"EPS_LIST":EPS_LIST,"FULL_RUN":FULL_RUN,
                      "BLOCK":BLOCK},
            "wd_beats_E_any_eps":wd_beats_e_any,
            "E_beats_rnd_any_eps":e_beats_rnd_any,
            "case":case,"interpretation":reading,
            "mean_n_E":float(np.mean(n_E_vals)),
            "rows":rows,
        },f,indent=2)

    print(f"\nOutputs: {OUT}/")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__=="__main__":
    main()
