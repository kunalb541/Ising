"""
tests.py — Regression tests for ising.py core functions.

Tests:
  1. Glauber acceptance probability (detailed balance)
  2. Domain-wall counting on known configurations
  3. Exposure degree d_c computation
  4. Junction classification
  5. Causal direction: aligned arm reduces more walls than misaligned
  6. Equilibration check: |m| distribution above T_c

Run with: python tests.py
"""

import math
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Import numba-compiled functions from ising.py.
# We trigger warmup before tests so JIT compilation is done first.
# ---------------------------------------------------------------------------

def import_ising():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "ising", pathlib.Path(__file__).parent / "ising.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Importing ising.py and warming up JIT ...", flush=True)
ising = import_ising()
np.random.seed(0)
ising.warmup()
print()

PASS = []
FAIL = []

def ok(name):
    PASS.append(name)
    print(f"  PASS  {name}")

def fail(name, msg):
    FAIL.append(name)
    print(f"  FAIL  {name}: {msg}")


# ---------------------------------------------------------------------------
# 1. Glauber acceptance probability — detailed balance check
#    For heat-bath rule: P(new=+1) = 1/(1+exp(-2*beta*h))
#    Flip probability when current=+1 equals 1/(1+exp(2*beta*h))
# ---------------------------------------------------------------------------

def test_glauber_acceptance():
    """
    Verify the Glauber flip probability satisfies the analytic formula.
    Run a large spin at site (i,j) many times with fixed neighbours and
    count flips to estimate p_flip, compare to analytic value.
    """
    beta = 1.0 / 1.80
    # All 4 neighbours set to -1; central spin = +1 → h = -4, dc = 4
    # Analytic p_flip for spin+1: 1/(1+exp(2*beta*h)) = 1/(1+exp(-8*beta))
    h = -4
    p_flip_analytic = 1.0 / (1.0 + math.exp(2.0 * beta * h))

    n_trials = 50_000
    L2 = 8
    # Build grid: central site (3,3) = +1, all neighbours = -1, rest don't matter
    flips = 0
    for _ in range(n_trials):
        spins = np.ones((L2, L2), dtype=np.int8)
        spins[3, 2] = -1; spins[3, 4] = -1
        spins[2, 3] = -1; spins[4, 3] = -1
        spins[3, 3] = 1
        old = int(spins[3, 3])
        ising.glauber_sweep(spins, beta)
        if int(spins[3, 3]) != old:
            flips += 1

    p_flip_empirical = flips / n_trials
    err = abs(p_flip_empirical - p_flip_analytic)
    # Allow 2% absolute tolerance for Monte Carlo noise
    if err < 0.02:
        ok("glauber_acceptance_probability")
    else:
        fail("glauber_acceptance_probability",
             f"analytic={p_flip_analytic:.4f}, empirical={p_flip_empirical:.4f}, err={err:.4f}")


# ---------------------------------------------------------------------------
# 2. Domain-wall counting on known configurations
# ---------------------------------------------------------------------------

def test_wall_count_all_aligned():
    """All spins +1 → zero domain walls."""
    L2 = 16
    spins = np.ones((L2, L2), dtype=np.int8)
    W = int(ising.wall_count(spins))
    if W == 0:
        ok("wall_count_all_aligned")
    else:
        fail("wall_count_all_aligned", f"expected 0, got {W}")


def test_wall_count_single_defect():
    """
    One isolated spin -1 in a sea of +1 on an LxL periodic grid.
    Contributes exactly 4 wall bonds (one with each neighbour).
    """
    L2 = 16
    spins = np.ones((L2, L2), dtype=np.int8)
    spins[8, 8] = -1
    W = int(ising.wall_count(spins))
    if W == 4:
        ok("wall_count_single_defect")
    else:
        fail("wall_count_single_defect", f"expected 4, got {W}")


def test_wall_count_checkerboard():
    """
    Checkerboard pattern (all neighbouring pairs differ).
    Each site has 4 neighbours → total wall bonds = 2*L*L (each bond counted once).
    """
    L2 = 8
    spins = np.fromfunction(
        lambda i, j: np.where((i + j) % 2 == 0, 1, -1), (L2, L2), dtype=int
    ).astype(np.int8)
    W = int(ising.wall_count(spins))
    expected = 2 * L2 * L2  # 2*L^2 bonds in periodic checkerboard
    if W == expected:
        ok("wall_count_checkerboard")
    else:
        fail("wall_count_checkerboard", f"expected {expected}, got {W}")


# ---------------------------------------------------------------------------
# 3. Exposure degree d_c computation
# ---------------------------------------------------------------------------

def test_dc_isolated_spin():
    """Isolated minority spin surrounded by 4 unlike neighbours: d_c = 4."""
    L2 = 8
    spins = np.ones((L2, L2), dtype=np.int8)
    spins[4, 4] = -1
    dc = ising.local_dc(spins, 4, 4)
    if dc == 4:
        ok("dc_isolated_spin")
    else:
        fail("dc_isolated_spin", f"expected 4, got {dc}")


def test_dc_bulk_spin():
    """Spin deep in aligned bulk: d_c = 0."""
    L2 = 8
    spins = np.ones((L2, L2), dtype=np.int8)
    dc = ising.local_dc(spins, 4, 4)
    if dc == 0:
        ok("dc_bulk_spin")
    else:
        fail("dc_bulk_spin", f"expected 0, got {dc}")


def test_count_E_single_defect():
    """Single isolated minority spin → E = 1."""
    L2 = 16
    spins = np.ones((L2, L2), dtype=np.int8)
    spins[8, 8] = -1
    E = int(ising.count_E(spins))
    if E == 1:
        ok("count_E_single_defect")
    else:
        fail("count_E_single_defect", f"expected 1, got {E}")


def test_count_E_ge3_wall_segment():
    """
    A horizontal wall segment site with 3 unlike neighbours (dc=3) should be
    counted by count_E_ge3 but not count_E.
    Build: row of -1 in +1 background; corner site has dc=3.
    """
    L2 = 16
    spins = np.ones((L2, L2), dtype=np.int8)
    # Place a 3-site horizontal stripe of -1 at row 8, cols 4..6
    # Site (8,4): left is +1 (dc contribution), right is -1, up/down +1 → dc=3
    spins[8, 4] = -1; spins[8, 5] = -1; spins[8, 6] = -1
    # Count at site (8,4): neighbours: (7,4)=+1, (9,4)=+1, (8,3)=+1, (8,5)=-1
    # dc = 3 (three unlike neighbours)
    dc = ising.local_dc(spins, 8, 4)
    counted_ge3 = int(ising.count_E_ge3(spins))
    counted_E = int(ising.count_E(spins))  # dc=4 only
    if dc == 3 and counted_ge3 >= 2 and counted_E == 0:
        ok("count_E_ge3_wall_segment")
    else:
        fail("count_E_ge3_wall_segment",
             f"dc={dc}, count_E_ge3={counted_ge3}, count_E={counted_E}")


# ---------------------------------------------------------------------------
# 4. Junction classification
# ---------------------------------------------------------------------------

def test_junction_corner_vs_collinear():
    """
    A corner wall bend (d_c=2, non-collinear) should be counted by count_J.
    A collinear d_c=2 site (two unlike neighbours on opposite sides) should not.
    """
    L2 = 10
    # --- corner site ---
    spins_corner = np.ones((L2, L2), dtype=np.int8)
    # Place an L-shaped domain wall: -1 region in top-left corner
    # Site (5,5): unlike neighbours at (4,5)=-1 and (5,4)=-1 → dc=2, non-collinear
    spins_corner[4, 4] = -1; spins_corner[4, 5] = -1
    spins_corner[5, 4] = -1
    # site (5,5): up=(4,5)=-1, left=(5,4)=-1, down=(6,5)=+1, right=(5,6)=+1
    # dc=2, non-collinear (up + left, not up+down or left+right)
    dc_corner = ising.local_dc(spins_corner, 5, 5)

    # --- collinear site ---
    spins_collinear = np.ones((L2, L2), dtype=np.int8)
    # Two unlike neighbours on opposite sides (up and down)
    spins_collinear[4, 5] = -1; spins_collinear[6, 5] = -1
    # site (5,5): up=-1, down=-1, left=+1, right=+1 → dc=2, collinear
    dc_col = ising.local_dc(spins_collinear, 5, 5)

    J_corner = int(ising.count_J(spins_corner))
    J_collinear = int(ising.count_J(spins_collinear))

    if dc_corner == 2 and dc_col == 2 and J_corner >= 1 and J_collinear == 0:
        ok("junction_corner_vs_collinear")
    else:
        fail("junction_corner_vs_collinear",
             f"dc_corner={dc_corner}, dc_col={dc_col}, "
             f"J_corner={J_corner}, J_collinear={J_collinear}")


# ---------------------------------------------------------------------------
# 5. Causal direction test
#    Aligned arm should produce a positive gap (more wall reduction);
#    misaligned should produce a negative gap (less wall reduction).
#    Run N_CAUSAL=200 worlds at eps=0.10 and check the mean gap signs.
# ---------------------------------------------------------------------------

def test_causal_direction():
    """
    Run a small causal experiment (N=500 worlds, eps=0.05 and 0.20).
    Verify: mean aligned gap > 0, mean misaligned gap < 0.
    The sign-flip discriminator should hold on average.
    NOTE: At T=1.80 exposed sites flip with p≈0.988 (ceiling effect),
    so the aligned gap is smaller in magnitude than the misaligned gap.
    We use eps=0.05 and eps=0.20 together (averaged) for a more robust signal,
    and N=500 to reduce Monte Carlo noise on the aligned direction.
    """
    N = 500
    eps_test = [0.05, 0.20]
    jobs = [(42 + s, eps_test) for s in range(N)]

    results = []
    for job in jobs:
        res = ising.sim_causal_world(job)
        results.append(res)

    arr = np.array(results)  # (N, 2, 4)
    # Average over the two eps values for robustness
    al_gap = (arr[:, 0, 0] + arr[:, 1, 0]) / 2.0  # aligned gap on dW
    mi_gap = (arr[:, 0, 1] + arr[:, 1, 1]) / 2.0  # misaligned gap on dW

    mean_al = float(np.mean(al_gap))
    mean_mi = float(np.mean(mi_gap))

    if mean_al > 0 and mean_mi < 0:
        ok(f"causal_direction (mean_aligned={mean_al:+.3f}, mean_misaligned={mean_mi:+.3f})")
    else:
        fail("causal_direction",
             f"mean_aligned={mean_al:+.3f} (want >0), "
             f"mean_misaligned={mean_mi:+.3f} (want <0)")


# ---------------------------------------------------------------------------
# 6. Equilibration check: |m| at T > T_c should be near zero
# ---------------------------------------------------------------------------

def test_equilibration_disordered():
    """
    At T=2.50 > T_c=2.269, after equilibration from random state,
    the absolute magnetisation should be small (|m| << 1).
    Run 50 worlds, check mean |m| < 0.15.
    """
    beta_dis = 1.0 / 2.50
    mag_vals = []
    for seed in range(50):
        np.random.seed(seed + 10000)
        spins = np.random.choice(
            np.array([-1, 1], dtype=np.int8), size=(ising.L, ising.L)
        )
        ising.run_glauber(spins, beta_dis, ising.EQUIL)
        mag_vals.append(ising.abs_mag(spins))

    mean_mag = float(np.mean(mag_vals))
    if mean_mag < 0.15:
        ok(f"equilibration_disordered (mean|m|={mean_mag:.3f} < 0.15)")
    else:
        fail("equilibration_disordered",
             f"mean|m|={mean_mag:.3f} at T=2.50, expected < 0.15")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Ising regression tests")
    print("=" * 60)
    print()

    test_glauber_acceptance()
    test_wall_count_all_aligned()
    test_wall_count_single_defect()
    test_wall_count_checkerboard()
    test_dc_isolated_spin()
    test_dc_bulk_spin()
    test_count_E_single_defect()
    test_count_E_ge3_wall_segment()
    test_junction_corner_vs_collinear()

    print()
    print("Running causal direction test (N=200 worlds) ...")
    test_causal_direction()

    print()
    print("Running equilibration check (50 worlds, T=2.50) ...")
    test_equilibration_disordered()

    print()
    print("=" * 60)
    n_pass = len(PASS)
    n_fail = len(FAIL)
    print(f"Results: {n_pass} passed, {n_fail} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
