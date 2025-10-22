# -*- coding: utf-8 -*-
"""
Task 2 — Stochastic-Demand VRP via GA (single depot, capacity=200)

Key ideas
---------
- Genome: a permutation of all customers (the visiting order).
- Evaluation under uncertainty: for each permutation, simulate S scenarios.
  In each scenario, demands are sampled as positive integers from a Normal(mean, 0.2*mean).
  While traversing the order, we REFILL BEFORE OVERFLOW (open a new trip whenever the next
  sampled demand wouldn't fit). This guarantees capacity is never exceeded.
- Fitness: Monte-Carlo estimate of expected total distance (lower is better).
- Comparison to deterministic Task 1:
    * Build the Task1 best route (by GA on mean demands, same objective as Task1).
    * Compute deterministic distance on means.
    * Estimate its stochastic "static feasibility rate": fraction of scenarios where those
      fixed mean-based segments remain feasible when demands are resampled (sum ≤ 200).
- Output: best expected distance (mean±std on a large eval set), feasibility comparison,
  and a PNG figure showing one sampled realization of the best stochastic route.

Usage
-----
python task2_stochastic_vrp.py --csv VRP.csv --out stochastic_best.png \
    --seed 42 --gens 1600 --pop 240 --train-sim 64 --eval-sim 512 --verbose-every 100
"""

from pathlib import Path
import argparse, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- data & distances -----------------------------
def load_data(csv_path: Path):
    t = pd.read_csv(csv_path)
    is_depot = t["CUST_OR_DEPOT"].str.upper() == "DEPOT"
    depots = t[is_depot].reset_index(drop=True)
    custs  = t[~is_depot].reset_index(drop=True)

    # choose depot NO==0 if present, else the first depot
    if (depots["NO"] == 0).any():
        d = depots.loc[depots["NO"] == 0].iloc[0]
    else:
        d = depots.iloc[0]

    depot_xy = np.array([float(d["XCOORD"]), float(d["YCOORD"])], float)
    Cxy  = custs[["XCOORD","YCOORD"]].to_numpy(float)
    Cno  = custs["NO"].to_numpy(int)
    Cdem = custs["DEMAND"].to_numpy(float)

    n = len(Cno)
    Dcc = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(Cxy[i]-Cxy[j]))
            Dcc[i,j] = Dcc[j,i] = d
    Ddc = np.array([float(np.linalg.norm(Cxy[i]-depot_xy)) for i in range(n)], float)
    return depot_xy, Cxy, Cno, Cdem, Dcc, Ddc


# ----------------------------- sampling & sim -------------------------------
def sample_positive_int_norm(means: np.ndarray, rng: np.random.Generator):
    """Truncated-to-positive integers: round(N(mean, 0.2*mean)) and clamp to >=1."""
    std = 0.2 * means
    z = rng.normal(loc=means, scale=std)
    z = np.maximum(1.0, np.round(z))
    return z.astype(int)

def simulate_distance_for_perm(
    perm, means, cap, Dcc, Ddc, rng: np.random.Generator, scenarios: int
):
    """
    Monte-Carlo average total distance following a refill-before-overflow policy.
    Returns mean, std of total distance across scenarios, and a representative
    segmentation from the LAST scenario (for plotting if wanted).
    """
    n = len(perm)
    totals = np.empty(scenarios, float)
    last_segs = None

    for s in range(scenarios):
        q = sample_positive_int_norm(means, rng)
        total, explore, refill, segs = decode_cost_stochastic(perm, q, cap, Dcc, Ddc)
        totals[s] = total
        if s == scenarios - 1:
            last_segs = segs

    return float(totals.mean()), float(totals.std(ddof=1) if scenarios > 1 else 0.0), last_segs

def decode_cost_stochastic(perm, demand_int, cap, Dcc, Ddc):
    """
    With sampled integer demands, traverse the permutation and open a new trip
    whenever the next customer wouldn't fit the remaining capacity.
    Returns total, explore, refill, segments.
    """
    explore = 0.0; refill = 0.0
    segs = []
    cur, load = [], 0

    for idx in perm:
        d = int(demand_int[idx])
        if cur and load + d > cap:
            # close current
            refill += Ddc[cur[0]] + Ddc[cur[-1]]
            for a,b in zip(cur[:-1], cur[1:]):
                explore += Dcc[a,b]
            segs.append(cur)
            # start new
            cur = [idx]; load = d
        else:
            cur.append(idx); load += d

    if cur:
        refill += Ddc[cur[0]] + Ddc[cur[-1]]
        for a,b in zip(cur[:-1], cur[1:]):
            explore += Dcc[a,b]
        segs.append(cur)

    return explore + refill, explore, refill, segs


# ----------------------------- GA (stochastic objective) --------------------
def tournament_select(pop, fit, k=3):
    cand = random.sample(range(len(pop)), k)
    cand.sort(key=lambda i: fit[i], reverse=True)
    return pop[cand[0]].copy()

def ox_crossover(p1, p2):
    n = len(p1); l, r = sorted(random.sample(range(n), 2))
    c = [-1]*n; c[l:r+1] = p1[l:r+1]
    fill = [x for x in p2 if x not in c]
    j = 0
    for i in range(n):
        if c[i] == -1:
            c[i] = fill[j]; j += 1
    return c

def swap_mutation(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def two_opt_on_perm(ind, Dcc):
    out = ind[:]; n = len(out); improved = True
    while improved:
        improved = False
        for i in range(n-3):
            a, b = out[i], out[i+1]
            for k in range(i+2, n-1):
                c, d = out[k], out[k+1]
                if Dcc[a,c] + Dcc[b,d] + 1e-12 < Dcc[a,b] + Dcc[c,d]:
                    out[i+1:k+1] = reversed(out[i+1:k+1]); improved = True
    return out


# ----------------------------- Deterministic baseline (Task 1) --------------
def deterministic_cost_and_segments(perm, means, cap, Dcc, Ddc):
    return decode_cost_stochastic(perm, means.astype(int), cap, Dcc, Ddc)

def feasibility_rate_of_fixed_segments(segs, means, cap, rng, trials: int):
    """
    Given fixed segments (from the deterministic solution),
    estimate the probability that sum of sampled demands per segment ≤ cap.
    """
    m = means.copy().astype(float)
    ok = 0
    for _ in range(trials):
        q = sample_positive_int_norm(m, rng)
        feasible = True
        for seg in segs:
            if int(q[seg].sum()) > cap:
                feasible = False; break
        ok += int(feasible)
    return ok / trials


# ----------------------------- Solve Task 2 ---------------------------------
def solve(csv: Path, out_png: Path,
          seed=42, cap=200.0,
          pop=240, gens=1600, pc=0.95, pm=0.25, elite=6, tour_k=3,
          twoopt_every=20, train_sim=64, eval_sim=512, verbose_every=100):

    # Load data
    random.seed(seed); np.random.seed(seed)
    depot_xy, Cxy, Cno, Cdem, Dcc, Ddc = load_data(csv)
    n = len(Cno)

    # --- Helper: NN seed
    def nn_seed():
        unused = set(range(n))
        cur = random.choice(tuple(unused)); unused.remove(cur)
        route = [cur]
        while unused:
            nxt = min(unused, key=lambda j: Dcc[cur,j])
            route.append(nxt); unused.remove(nxt); cur = nxt
        return route

    # ================== 1) Deterministic baseline (Task 1 style) ==================
    # We optimize on mean demands only (no sampling) to get a fair baseline route.
    pop_det = [nn_seed()] + [random.sample(range(n), n) for _ in range(219)]
    best_det, best_det_cost = None, float("inf")
    for _ in range(1200):
        fits = []
        for ind in pop_det:
            c, _, _, _ = deterministic_cost_and_segments(ind, Cdem, cap, Dcc, Ddc)
            fits.append(-c)
            if c < best_det_cost:
                best_det, best_det_cost = ind[:], c
        elite_idx = np.argsort(fits)[-4:][::-1]
        new_pop = [pop_det[i][:] for i in elite_idx]
        while len(new_pop) < len(pop_det):
            p1 = tournament_select(pop_det, fits, tour_k)
            p2 = tournament_select(pop_det, fits, tour_k)
            child = ox_crossover(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm: child = swap_mutation(child)
            new_pop.append(child)
        pop_det = new_pop

    det_total, det_exp, det_ref, det_segs = deterministic_cost_and_segments(best_det, Cdem, cap, Dcc, Ddc)

    # Estimate feasibility rate of these fixed segments under stochastic demand
    rng_eval = np.random.default_rng(seed + 777)
    det_feas_rate = feasibility_rate_of_fixed_segments(det_segs, Cdem, cap, rng_eval, trials=eval_sim)

    # ================== 2) Stochastic GA (optimize expected distance) =============
    # Single run, continuous from Gen 1 → gens
    popu = [nn_seed()] + [random.sample(range(n), n) for _ in range(pop-1)]
    best, best_mean, best_std, best_segs_last = None, float("inf"), 0.0, None

    for gen in range(1, gens+1):
        # Fix a RNG per generation to reduce noise across individuals
        rng_gen = np.random.default_rng(seed + gen)

        fits = []
        for ind in popu:
            mean_cost, _, _ = simulate_distance_for_perm(
                ind, Cdem, cap, Dcc, Ddc, rng_gen, scenarios=train_sim
            )
            fits.append(-mean_cost)
            if mean_cost < best_mean:
                best, best_mean = ind[:], mean_cost

        # Optional 2-opt on the current best (helps convergence)
        if twoopt_every and gen % twoopt_every == 0:
            improved = two_opt_on_perm(best, Dcc)
            tmp_mean, _, _ = simulate_distance_for_perm(
                improved, Cdem, cap, Dcc, Ddc, rng_gen, scenarios=train_sim
            )
            if tmp_mean < best_mean:
                best, best_mean = improved, tmp_mean

        # Elitism + offspring
        elite_idx = np.argsort(fits)[-elite:][::-1]
        new_pop = [popu[i][:] for i in elite_idx]
        while len(new_pop) < pop:
            p1 = tournament_select(popu, fits, tour_k)
            p2 = tournament_select(popu, fits, tour_k)
            child = ox_crossover(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm: child = swap_mutation(child)
            new_pop.append(child)
        popu = new_pop

        if verbose_every and gen % verbose_every == 0:
            print(f"Gen {gen:4d} | best(mean over {train_sim}) = {best_mean:.2f}")

    # High-precision evaluation of best stochastic route
    rng_final = np.random.default_rng(seed + 999)
    best_mean, best_std, best_segs_last = simulate_distance_for_perm(
        best, Cdem, cap, Dcc, Ddc, rng_final, scenarios=eval_sim
    )

    # ================== 3) Report & Plot =========================================
    print("\n===== Deterministic (Task1-style, means only) =====")
    print(f"Segments: {len(det_segs)}")
    print(f"Explore:  {det_exp:.2f}   Refill: {det_ref:.2f}")
    print(f"Total:    {det_total:.2f}")
    print(f"Feasibility under stochastic demands (fixed segments): {100*det_feas_rate:.2f}%")

    print("\n===== Stochastic (optimized expected distance) =====")
    print(f"Expected total distance (mean ± std over {eval_sim} sims): {best_mean:.2f} ± {best_std:.2f}")

    # One sample segs (just for visualization)
    rng_plot = np.random.default_rng(seed + 2025)
    _, _, _, segs_for_plot = decode_cost_stochastic(
        best, sample_positive_int_norm(Cdem, rng_plot), cap, Dcc, Ddc
    )

    # Plot & save (one sampled realization)
    fig = plt.figure(figsize=(6.6, 6.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.scatter(Cxy[:,0], Cxy[:,1], s=14, label="Customers")
    ax.scatter([depot_xy[0]],[depot_xy[1]], marker="*", s=120, c="tab:red", label="Depot")

    for seg in segs_for_plot:
        xs = [Cxy[i,0] for i in seg]; ys = [Cxy[i,1] for i in seg]
        ax.plot(xs, ys, "-", linewidth=1.2)
    for si, seg in enumerate(segs_for_plot, 1):
        p1, p2 = Cxy[seg[0]], Cxy[seg[-1]]
        ax.plot([depot_xy[0], p1[0]], [depot_xy[1], p1[1]], "--", linewidth=1.0)
        ax.plot([p2[0], depot_xy[0]], [p2[1], depot_xy[1]], "--", linewidth=1.0)
        ctr = Cxy[seg].mean(axis=0); ax.text(ctr[0], ctr[1], f"S{si}", fontsize=9, weight="bold")
    ax.set_title("Stochastic VRP — one sampled realization of best route")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.axis("equal"); ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {Path(out_png).resolve()}")


# ----------------------------- CLI ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("VRP.csv"))
    ap.add_argument("--out", type=Path, default=Path("stochastic_best.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=float, default=200.0)
    ap.add_argument("--pop", type=int, default=240)
    ap.add_argument("--gens", type=int, default=1600)
    ap.add_argument("--pc", type=float, default=0.95)
    ap.add_argument("--pm", type=float, default=0.25)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--tour-k", type=int, default=3)
    ap.add_argument("--twoopt-every", type=int, default=20, help="0=disable")
    ap.add_argument("--train-sim", type=int, default=64, help="MC samples per fitness call")
    ap.add_argument("--eval-sim", type=int, default=512, help="MC samples for final evaluation")
    ap.add_argument("--verbose-every", type=int, default=100)
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found.")
    solve(args.csv, args.out, seed=args.seed, cap=args.cap, pop=args.pop, gens=args.gens,
          pc=args.pc, pm=args.pm, elite=args.elite, tour_k=args.tour_k,
          twoopt_every=args.twoopt_every, train_sim=args.train_sim, eval_sim=args.eval_sim,
          verbose_every=args.verbose_every)

if __name__ == "__main__":
    main()


