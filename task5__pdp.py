# -*- coding: utf-8 -*-
"""
Task 5 — Pickup & Delivery VRP (single vehicle, single depot)
- 30% customers are pickups (negative demand), others deliveries (positive).
- Vehicle load must always be within [0, CAP].
- Objective: minimize total distance with feasibility (hard) enforced via repair + penalty.
- Output: route figure (map + load profile), total distance, and feasibility check.
"""

from pathlib import Path
import argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- Load VRP Data ----------------------
def load_vrp(csv: Path):
    t = pd.read_csv(csv)
    is_depot = t["CUST_OR_DEPOT"].str.upper() == "DEPOT"
    depots = t[is_depot].reset_index(drop=True)
    custs = t[~is_depot].reset_index(drop=True)

    # depot preference: NO==0 else first depot
    if (depots["NO"] == 0).any():
        drow = depots.loc[depots["NO"] == 0].iloc[0]
    else:
        drow = depots.iloc[0]

    depot = np.array([float(drow["XCOORD"]), float(drow["YCOORD"])], float)
    xy = custs[["XCOORD", "YCOORD"]].to_numpy(float)
    dem = custs["DEMAND"].to_numpy(float)
    ids = custs["NO"].to_numpy(int)

    # distance precompute
    n = len(ids)
    D = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(xy[i] - xy[j]))
            D[i, j] = D[j, i] = d
    Ddep = np.array([float(np.linalg.norm(xy[i] - depot)) for i in range(n)], float)
    return depot, xy, ids, dem, D, Ddep


# ---------------------- Modify Demands: 30% Pickups ----------------------
def make_pickups(dem_base: np.ndarray, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dem_base)
    k = int(round(frac * n))
    pick_idx = rng.choice(n, size=k, replace=False)
    dem = dem_base.copy()
    dem[pick_idx] = -np.abs(dem[pick_idx])
    # deliveries strictly positive
    for i in range(n):
        if i not in set(pick_idx):
            dem[i] = abs(dem[i])
    return dem, pick_idx


# ---------------------- Objectives and Feasibility Check ----------------------
def total_distance(order, D, Ddep):
    """Compute the total distance of the route."""
    if len(order) == 0: return 0.0
    return float(Ddep[order[0]] +
                 sum(D[a, b] for a, b in zip(order[:-1], order[1:])) +
                 Ddep[order[-1]])


def check_feas(order, dem, cap):
    """Return (feasible, violations, min_load, max_load)."""
    load = 0.0
    min_l = 0.0
    max_l = 0.0
    vio = 0
    for i in order:
        load += dem[i]
        min_l = min(min_l, load)
        max_l = max(max_l, load)
        if load < 0.0 or load > cap:
            vio += 1
    return (vio == 0), vio, min_l, max_l


# ---------------------- Strict Feasibility Repair ----------------------
def repair_order(order, dem, cap, D=None, Ddep=None):
    """
    Strong feasibility repair / constructor.
    If D and Ddep are provided, we greedily rebuild a route that keeps
    cumulative load in [0, cap] at every step (nearest-feasible-next rule).
    Fallback: old swap-repair when D/Ddep is None.
    """
    # --- If no D and Ddep, fallback to the old repair method ---
    if D is None or Ddep is None:
        out = order[:]
        n = len(out)
        load = 0.0
        i = 0
        while i < n:
            need = dem[out[i]]
            nxt = load + need
            if 0.0 <= nxt <= cap:
                load = nxt
                i += 1
                continue
            fixed = False
            for j in range(i + 1, n):
                cand = out[j]
                cand_nxt = load + dem[cand]
                if 0.0 <= cand_nxt <= cap:
                    out[i], out[j] = out[j], out[i]
                    load = cand_nxt
                    fixed = True
                    break
            if not fixed:
                load = nxt  # This will be penalized
                i += 1
        return out

    # --- Strict feasibility repair: greedily build a route from the depot ---
    n = len(dem)
    unvis = set(range(n))
    cur = int(np.argmin(Ddep))  # Start from the depot
    route = [cur]
    unvis.remove(cur)
    load = max(0.0, min(cap, dem[cur] if 0 <= dem[cur] <= cap else 0.0))

    if not (0.0 <= dem[cur] <= cap):
        cur = None
        for k in sorted(range(n), key=lambda i: Ddep[i]):
            if 0.0 <= dem[k] <= cap:
                cur = k
                break
        if cur is None:  # Extreme case: all first steps infeasible
            cur = int(np.argmin(Ddep))
            load = 0.0
        route = [cur]
        unvis = set(range(n))
        unvis.remove(cur)
        load = dem[cur] if 0.0 <= dem[cur] <= cap else 0.0

    while unvis:
        feas_cands = []
        for j in unvis:
            nxt = load + dem[j]
            if 0.0 <= nxt <= cap:
                feas_cands.append(j)
        if feas_cands:
            nxt = min(feas_cands, key=lambda j: D[cur, j])
            load += dem[nxt]
            route.append(nxt)
            unvis.remove(nxt)
            cur = nxt
        else:
            # No immediate feasible: pick the best available
            if load > cap:
                pool = [j for j in unvis if load + dem[j] >= 0]  # Prioritize pickups
                if pool:
                    nxt = min(pool, key=lambda j: (abs((load + dem[j]) - cap), D[cur, j]))
                else:
                    nxt = min(unvis, key=lambda j: (max(0.0, load + dem[j] - cap), D[cur, j]))
            else:
                pool = [j for j in unvis if load + dem[j] <= cap]  # Prioritize deliveries
                if pool:
                    nxt = min(pool, key=lambda j: (abs((load + dem[j]) - 0.0), D[cur, j]))
                else:
                    nxt = min(unvis, key=lambda j: (max(0.0, 0.0 - (load + dem[j])), D[cur, j]))
            load += dem[nxt]
            route.append(nxt)
            unvis.remove(nxt)
            cur = nxt
    return route


# ---------------------- GA Operators ----------------------
def nn_seed(D):
    """Generate a nearest-neighbor route as the initial solution."""
    n = D.shape[0]
    unused = set(range(n))
    cur = random.choice(tuple(unused));
    unused.remove(cur)
    route = [cur]
    while unused:
        nxt = min(unused, key=lambda j: D[cur, j])
        route.append(nxt);
        unused.remove(nxt);
        cur = nxt
    return route


def ox_cross(p1, p2):
    """Order Crossover (OX) for combining two parent routes."""
    n = len(p1);
    l, r = sorted(random.sample(range(n), 2))
    c = [-1] * n;
    c[l:r + 1] = p1[l:r + 1]
    fill = [x for x in p2 if x not in c];
    j = 0
    for i in range(n):
        if c[i] == -1: c[i] = fill[j]; j += 1
    return c


def swap_mut(ind):
    """Swap mutation for altering a route."""
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind


def two_opt(ind, D):
    """2-opt optimization to improve the route."""
    out = ind[:];
    n = len(out);
    improved = True
    while improved:
        improved = False
        for i in range(n - 3):
            a, b = out[i], out[i + 1]
            for k in range(i + 2, n - 1):
                c, d = out[k], out[k + 1]
                if D[a, c] + D[b, d] + 1e-12 < D[a, b] + D[c, d]:
                    out[i + 1:k + 1] = reversed(out[i + 1:k + 1]);
                    improved = True
    return out


# ---------------------- GA Main Function ----------------------
def ga_pdp(xy, dem, D, Ddep, cap=200.0, seed=42,
           pop=240, gens=1500, pc=0.95, pm=0.25, elite=6,
           twoopt_every=25, verbose_every=100):
    random.seed(seed);
    np.random.seed(seed)
    n = len(dem)

    # init population: NN seed + random perms; repair for feasibility tendency
    popu = [repair_order(nn_seed(D), dem, cap, D, Ddep)] + \
           [repair_order(random.sample(range(n), n), dem, cap, D, Ddep) for _ in range(pop - 1)]

    def fitness(ind):
        """Minimize: distance + penalty for constraint violations."""
        dist = total_distance(ind, D, Ddep)
        feas, vio, min_l, max_l = check_feas(ind, dem, cap)
        if feas:
            return dist, 0.0
        # soft penalties to guide search
        overflow = max(0.0, max_l - cap)
        underflow = max(0.0, -min_l)
        penalty = 1e5 * vio + 5e3 * (overflow + underflow)
        return dist + penalty, penalty

    best, best_fit, best_pen = None, float("inf"), float("inf")

    for g in range(1, gens + 1):
        vals = [];
        pens = []
        for ind in popu:
            v, p = fitness(ind)
            vals.append(v);
            pens.append(p)
            if (p < 1e-9 and v < best_fit) or (best is None):
                best, best_fit, best_pen = ind[:], v, p

        # periodic 2-opt on current best
        if twoopt_every and g % twoopt_every == 0 and best is not None:
            imp = two_opt(best, D)
            imp = repair_order(imp, dem, cap, D, Ddep)
            v, p = fitness(imp)
            if v < best_fit:
                best, best_fit, best_pen = imp, v, p

        # elitism
        elite_idx = list(np.argsort(vals))[:elite]
        new_pop = [popu[i][:] for i in elite_idx]
        # offspring
        while len(new_pop) < pop:
            p1 = popu[random.randrange(pop)]
            p2 = popu[random.randrange(pop)]
            child = ox_cross(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm: child = swap_mut(child)
            child = repair_order(child, dem, cap, D, Ddep)
            new_pop.append(child)
        popu = new_pop

        if verbose_every and g % verbose_every == 0:
            feas, _, _, _ = check_feas(best, dem, cap)
            print(f"[GA] Gen {g:4d} | best={best_fit:.2f} | feasible={feas}")

    # finalize
    dist = total_distance(best, D, Ddep)
    feas, vio, min_l, max_l = check_feas(best, dem, cap)
    return best, dict(
        distance=dist, feasible=feas, violations=int(vio),
        min_load=min_l, max_load=max_l
    )


# ---------------------- Visualization ----------------------
def plot_route_and_load(xy, depot, order, dem, cap, title, out_png, pickups_idx):
    # route panel
    fig = plt.figure(figsize=(12, 5), dpi=130)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    xs = [depot[0]] + [xy[i, 0] for i in order] + [depot[0]]
    ys = [depot[1]] + [xy[i, 1] for i in order] + [depot[1]]
    ax.plot(xs, ys, "-", lw=1.2, color="tab:gray", alpha=0.9)
    # mark customers (deliveries vs pickups)
    mask_pick = np.zeros(len(xy), dtype=bool);
    mask_pick[pickups_idx] = True
    ax.scatter(xy[~mask_pick, 0], xy[~mask_pick, 1], s=18, c="tab:blue", label="Delivery (+)")
    ax.scatter(xy[mask_pick, 0], xy[mask_pick, 1], s=18, c="tab:orange", label="Pickup (-)")
    ax.scatter([depot[0]], [depot[1]], marker="*", s=140, c="tab:red", edgecolor="k", label="Depot")
    ax.set_title(title);
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.axis("equal");
    ax.legend()

    # load profile
    ax2 = fig.add_subplot(gs[0, 1])
    load = 0.0;
    loads = [0.0]
    for i in order:
        load += dem[i];
        loads.append(load)
    ax2.plot(range(len(loads)), loads, marker="o", ms=3)
    ax2.axhline(0.0, color="k", lw=1, ls="--")
    ax2.axhline(cap, color="k", lw=1, ls="--")
    ax2.set_title("Load profile");
    ax2.set_xlabel("Step");
    ax2.set_ylabel("Load")
    ax2.set_ylim(min(-5, min(loads) - 10), max(cap + 5, max(loads) + 10))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {Path(out_png).resolve()}")
    plt.show()


# ---------------------- Runner ----------------------
def run(csv: Path, out_png: Path,
        cap=200.0, pickup_frac=0.30,
        seed=42, pop=240, gens=1500, pc=0.95, pm=0.25, elite=6, twoopt_every=25,
        verbose=True):
    depot, xy, ids, dem_base, D, Ddep = load_vrp(csv)
    dem, pickups_idx = make_pickups(dem_base, frac=pickup_frac, seed=seed)

    print(f"Customers: {len(ids)} | pickups: {len(pickups_idx)} ({pickup_frac * 100:.0f}%) | cap={cap}")
    best, stats = ga_pdp(
        xy, dem, D, Ddep, cap=cap, seed=seed, pop=pop, gens=gens,
        pc=pc, pm=pm, elite=elite, twoopt_every=twoopt_every,
        verbose_every=100 if verbose else 0
    )

    print("\n===== PDVRP RESULT =====")
    print(f"Total distance: {stats['distance']:.2f}")
    print(f"Feasible: {stats['feasible']} | violations: {stats['violations']}")
    print(f"Min load: {stats['min_load']:.1f} | Max load: {stats['max_load']:.1f}")

    title = f"Pickup & Delivery VRP — distance={stats['distance']:.1f}, feasible={stats['feasible']}"
    plot_route_and_load(xy, depot, best, dem, cap, title, out_png, pickups_idx)


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("VRP.csv"))
    ap.add_argument("--out", type=Path, default=Path("pdp_route.png"))
    ap.add_argument("--cap", type=float, default=200.0)
    ap.add_argument("--pickup-frac", type=float, default=0.30, help="fraction of customers set to pickups")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=240)
    ap.add_argument("--gens", type=int, default=1500)
    ap.add_argument("--pc", type=float, default=0.95)
    ap.add_argument("--pm", type=float, default=0.25)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--twoopt-every", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found.")
    run(args.csv, args.out, cap=args.cap, pickup_frac=args.pickup_frac, seed=args.seed,
        pop=args.pop, gens=args.gens, pc=args.pc, pm=args.pm, elite=args.elite,
        twoopt_every=args.twoopt_every, verbose=args.verbose)


if __name__ == "__main__":
    main()




