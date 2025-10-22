# -*- coding: utf-8 -*-
"""
Simple GA for VRP (single depot, capacity=200) — one continuous run.

Features:
- Genome: permutation of all customers
- Decode: greedy split by capacity; cost = explore(cust-cust) + refill(depot legs)
- GA: tournament selection + OX crossover + swap mutation + elitism
- Optional 2-opt (on permutation) every k generations
- Saves a PNG figure (300 dpi)

Usage:
    python task1_vrp_simple.py --csv VRP.csv --out VRP_best.png \
        --seed 42 --gens 2200 --pop 260 --twoopt-every 20 --verbose-every 100
"""

from pathlib import Path
import argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- data & distances -----------------------------
def load_data(csv_path: Path):
    """Read VRP.csv and build distance matrices."""
    t = pd.read_csv(csv_path)
    is_depot = t["CUST_OR_DEPOT"].str.upper() == "DEPOT"
    depots = t[is_depot].reset_index(drop=True)
    custs  = t[~is_depot].reset_index(drop=True)

    # choose depot with NO==0 if present, else the first depot row
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


# ----------------------------- decoding -------------------------------------
def split_by_capacity(perm, demand, cap):
    segs, cur, load = [], [], 0.0
    for idx in perm:
        dem = demand[idx]
        if cur and load + dem > cap:
            segs.append(cur); cur = [idx]; load = dem
        else:
            cur.append(idx); load += dem
    if cur: segs.append(cur)
    return segs

def decode_cost(perm, demand, cap, Dcc, Ddc):
    segs = split_by_capacity(perm, demand, cap)
    explore = 0.0; refill = 0.0
    for seg in segs:
        refill += Ddc[seg[0]] + Ddc[seg[-1]]
        for a,b in zip(seg[:-1], seg[1:]):
            explore += Dcc[a,b]
    return explore + refill, explore, refill, segs


# ----------------------------- GA operators ---------------------------------
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


# ----------------------------- local search (optional) -----------------------
def two_opt_on_perm(ind, Dcc, passes=1):
    """2-opt on the customer graph (does not change capacity splits)."""
    out = ind[:]; n = len(out)
    for _ in range(passes):
        improved = False
        for i in range(n-3):
            a, b = out[i], out[i+1]
            for k in range(i+2, n-1):
                c, d = out[k], out[k+1]
                old = Dcc[a,b] + Dcc[c,d]
                neu = Dcc[a,c] + Dcc[b,d]
                if neu + 1e-12 < old:
                    out[i+1:k+1] = reversed(out[i+1:k+1])
                    improved = True
        if not improved: break
    return out


# ----------------------------- solver (single run) ---------------------------
def solve(csv: Path, out_png: Path, seed=42, pop=260, gens=2200,
          cap=200.0, pc=0.95, pm=0.25, elite=6, tour_k=3,
          twoopt_every=20, verbose_every=100):
    random.seed(seed); np.random.seed(seed)
    depot_xy, Cxy, Cno, Cdem, Dcc, Ddc = load_data(csv)
    n = len(Cno)

    # init population: one NN seed + the rest random
    def nn_seed():
        unused = set(range(n))
        cur = random.choice(tuple(unused)); unused.remove(cur)
        route = [cur]
        while unused:
            nxt = min(unused, key=lambda j: Dcc[cur,j])
            route.append(nxt); unused.remove(nxt); cur = nxt
        return route

    popu = [nn_seed()] + [random.sample(range(n), n) for _ in range(pop-1)]

    best, best_cost, best_split, best_segs = None, float("inf"), (0.0,0.0), []
    for gen in range(1, gens+1):
        fits = []
        for ind in popu:
            c, e, r, segs = decode_cost(ind, Cdem, cap, Dcc, Ddc)
            fits.append(-c)
            if c < best_cost:
                best, best_cost, best_split, best_segs = ind[:], c, (e,r), segs

        # periodic 2-opt on current best (optional)
        if twoopt_every and gen % twoopt_every == 0:
            imp = two_opt_on_perm(best, Dcc, passes=1)
            c2, e2, r2, segs2 = decode_cost(imp, Cdem, cap, Dcc, Ddc)
            if c2 < best_cost:
                best, best_cost, best_split, best_segs = imp, c2, (e2,r2), segs2

        # elitism + offspring
        elite_idx = np.argsort(fits)[-elite:][::-1]
        new_pop = [popu[i][:] for i in elite_idx]
        while len(new_pop) < pop:
            p1 = tournament_select(popu, fits, tour_k)
            p2 = tournament_select(popu, fits, tour_k)
            child = ox_crossover(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm:
                child = swap_mutation(child)
            new_pop.append(child)
        popu = new_pop

        if verbose_every and gen % verbose_every == 0:
            print(f"Gen {gen:4d} | best={best_cost:.2f} "
                  f"(explore={best_split[0]:.2f}, refill={best_split[1]:.2f})")

    # final report
    total, explore, refill, segs = decode_cost(best, Cdem, cap, Dcc, Ddc)
    print("\n=== BEST (single run) ===")
    print(f"Segments: {len(segs)}")
    print(f"Explore distance: {explore:.2f}")
    print(f"Refill  distance: {refill:.2f}")
    print(f"Total   distance: {total:.2f}")
    for i, seg in enumerate(segs, 1):
        seg_no = [int(Cno[k]) for k in seg]
        seg_load = int(sum(Cdem[k] for k in seg))
        seg_len = Ddc[seg[0]] + Ddc[seg[-1]] + sum(Dcc[a,b] for a,b in zip(seg[:-1], seg[1:]))
        print(f"  Route {i:02d} | load={seg_load:3d} | dist={seg_len:8.2f} | customers: {seg_no}")

    # plot & save
    fig = plt.figure(figsize=(6.6, 6.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.scatter(Cxy[:,0], Cxy[:,1], s=14, label="Customers")
    ax.scatter([depot_xy[0]],[depot_xy[1]], marker="*", s=120, c="tab:red", label="Depot")
    for seg in segs:
        xs = [Cxy[i,0] for i in seg]; ys = [Cxy[i,1] for i in seg]
        ax.plot(xs, ys, "-", linewidth=1.2)
    for si, seg in enumerate(segs, 1):
        p1, p2 = Cxy[seg[0]], Cxy[seg[-1]]
        ax.plot([depot_xy[0], p1[0]], [depot_xy[1], p1[1]], "--", linewidth=1.0)
        ax.plot([p2[0], depot_xy[0]], [p2[1], depot_xy[1]], "--", linewidth=1.0)
        ctr = Cxy[seg].mean(axis=0); ax.text(ctr[0], ctr[1], f"S{si}", fontsize=9, weight="bold")
    ax.set_title("VRP Route — Exploring vs Refill Legs")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.axis("equal"); ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {Path(out_png).resolve()}")


# ----------------------------- CLI ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("VRP.csv"))
    ap.add_argument("--out", type=Path, default=Path("VRP_best.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=2200)
    ap.add_argument("--pop",  type=int, default=260)
    ap.add_argument("--pc",   type=float, default=0.95)
    ap.add_argument("--pm",   type=float, default=0.25)
    ap.add_argument("--elite",type=int,   default=6)
    ap.add_argument("--tour-k", type=int, default=3)
    ap.add_argument("--twoopt-every", type=int, default=20, help="0=disable")
    ap.add_argument("--verbose-every", type=int, default=100)
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found.")
    solve(args.csv, args.out, seed=args.seed, pop=args.pop, gens=args.gens,
          cap=200.0, pc=args.pc, pm=args.pm, elite=args.elite, tour_k=args.tour_k,
          twoopt_every=args.twoopt_every, verbose_every=args.verbose_every)

if __name__ == "__main__":
    main()







