# -*- coding: utf-8 -*-
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =============== Load and Augment Data (Shift Y+150 to create 100 new customers) ===============
def load_and_augment(csv_path: Path, y_shift: float = 150.0):
    """Read VRP.csv and create augmented customer list with Y-shifted by 150."""
    t = pd.read_csv(csv_path)
    is_depot = t["CUST_OR_DEPOT"].str.upper() == "DEPOT"
    depots = t[is_depot].reset_index(drop=True)
    custs  = t[~is_depot].reset_index(drop=True)
    max_no = int(custs["NO"].max())

    dup = custs.copy()
    dup["YCOORD"] = dup["YCOORD"] + y_shift
    dup["NO"] = np.arange(max_no + 1, max_no + 1 + len(dup), dtype=int)

    aug = pd.concat([custs, dup], ignore_index=True)
    return depots, aug

# =============== Distances and Decoding (Split by capacity to ensure no overflow) ===============
def precompute_dist(Cxy, depot_xy):
    n = Cxy.shape[0]
    Dcc = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(Cxy[i] - Cxy[j]))
            Dcc[i, j] = Dcc[j, i] = d
    Ddc = np.array([float(np.linalg.norm(Cxy[i] - depot_xy)) for i in range(n)], float)
    return Dcc, Ddc

def split_by_capacity(order, dem, cap):
    """Split the route into segments that do not exceed the vehicle's capacity."""
    segs, cur, load = [], [], 0.0
    for idx in order:
        d = dem[idx]
        if cur and load + d > cap:
            segs.append(cur); cur, load = [idx], d
        else:
            cur.append(idx); load += d
    if cur: segs.append(cur)
    return segs

def decode_order(order, dem, cap, Dcc, Ddc):
    """Return total distance, exploration distance, refill distance, and route segments."""
    segs = split_by_capacity(order, dem, cap)
    explore = 0.0; refill = 0.0
    for s in segs:
        refill += Ddc[s[0]] + Ddc[s[-1]]
        for a, b in zip(s[:-1], s[1:]):
            explore += Dcc[a, b]
    return explore + refill, explore, refill, segs

# =============== GA Operators (OX Crossover + Swap Mutation + 2-Opt) ===============
def nn_seed(Dcc):
    """Generate a nearest-neighbor route as the initial solution."""
    n = Dcc.shape[0]
    unused = set(range(n))
    cur = random.choice(tuple(unused)); unused.remove(cur)
    route = [cur]
    while unused:
        nxt = min(unused, key=lambda j: Dcc[cur, j])
        route.append(nxt); unused.remove(nxt); cur = nxt
    return route

def ox(p1, p2):
    """Order Crossover (OX) for combining two parent routes."""
    n = len(p1); l, r = sorted(random.sample(range(n), 2))
    c = [-1]*n; c[l:r+1] = p1[l:r+1]
    fill = [x for x in p2 if x not in c]; j = 0
    for i in range(n):
        if c[i] == -1: c[i] = fill[j]; j += 1
    return c

def swap_mut(ind):
    """Swap mutation for altering a route."""
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def two_opt(ind, Dcc):
    """2-opt optimization to improve the route."""
    out = ind[:]; n = len(out); improved = True
    while improved:
        improved = False
        for i in range(n-3):
            a,b = out[i], out[i+1]
            for k in range(i+2, n-1):
                c,d = out[k], out[k+1]
                if Dcc[a,c] + Dcc[b,d] + 1e-12 < Dcc[a,b] + Dcc[c,d]:
                    out[i+1:k+1] = reversed(out[i+1:k+1]); improved = True
    return out

def solve_region_ga(Cxy, Cdem, depot_xy, cap=200.0, seed=42,
                    pop=140, gens=800, pc=0.95, pm=0.25, elite=6, twoopt_every=20):
    """Solve the routing problem for a single region using GA."""
    random.seed(seed); np.random.seed(seed)
    n = len(Cdem)
    Dcc, Ddc = precompute_dist(Cxy, depot_xy)
    popu = [nn_seed(Dcc)] + [random.sample(range(n), n) for _ in range(pop-1)]

    best, best_cost, best_split, best_segs = None, float("inf"), (0.0,0.0), []
    for g in range(1, gens+1):
        fits = []
        for ind in popu:
            c, e, r, segs = decode_order(ind, Cdem, cap, Dcc, Ddc)
            fits.append(-c)
            if c < best_cost:
                best, best_cost, best_split, best_segs = ind[:], c, (e, r), segs
        if twoopt_every and g % twoopt_every == 0:
            imp = two_opt(best, Dcc)
            c2, e2, r2, s2 = decode_order(imp, Cdem, cap, Dcc, Ddc)
            if c2 < best_cost:
                best, best_cost, best_split, best_segs = imp, c2, (e2, r2), s2

        elite_idx = list(np.argsort(fits))[-elite:][::-1]
        new_pop = [popu[i][:] for i in elite_idx]
        while len(new_pop) < pop:
            p1 = popu[random.randrange(pop)]
            p2 = popu[random.randrange(pop)]
            child = ox(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm: child = swap_mut(child)
            new_pop.append(child)
        popu = new_pop

    return dict(total=best_cost, explore=best_split[0], refill=best_split[1],
                segments=best_segs, Dcc=Dcc, Ddc=Ddc)

# =============== Clustering + GA + Visualization ===============
def solve_and_plot(csv="VRP.csv", out_png="clustered_routes.png",
                   k=10, seed=42, cap=200.0, pop=160, gens=1200,
                   pc=0.95, pm=0.25, elite=6, twoopt_every=15, show=True):
    """Solve and visualize the large-scale VRP problem."""
    depots, aug = load_and_augment(Path(csv), y_shift=150.0)
    dep_xy = depots[["XCOORD","YCOORD"]].to_numpy(float)

    Cxy_all = aug[["XCOORD","YCOORD"]].to_numpy(float)
    Cdm_all = aug["DEMAND"].to_numpy(float)

    # Clustering
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(Cxy_all)
    centers = km.cluster_centers_

    # Assign nearest depot to each cluster
    assign_depot = []
    for c in centers:
        d = np.linalg.norm(dep_xy - c, axis=1)
        assign_depot.append(int(np.argmin(d)))

    # Solve each cluster independently
    total_distance = 0.0
    region_results = []
    print("\n==================== CLUSTERED VRP REPORT ====================")
    print(f"Depots: {len(dep_xy)} | Customers: {len(Cxy_all)} | k={k}\n")

    for cl in range(k):
        m = (labels == cl)
        Cxy = Cxy_all[m]; Cdm = Cdm_all[m]
        depot_xy = dep_xy[assign_depot[cl]]
        res = solve_region_ga(Cxy, Cdm, depot_xy, cap=cap, seed=seed+31*cl,
                              pop=pop, gens=gens, pc=pc, pm=pm,
                              elite=elite, twoopt_every=twoopt_every)
        total_distance += res["total"]
        region_results.append((cl, Cxy, depot_xy, res))
        print(f"Region #{cl:02d} | size={len(Cxy):3d} | total={res['total']:.2f} "
              f"(explore={res['explore']:.2f}, refill={res['refill']:.2f})")

    print(f"\nTOTAL DISTANCE: {total_distance:.2f}")

    # Plot the results (popup + save)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.8, 8.6), dpi=120)
    ax.scatter(dep_xy[:,0], dep_xy[:,1], marker="s", s=80, c="k", label="Depot")
    for cl, Cxy, depot_xy, res in region_results:
        color = cmap(cl % 20)
        ax.scatter(Cxy[:,0], Cxy[:,1], s=16, color=color, alpha=0.85, label=f"C{cl} (n={len(Cxy)})")
        for seg in res["segments"]:
            xs = [Cxy[i,0] for i in seg]; ys = [Cxy[i,1] for i in seg]
            ax.plot(xs, ys, "-", lw=1.2, color=color)
            ax.plot([depot_xy[0], xs[0]], [depot_xy[1], ys[0]], "--", lw=1.0, color=color)
            ax.plot([xs[-1], depot_xy[0]], [ys[-1], depot_xy[1]], "--", lw=1.0, color=color)
        # Label each cluster
        ctr = Cxy.mean(0)
        ax.text(ctr[0], ctr[1], f"#{cl}", fontsize=9, weight="bold", color=color)

    ax.set_title(f"Task 3: Clustered VRP (200 customers)\nTotal distance = {total_distance:.2f}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.axis("equal")
    # Keep only one legend entry per region
    handles, labels_ = ax.get_legend_handles_labels()
    uniq = dict(zip(labels_, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {Path(out_png).resolve()}")

    if show:
        plt.show()   # Display the plot in PyCharm

if __name__ == "__main__":
    # Direct execution: Print results, show the plot, and save PNG
    solve_and_plot(
        csv="VRP.csv",
        out_png="clustered_routes.png",
        k=10,           # Change k (clusters) from 8 to 14 based on balance
        seed=42,
        cap=200.0,
        pop=180,
        gens=1200,
        twoopt_every=15,
        show=True       # Display in PyCharm
    )













