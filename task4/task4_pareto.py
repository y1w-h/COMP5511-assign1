# -*- coding: utf-8 -*-
"""
Task 4 — Multi-Objective VRP (single depot, 100 customers, TSP-style)
Objectives:
  f1: minimize total distance (including depot -> first and last -> depot)
  f2: maximize sum_i (EFF_i - d_i), where d_i is cumulative distance from depot
Approaches:
  (A) Weighted GA with f = w*f1 - (1-w)*f2  for multiple w
  (B) NSGA-II to approximate Pareto front
Outputs:
  - Console report for weighted runs and NSGA-II non-dominated set
  - pareto plot (f1 vs f2) + two sample routes (best f1 & best f2)
"""

from pathlib import Path
import argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- data & distances ----------------------
def load_vrp(csv: Path):
    t = pd.read_csv(csv)
    is_depot = t["CUST_OR_DEPOT"].str.upper() == "DEPOT"
    depots = t[is_depot].reset_index(drop=True)
    custs  = t[~is_depot].reset_index(drop=True)
    # depot pick: NO==0 preferred else first
    if (depots["NO"] == 0).any():
        drow = depots.loc[depots["NO"] == 0].iloc[0]
    else:
        drow = depots.iloc[0]
    depot = np.array([float(drow["XCOORD"]), float(drow["YCOORD"])], float)
    xy   = custs[["XCOORD","YCOORD"]].to_numpy(float)
    eff  = custs["EFFICIENCY"].to_numpy(float)
    ids  = custs["NO"].to_numpy(int)
    n = len(ids)

    # distance matrices
    D = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(xy[i] - xy[j]))
            D[i,j] = D[j,i] = d
    Ddep = np.array([float(np.linalg.norm(xy[i] - depot)) for i in range(n)], float)
    return depot, xy, ids, eff, D, Ddep

# ---------------------- objectives ----------------------
def f1_total_distance(order, D, Ddep):
    """Depot->first + sum + last->depot."""
    if len(order) == 0: return 0.0
    total = Ddep[order[0]] + sum(D[a,b] for a,b in zip(order[:-1], order[1:])) + Ddep[order[-1]]
    return float(total)

def f2_total_eff(order, eff, D, Ddep):
    """Sum_i (EFF_i - d_i), cumulative d_i along route from depot."""
    if len(order) == 0: return 0.0
    cum = Ddep[order[0]]
    s = eff[order[0]] - cum
    for a,b in zip(order[:-1], order[1:]):
        cum += D[a,b]
        s += eff[b] - cum
    return float(s)

def eval_biobj(order, eff, D, Ddep):
    return f1_total_distance(order, D, Ddep), f2_total_eff(order, eff, D, Ddep)

# ---------------------- GA operators ----------------------
def tournament_select(pop, keyvals, k=3, reverse=False):
    """Select by key (higher is better if reverse=True, else lower)."""
    cand = random.sample(range(len(pop)), k)
    best = cand[0]
    for c in cand[1:]:
        if (keyvals[c] > keyvals[best]) if reverse else (keyvals[c] < keyvals[best]):
            best = c
    return pop[best][:]

def ox_cross(p1, p2):
    n = len(p1)
    l, r = sorted(random.sample(range(n), 2))
    c = [-1]*n
    c[l:r+1] = p1[l:r+1]
    fill = [x for x in p2 if x not in c]
    j = 0
    for i in range(n):
        if c[i] == -1:
            c[i] = fill[j]; j += 1
    return c

def swap_mut(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def two_opt_perm(ind, D):
    out = ind[:]; n = len(out); improved = True
    while improved:
        improved = False
        for i in range(n-3):
            a,b = out[i], out[i+1]
            for k in range(i+2, n-1):
                c,d = out[k], out[k+1]
                if D[a,c] + D[b,d] + 1e-12 < D[a,b] + D[c,d]:
                    out[i+1:k+1] = reversed(out[i+1:k+1]); improved = True
    return out

def nn_seed(D):
    n = D.shape[0]
    unused = set(range(n))
    cur = random.choice(tuple(unused)); unused.remove(cur)
    route = [cur]
    while unused:
        nxt = min(unused, key=lambda j: D[cur,j])
        route.append(nxt); unused.remove(nxt); cur = nxt
    return route

# ---------------------- Weighted GA ----------------------
def weighted_ga(eff, D, Ddep, w=0.5, seed=42, pop=240, gens=1500, pc=0.95, pm=0.25, elite=6, twoopt_every=25, verbose_every=100):
    random.seed(seed); np.random.seed(seed)
    n = D.shape[0]
    popu = [nn_seed(D)] + [random.sample(range(n), n) for _ in range(pop-1)]
    def score(order):
        f1, f2 = eval_biobj(order, eff, D, Ddep)
        return w*f1 - (1.0-w)*f2
    best, best_val, best_f = None, float("inf"), (None,None)

    for g in range(1, gens+1):
        vals = [score(ind) for ind in popu]
        # record best
        i = int(np.argmin(vals))
        if vals[i] < best_val:
            best_val = vals[i]; best = popu[i][:]; best_f = eval_biobj(best, eff, D, Ddep)
        # periodic 2-opt on current best
        if twoopt_every and g % twoopt_every == 0:
            imp = two_opt_perm(best, D)
            v = score(imp)
            if v < best_val:
                best_val = v; best = imp; best_f = eval_biobj(best, eff, D, Ddep)
        # elitism + offspring
        elite_idx = list(np.argsort(vals))[:elite]
        new_pop = [popu[i][:] for i in elite_idx]
        while len(new_pop) < pop:
            p1 = tournament_select(popu, vals, k=3, reverse=False)
            p2 = tournament_select(popu, vals, k=3, reverse=False)
            child = ox_cross(p1,p2) if random.random()<pc else p1[:]
            if random.random()<pm: child = swap_mut(child)
            new_pop.append(child)
        popu = new_pop
        if verbose_every and g % verbose_every == 0:
            print(f"[Weighted w={w:0.2f}] Gen {g:4d} | best f={best_val:.3f} | f1={best_f[0]:.2f}, f2={best_f[1]:.2f}")
    return best, best_f

# ---------------------- NSGA-II ----------------------
def dominates(fa, fb):
    """fa dominates fb if fa is no worse in all (f1 min, f2 max) and better in at least one."""
    # Convert to minimization pair: (f1, -f2)
    za = (fa[0], -fa[1]); zb = (fb[0], -fb[1])
    no_worse = (za[0] <= zb[0] and za[1] <= zb[1])
    strictly_better = (za[0] < zb[0] or za[1] < zb[1])
    return no_worse and strictly_better

def fast_non_dom_sort(F):
    n = len(F)
    S = [set() for _ in range(n)]
    n_dom = [0]*n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p==q: continue
            if dominates(F[p], F[q]):
                S[p].add(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p]==0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q]-=1
                if n_dom[q]==0:
                    Q.append(q)
        i+=1; fronts.append(Q)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front, F):
    m = len(front)
    if m==0: return {}
    dist = {i:0.0 for i in front}
    # dimension 1: f1 (min)
    ord1 = sorted(front, key=lambda i: F[i][0])
    dist[ord1[0]] = dist[ord1[-1]] = float('inf')
    fmin, fmax = F[ord1[0]][0], F[ord1[-1]][0]
    rng = (fmax - fmin) if fmax>fmin else 1.0
    for j in range(1,m-1):
        dist[ord1[j]] += (F[ord1[j+1]][0] - F[ord1[j-1]][0]) / rng
    # dimension 2: f2 (max) -> still use numeric span
    ord2 = sorted(front, key=lambda i: F[i][1])
    dist[ord2[0]] = dist[ord2[-1]] = float('inf')
    fmin, fmax = F[ord2[0]][1], F[ord2[-1]][1]
    rng = (fmax - fmin) if fmax>fmin else 1.0
    for j in range(1,m-1):
        dist[ord2[j]] += (F[ord2[j+1]][1] - F[ord2[j-1]][1]) / rng
    return dist

def nsga2(eff, D, Ddep, seed=42, pop=240, gens=800, pc=0.95, pm=0.25, twoopt_every=25, verbose_every=100):
    random.seed(seed); np.random.seed(seed)
    n = D.shape[0]
    popu = [nn_seed(D)] + [random.sample(range(n), n) for _ in range(pop-1)]
    Fvals = [eval_biobj(ind, eff, D, Ddep) for ind in popu]

    for g in range(1, gens+1):
        # create offspring
        children=[]
        while len(children) < pop:
            i = random.randrange(pop); j = random.randrange(pop)
            p1, p2 = popu[i][:], popu[j][:]
            child = ox_cross(p1,p2) if random.random()<pc else p1[:]
            if random.random()<pm: child = swap_mut(child)
            children.append(child)
        # evaluate
        Fchild = [eval_biobj(ind, eff, D, Ddep) for ind in children]

        # combine and sort by fronts + crowding
        all_pop = popu + children
        all_F   = Fvals + Fchild
        fronts = fast_non_dom_sort(all_F)

        new_pop, new_F = [], []
        for fr in fronts:
            if len(new_pop) + len(fr) <= pop:
                new_pop += [all_pop[i] for i in fr]
                new_F   += [all_F[i] for i in fr]
            else:
                cd = crowding_distance(fr, all_F)
                fr_sorted = sorted(fr, key=lambda i: cd[i], reverse=True)
                remain = pop - len(new_pop)
                cut = fr_sorted[:remain]
                new_pop += [all_pop[i] for i in cut]
                new_F   += [all_F[i] for i in cut]
                break

        popu, Fvals = new_pop, new_F

        # occasional 2-opt on the current extreme solutions (best f1, best f2)
        if twoopt_every and g % twoopt_every == 0:
            idx_best_f1 = int(np.argmin([f[0] for f in Fvals]))
            idx_best_f2 = int(np.argmax([f[1] for f in Fvals]))
            for idx in {idx_best_f1, idx_best_f2}:
                imp = two_opt_perm(popu[idx], D)
                fnew = eval_biobj(imp, eff, D, Ddep)
                if dominates(fnew, Fvals[idx]) or (fnew==Fvals[idx]):
                    popu[idx] = imp; Fvals[idx] = fnew

        if verbose_every and g % verbose_every == 0:
            # count unique non-dominated in current population
            fronts_now = fast_non_dom_sort(Fvals)
            print(f"[NSGA-II] Gen {g:4d} | nondom size={len(fronts_now[0])} | "
                  f"best f1={min(f[0] for f in Fvals):.2f}, best f2={max(f[1] for f in Fvals):.2f}")

    # extract final nondominated set
    fronts_done = fast_non_dom_sort(Fvals)
    nd_idx = fronts_done[0]
    return [popu[i] for i in nd_idx], [Fvals[i] for i in nd_idx]

# ---------------------- plotting helpers ----------------------
def plot_route(xy, depot, order, title, ax):
    xs = [depot[0]] + [xy[i,0] for i in order] + [depot[0]]
    ys = [depot[1]] + [xy[i,1] for i in order] + [depot[1]]
    ax.plot(xs, ys, "-", linewidth=1.2)
    ax.scatter(xy[:,0], xy[:,1], s=12)
    ax.scatter([depot[0]],[depot[1]], marker="*", s=120, c="tab:red")
    ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.axis("equal")

# ---------------------- run everything ----------------------
def run(csv: Path, out_png: Path,
        weights=(0.0, 0.25, 0.5, 0.75, 1.0),
        seed=42, pop=240, gens_w=1500, gens_nsga=800,
        pc=0.95, pm=0.25, twoopt_every=25, verbose=True):

    depot, xy, ids, eff, D, Ddep = load_vrp(csv)

    # A) Weighted GA over multiple w
    print("\n==== Weighted GA (single-objective scalarization) ====")
    weighted_solutions = []
    for w in weights:
        best, (bf1, bf2) = weighted_ga(eff, D, Ddep, w=w, seed=seed+int(100*w),
                                       pop=pop, gens=gens_w, pc=pc, pm=pm,
                                       twoopt_every=twoopt_every,
                                       verbose_every=200 if verbose else 0)
        print(f"w={w:.2f} -> f1={bf1:.2f}, f2={bf2:.2f}")
        weighted_solutions.append((w, best, (bf1, bf2)))

    # B) NSGA-II
    print("\n==== NSGA-II (Pareto dominance) ====")
    nd_routes, nd_F = nsga2(eff, D, Ddep, seed=seed+777, pop=pop,
                            gens=gens_nsga, pc=pc, pm=pm,
                            twoopt_every=twoopt_every,
                            verbose_every=200 if verbose else 0)
    # Sort nondominated by f1
    order_nd = np.argsort([f[0] for f in nd_F])
    nd_routes = [nd_routes[i] for i in order_nd]
    nd_F      = [nd_F[i] for i in order_nd]

    print(f"Non-dominated set size: {len(nd_F)}")
    if len(nd_F):
        print(f"  best f1={nd_F[0][0]:.2f}, max f2={max(f[1] for f in nd_F):.2f}")

    # ------------ Plot Pareto + two example routes ------------
    fig = plt.figure(figsize=(12, 5), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1,1.2])

    # Pareto panel
    ax0 = fig.add_subplot(gs[0, :])
    # scatter all NSGA-II population
    xs = [f[0] for f in nd_F]
    ys = [f[1] for f in nd_F]
    ax0.scatter(xs, ys, s=16, c="tab:blue", label="NSGA-II nondominated")
    # weighted points
    for w, _, f in weighted_solutions:
        ax0.scatter([f[0]], [f[1]], s=28, marker="x", label=f"weighted w={w:.2f}")
    ax0.set_xlabel("f1 = total distance (min)")
    ax0.set_ylabel("f2 = total efficiency (max)")
    ax0.set_title("Pareto Front & Weighted Solutions")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.25)

    # route examples: best f1 and best f2 from NSGA-II
    if nd_routes:
        idx_best_f1 = int(np.argmin([f[0] for f in nd_F]))
        idx_best_f2 = int(np.argmax([f[1] for f in nd_F]))
        ax1 = fig.add_subplot(gs[1, 0])
        plot_route(xy, depot, nd_routes[idx_best_f1], f"Best f1: {nd_F[idx_best_f1][0]:.1f}", ax1)
        ax2 = fig.add_subplot(gs[1, 1])
        plot_route(xy, depot, nd_routes[idx_best_f2], f"Best f2: {nd_F[idx_best_f2][1]:.1f}", ax2)
        # best weighted (closest to center weight)
        wmid_idx = np.argmin([abs(w-0.5) for w,_,_ in weighted_solutions])
        wmid_route = weighted_solutions[wmid_idx][1]
        wmid_f = weighted_solutions[wmid_idx][2]
        ax3 = fig.add_subplot(gs[1, 2])
        plot_route(xy, depot, wmid_route, f"Weighted w≈0.5\nf1={wmid_f[0]:.1f}, f2={wmid_f[1]:.1f}", ax3)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {Path(out_png).resolve()}")

    # ------------ Text summary ------------
    print("\n=== Summary ===")
    print("Weighted GA results:")
    for w, _, f in weighted_solutions:
        print(f"  w={w:.2f}: f1={f[0]:.2f}, f2={f[1]:.2f}")
    print(f"NSGA-II nondominated set size: {len(nd_F)} (saved in plot).")

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("VRP.csv"))
    ap.add_argument("--out", type=Path, default=Path("pareto.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=240)
    ap.add_argument("--gens-w", type=int, default=1500, help="gens for weighted GA")
    ap.add_argument("--gens-nsga", type=int, default=800, help="gens for NSGA-II")
    ap.add_argument("--twoopt-every", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found.")
    run(args.csv, args.out, seed=args.seed, pop=args.pop,
        gens_w=args.gens_w, gens_nsga=args.gens_nsga,
        twoopt_every=args.twoopt_every, verbose=args.verbose)

if __name__ == "__main__":
    main()






