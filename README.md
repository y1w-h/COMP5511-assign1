# COMP5511-assign1
https://github.com/y1w-h/COMP5511-assign1

## 📘 Overview

This project uses five evolutionary algorithms to solve different forms of the Vehicle Routing Problem. Starting with a simple single-depot VRP, the challenges become more complex, involving stochastic, large-scale, multi-objective, and pickup and delivery scenarios.

All implementations are entirely developed in Python, using no external GA libraries, and feature route visualization as well as distance reporting.



## 🧩 File Structure
File Description

VRP.csv Input dataset containing depots, customer coordinates, demands, and efficiency scores.

task1_vrp.py Classical VRP using Genetic Algorithm (GA).

task2_stochastic_vrp.py Stochastic-demand VRP (Monte Carlo simulation + GA).

task3_large_vrp.py Large-scale VRP (K-Means clustering + regional GA).

task4_pareto.py Multi-objective VRP (Weighted GA + NSGA-II).

task5__pdp.py Pickup & Delivery VRP (with negative demands, feasibility repair, and penalties).


VRP_best.png, stochastic_best.png, clustered_routes.png, pareto.png, pdp_route.png. Output route visualizations for Tasks 1–5, respectively.


## ✅Environment Setup

### 1. Python Version

The code was tested on Python 3.13.
Recommended environment:

conda create -n comp5511 python=3.13
conda activate comp5511

### 2. Install Dependencies
pip install numpy pandas matplotlib scikit-learn



 ## 🤔How to Run Each Task

### Task 1 – Classical VRP 

Objective: Reduce overall distance while maintaining capacity (200).

Run:

python task1_vrp.py --csv VRP.csv --out VRP_best.png --seed 42 --gens 2200 --pop 260

Output:

Console summary of route segments, load, and distances

VRP_best.png: visualization of the best route



### Task 2 – Stochastic Demand VRP

Objective: Under ambiguous conditions, maximize the anticipated total distance.

Run:

python task2_stochastic_vrp.py --csv VRP.csv --out stochastic_best.png --train-sim 64 --eval-sim 512

Output:

Comparison of deterministic and stochastic results
Feasibility rate and anticipated distance

stochastic_best.png: sampled route visualization



### Task 3 – Large-Scale VRP

Goal: Solve a 200-customer VRP with clustering and GA.

Run:

python task3_large_vrp.py --csv VRP.csv --out clustered_routes.png

Output:

Cluster-wise report of distances and loads

clustered_routes.png: all clustered routes visualized



### Task 4 – Multi-Objective VRP

Goal:

1: minimize total distance

2: maximize total efficiency (EFF_i - d_i)
Methods: Weighted GA and NSGA-II

Run:

python task4_pareto.py --csv VRP.csv --out pareto.png --verbose

Output:

Weighted GA results for multiple weights

NSGA-II Pareto front

pareto.png: Pareto front + sample routes



### Task 5 – Pickup & Delivery VRP

Objective: Minimize total distance while keeping vehicle load within ∈ [0, 200].
Run:

python task5__pdp.py --csv VRP.csv --out pdp_route.png --pickup-frac 0.3

Output:

Route viability, maximum and minimum loads, and overall distance

pdp_route.png: route + load profile plot



## 👏🏻 Implementation Highlights
Task Key Techniques Notes
1 Conventional GA (OX crossover, swap mutation, tournament selection, 2-opt local search) Deterministic starting point
2 Simulation of stochastic demands using Monte Carlo Expected distance = fitness
3 per-region GA optimization combined with K-Means clustering, scalable to more than 200 clients
Four-weighted GA with NSGA-II Efficiency and distance on a Pareto front
5 GA with load constraints, repairability, and penalty. Responds to both positive (delivery) and negative (pickup) requests


## 📊 Output and Evaluation

Every task prints:

The optimal route's overall mileage

Details of the load or segments

Pareto or statistical results (for multi-objective and stochastic jobs)

Every run produces a 300 dpi high-resolution PNG figure that summarizes the best path.



## 🧪 Recommended Testing

To examine, run each script several times using various random seeds:

Size of the population

Rate of mutation

Rate of crossover

The number of generations

This enables sensitivity analysis for the experimental results section of the report.

## 🧾 Citation References

Deb, K, et al. “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.” IEEE Transactions on Evolutionary Computation, 2017, www.semanticscholar.org/paper/A-fast-and-elitist-multiobjective-genetic-NSGA-II-Deb-Agrawal/6eddc19efa13f7e70301908d98e85a19d6f32a02.

fmder. “Home -             VRP-REP: The Vehicle Routing Problem Repository.” Vrp-Rep.org, 2025, vrp-rep.org/. Accessed 22 Oct. 2025.

Prins, Christian. “A Simple and Effective Evolutionary Algorithm for the Vehicle Routing Problem.” Computers & Operations Research, vol. 31, no. 12, Oct. 2004, pp. 1985–2002, https://doi.org/10.1016/s0305-0548(03)00158-8.

DEAP Evolutionary Algorithm Library — https://github.com/DEAP/deap
