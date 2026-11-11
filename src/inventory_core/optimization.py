"""Bayesian optimisation of the (s, Q) continuous-review reorder policy.

Decision variables : s = reorder point (units), Q = order quantity (units)
Objective (minimise): expected annual cost = ordering + holding + shortage (lost-sales penalty)
Constraints        : fill rate >= target, peak on-hand <= warehouse capacity,
                     average inventory value <= budget, Q >= MOQ (search bound)
Each candidate is evaluated by a daily stochastic inventory simulation under COMMON RANDOM
NUMBERS (identical demand / lead-time draws for every candidate) so comparisons are fair.
The chosen policy is then re-validated on an independent random stream.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .simulation import lead_time_dist


@dataclass
class CostParams:
    unit_cost: float = 1.0           # GBP per unit (purchase cost)
    holding_rate: float = 0.25       # annual holding cost as fraction of unit cost
    order_cost: float = 50.0         # GBP per purchase order
    shortage_cost: float = 2.0       # GBP per unit of lost sale
    target_fill: float = 0.95
    capacity: float = 1e9            # max units on hand (warehouse slots for the SKU)
    budget: float = 1e12             # max average inventory value (GBP)
    moq: float = 1.0


class PolicySimulator:
    def __init__(self, wk_mean, wk_sd, lt_mean_wk, lt_sd_wk, lt_dist="Gamma", days=364, reps=150,
                 warmup=28, seed=2025):
        self.days, self.reps, self.warmup = days, reps, warmup
        rng = np.random.default_rng(seed)
        m, s = max(wk_mean / 7, 1e-6), max(wk_sd / np.sqrt(7), 1e-6)    # weekly -> daily (i.i.d. days)
        k, th = (m / s) ** 2, s ** 2 / m
        self.D = rng.gamma(k, th, size=(reps, days + warmup))           # common random numbers
        ld = lead_time_dist(lt_dist, lt_mean_wk * 7, max(lt_sd_wk * 7, 1e-3))
        self.LT = np.maximum(1, np.rint(ld.rvs(size=(reps, 400), random_state=rng))).astype(int)
        self.maxlt = int(self.LT.max()) + 2
        self.daily_mean = m

    def run(self, s: float, Q: float, cp: CostParams) -> dict:
        R, T, W = self.reps, self.days + self.warmup, self.warmup
        on_hand = np.full(R, s + Q, float)
        on_order = np.zeros(R)
        arrivals = np.zeros((R, T + self.maxlt + 1))
        n_ord = np.zeros(R, int)
        hold = short = dem = orders = 0.0
        peak = np.zeros(R)
        rows = np.arange(R)
        for t in range(T):
            a = arrivals[:, t]
            on_hand += a
            on_order -= a
            d = self.D[:, t]
            sold = np.minimum(on_hand, d)
            on_hand -= sold
            ip = on_hand + on_order
            need = ip <= s
            if need.any():
                r = rows[need]
                n = np.floor((s - ip[need]) / Q) + 1
                lt = self.LT[r, n_ord[r] % self.LT.shape[1]]
                np.add.at(arrivals, (r, t + lt), n * Q)
                on_order[r] += n * Q
                n_ord[r] += 1
                if t >= W:
                    orders += need.sum()
            if t >= W:
                hold += on_hand.sum()
                short += (d - sold).sum()
                dem += d.sum()
                peak = np.maximum(peak, on_hand + sold)  # stock level after receipts, before sales
        scale = 365 / (self.days * R)
        avg_oh = hold / (self.days * R)
        c_hold = avg_oh * cp.unit_cost * cp.holding_rate                  # annual holding cost
        c_ord = orders * cp.order_cost * scale
        c_short = short * cp.shortage_cost * scale
        fill = 1 - short / max(dem, 1e-9)
        peak95 = float(np.quantile(peak, 0.95))
        avg_val = avg_oh * cp.unit_cost
        viol = {"fill": max(0.0, cp.target_fill - fill) / max(cp.target_fill, 1e-9),
                "capacity": max(0.0, peak95 - cp.capacity) / max(cp.capacity, 1),
                "budget": max(0.0, avg_val - cp.budget) / max(cp.budget, 1)}
        return {"s": s, "Q": Q, "total_cost": c_hold + c_ord + c_short, "holding_cost": c_hold,
                "ordering_cost": c_ord, "shortage_cost": c_short, "fill_rate": fill,
                "orders_per_year": orders * scale, "avg_on_hand": avg_oh, "peak_on_hand_p95": peak95,
                "avg_inventory_value": avg_val, "units_short_per_year": short * scale,
                "feasible": all(v <= 1e-9 for v in viol.values()),
                "viol_fill": viol["fill"], "viol_capacity": viol["capacity"], "viol_budget": viol["budget"]}


def penalised(res: dict, scale: float) -> float:
    v = 25 * res["viol_fill"] + 5 * res["viol_capacity"] + 5 * res["viol_budget"]
    return res["total_cost"] + scale * v + (scale * 0.5 if v > 0 else 0)


def _ei(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def gp_bayes_opt(sim: PolicySimulator, cp: CostParams, bounds: dict, n_init=10, n_iter=30, seed=0,
                 penalty_scale=None, progress=None) -> dict:
    """Gaussian-process Bayesian optimisation with Expected Improvement (Matern-5/2 kernel)."""
    rng = np.random.default_rng(seed)
    lo = np.array([bounds["s"][0], bounds["Q"][0]], float)
    hi = np.array([bounds["s"][1], bounds["Q"][1]], float)
    to_x = lambda u: np.rint(lo + u * (hi - lo))
    U = qmc.Sobol(2, seed=seed).random(n_init)
    penalty_scale = penalty_scale or 1.0
    hist, Y = [], []
    t0 = time.time()

    def evaluate(u, phase, ei=np.nan):
        s, Q = to_x(u)
        Q = max(Q, cp.moq)
        r = sim.run(s, Q, cp)
        y = penalised(r, penalty_scale)
        hist.append({"trial": len(hist) + 1, "phase": phase, **r, "objective": y, "ei": ei,
                     "u_s": u[0], "u_Q": u[1]})
        Y.append(y)
        if progress:
            progress(len(hist), n_init + n_iter, hist[-1])

    for u in U:
        evaluate(u, "Sobol init")
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(1e-3)
    gp = None
    for _ in range(n_iter):
        X = np.array([[h["u_s"], h["u_Q"]] for h in hist])
        y = np.array(Y)
        ys = (np.log(y) - np.log(y).mean()) / (np.log(y).std() + 1e-9)   # log-cost, standardised
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2,
                                      random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, ys)
        best_u = X[np.argmin(ys)]
        cand = np.vstack([rng.uniform(size=(3000, 2)),
                          np.clip(best_u + rng.normal(0, 0.05, size=(600, 2)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        ei = _ei(mu, sd, ys.min())
        i = int(np.argmax(ei))
        evaluate(cand[i], "GP-EI", float(ei[i]))
    df = pd.DataFrame(hist)
    df["best_so_far"] = df["objective"].cummin()
    feas = df[df["feasible"]]
    best = (feas if len(feas) else df).sort_values("total_cost" if len(feas) else "objective").iloc[0]
    # surrogate surface for visualisation
    surf = None
    if gp is not None:
        g = np.linspace(0, 1, 40)
        GS, GQ = np.meshgrid(g, g)
        mu = gp.predict(np.c_[GS.ravel(), GQ.ravel()])
        y = np.array(Y)
        mu = np.exp(mu * (np.log(y).std() + 1e-9) + np.log(y).mean())
        surf = {"s": lo[0] + g * (hi[0] - lo[0]), "Q": lo[1] + g * (hi[1] - lo[1]), "cost": mu.reshape(40, 40)}
    return {"trials": df, "best": best.to_dict(), "surface": surf, "seconds": time.time() - t0,
            "method": "Gaussian-process BO (Matern 5/2, Expected Improvement)"}


def tpe_bayes_opt(sim: PolicySimulator, cp: CostParams, bounds: dict, n_trials=40, seed=0,
                  penalty_scale=None, progress=None) -> dict:
    """Optuna Tree-structured Parzen Estimator (sequential model-based / Bayesian optimisation)
    with native constraint handling."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hist = []
    t0 = time.time()

    def objective(trial):
        s = trial.suggest_int("s", int(bounds["s"][0]), int(bounds["s"][1]))
        Q = trial.suggest_int("Q", int(max(bounds["Q"][0], cp.moq)), int(bounds["Q"][1]))
        r = sim.run(s, Q, cp)
        for key in ("fill", "capacity", "budget"):          # feasible when <= 0
            trial.set_constraint(key, r[f"viol_{key}"] - 1e-12)
        y = penalised(r, penalty_scale or 1.0)
        hist.append({"trial": len(hist) + 1, "phase": "TPE" if len(hist) >= 10 else "Random init", **r,
                     "objective": y, "ei": np.nan})
        if progress:
            progress(len(hist), n_trials, hist[-1])
        return r["total_cost"]

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    df = pd.DataFrame(hist)
    df["best_so_far"] = df["objective"].cummin()
    feas = df[df["feasible"]]
    best = (feas if len(feas) else df).sort_values("total_cost" if len(feas) else "objective").iloc[0]
    return {"trials": df, "best": best.to_dict(), "surface": None, "seconds": time.time() - t0,
            "method": "Optuna TPE (Bayesian, constraint-aware)"}
