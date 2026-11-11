"""Monte Carlo simulation of lead-time demand (LTD) with stochastic demand and lead times,
optional dependence between them (Gaussian copula), risk scenarios and buffer requirements."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

DEMAND_DISTS = ["Normal (truncated at 0)", "Gamma", "Lognormal", "Poisson", "Empirical (bootstrap)"]
LEAD_DISTS = ["Normal (truncated)", "Gamma", "Lognormal", "Uniform", "Triangular (right-skewed)"]


def demand_sampler(dist: str, mean: float, sd: float, rng, empirical=None):
    mean, sd = max(mean, 1e-6), max(sd, 1e-6)
    if dist.startswith("Normal"):
        return lambda size: np.clip(rng.normal(mean, sd, size), 0, None)
    if dist == "Gamma":
        k, th = (mean / sd) ** 2, sd ** 2 / mean
        return lambda size: rng.gamma(k, th, size)
    if dist == "Lognormal":
        s2 = np.log(1 + sd ** 2 / mean ** 2)
        return lambda size: rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), size)
    if dist == "Poisson":
        return lambda size: rng.poisson(mean, size).astype(float)
    if dist.startswith("Empirical"):
        arr = np.asarray(empirical if empirical is not None else [mean], float)
        return lambda size: rng.choice(arr, size=size, replace=True)
    raise ValueError(dist)


def lead_time_dist(dist: str, mean: float, sd: float):
    """Frozen scipy distribution for lead time (in periods)."""
    mean, sd = max(mean, 1e-3), max(sd, 1e-6)
    if dist.startswith("Normal"):
        a = (0.05 * mean - mean) / sd
        return stats.truncnorm(a, np.inf, loc=mean, scale=sd)
    if dist == "Gamma":
        return stats.gamma((mean / sd) ** 2, scale=sd ** 2 / mean)
    if dist == "Lognormal":
        s2 = np.log(1 + sd ** 2 / mean ** 2)
        return stats.lognorm(np.sqrt(s2), scale=np.exp(np.log(mean) - s2 / 2))
    if dist == "Uniform":
        lo = max(0.0, mean - np.sqrt(3) * sd)
        return stats.uniform(lo, 2 * (mean - lo))
    if dist.startswith("Triangular"):
        # right-skewed (mode at 25% of range -> long tail of late deliveries); mean and sd matched exactly
        c = 0.25
        w = sd * np.sqrt(18 / (1 - c + c ** 2))
        lo = max(0.0, mean - (1 + c) * w / 3)
        w = 3 * (mean - lo) / (1 + c) if lo == 0.0 else w
        return stats.triang(c, loc=lo, scale=w)
    raise ValueError(dist)


def simulate_ltd(n: int, d_dist: str, d_mean: float, d_sd: float, l_dist: str, l_mean: float, l_sd: float,
                 rho: float = 0.0, seed: int = 42, empirical=None) -> pd.DataFrame:
    """Simulate lead-time demand = sum of per-period demand over a random (fractional) lead time.
    rho > 0 links lead time to demand intensity (busy periods -> slower suppliers)."""
    rng = np.random.default_rng(seed)
    ld = lead_time_dist(l_dist, l_mean, l_sd)
    lmax = int(np.ceil(min(ld.ppf(0.99999), l_mean + 12 * l_sd, 200))) + 1
    D = demand_sampler(d_dist, d_mean, d_sd, rng, empirical)((n, lmax))
    if abs(rho) > 1e-9:
        zd = (D.mean(axis=1) - D.mean()) / max(D.mean(axis=1).std(), 1e-9)
        z = rho * zd + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        u = stats.norm.cdf(z)
    else:
        u = rng.uniform(size=n)
    L = np.clip(ld.ppf(np.clip(u, 1e-9, 1 - 1e-9)), 0, lmax - 1e-9)
    full = np.floor(L).astype(int)
    frac = L - full
    cols = np.arange(lmax)[None, :]
    ltd = (D * (cols < full[:, None])).sum(axis=1) + frac * D[np.arange(n), full]
    return pd.DataFrame({"lead_time": L, "ltd": ltd})


def summarize(ltd: np.ndarray, service: float, rop: float, Q: float) -> dict:
    ltd = np.asarray(ltd)
    mean = float(ltd.mean())
    rop_sim = float(np.quantile(ltd, service))
    esc = float(np.mean(np.clip(ltd - rop, 0, None)))
    return {
        "mean_ltd": mean, "sd_ltd": float(ltd.std(ddof=1)), "p95_ltd": float(np.quantile(ltd, 0.95)),
        "p99_ltd": float(np.quantile(ltd, 0.99)),
        "rop_sim": rop_sim, "ss_sim": rop_sim - mean,
        "stockout_prob": float(np.mean(ltd > rop)),         # 1 - cycle service level (alpha)
        "cycle_service": float(np.mean(ltd <= rop)),
        "expected_shortage": esc,                            # units short per replenishment cycle
        "fill_rate": float(max(0.0, 1 - esc / Q)) if Q > 0 else np.nan,   # beta service
        "skew": float(stats.skew(ltd)),
    }


def scenario_set(d_mean, d_sd, l_mean, l_sd, spike=0.30, delay=0.50, vol=1.0) -> dict:
    return {
        "Baseline": (d_mean, d_sd, l_mean, l_sd),
        f"Demand spike (+{spike:.0%})": (d_mean * (1 + spike), d_sd * (1 + spike), l_mean, l_sd),
        f"Supplier delay (+{delay:.0%} LT)": (d_mean, d_sd, l_mean * (1 + delay), l_sd * 2),
        f"High variability (σ×{1 + vol:.1f})": (d_mean, d_sd * (1 + vol), l_mean, l_sd * (1 + vol)),
        "Combined stress": (d_mean * (1 + spike), d_sd * (1 + vol), l_mean * (1 + delay), l_sd * 2),
    }
