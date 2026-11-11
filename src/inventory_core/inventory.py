"""Analytical safety stock, reorder point, EOQ, stockout risk and inventory status classification.

Safety stock (assignment formula):  SS  = Z * sqrt( sigma_D^2 * L  +  D_bar^2 * sigma_L^2 )
Reorder point:                      ROP = D_bar * L + SS
All quantities use one time unit (here: WEEKS). Assumes demand per period i.i.d., lead time
independent of demand, and lead-time demand approximately Normal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

STATUS_ORDER = ["Critical Stockout Risk", "Reorder Now", "Monitor", "Healthy"]
STATUS_COLOR = {"Healthy": "#10B981", "Monitor": "#F59E0B", "Reorder Now": "#F97316",
                "Critical Stockout Risk": "#EF4444"}


def z_value(service_level: float) -> float:
    return float(norm.ppf(service_level))


def sigma_ltd(d_bar: float, sd_d: float, L: float, sd_l: float) -> float:
    return float(np.sqrt(sd_d ** 2 * L + d_bar ** 2 * sd_l ** 2))


def safety_stock(service_level: float, d_bar: float, sd_d: float, L: float, sd_l: float) -> float:
    return z_value(service_level) * sigma_ltd(d_bar, sd_d, L, sd_l)


def reorder_point(service_level: float, d_bar: float, sd_d: float, L: float, sd_l: float) -> float:
    return d_bar * L + safety_stock(service_level, d_bar, sd_d, L, sd_l)


def ss_components(service_level, d_bar, sd_d, L, sd_l) -> dict:
    """Decompose variance of lead-time demand into demand-driven and lead-time-driven parts."""
    vd, vl = sd_d ** 2 * L, d_bar ** 2 * sd_l ** 2
    tot = vd + vl
    z = z_value(service_level)
    return {"z": z, "var_demand": vd, "var_leadtime": vl, "sigma_ltd": np.sqrt(tot),
            "share_demand": vd / tot if tot else 0, "share_leadtime": vl / tot if tot else 0,
            "ss": z * np.sqrt(tot), "ltd_mean": d_bar * L, "rop": d_bar * L + z * np.sqrt(tot)}


def eoq(annual_demand: float, order_cost: float, holding_cost_unit_year: float) -> float:
    if holding_cost_unit_year <= 0:
        return float(annual_demand)
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_unit_year))


def normal_loss(k: float) -> float:
    """Standard normal loss function G(k) = E[(Z-k)+]."""
    return float(norm.pdf(k) - k * (1 - norm.cdf(k)))


def stockout_risk(inv_position: float, d_bar, sd_d, L, sd_l) -> float:
    """P(lead-time demand > inventory position) under the Normal approximation."""
    s = sigma_ltd(d_bar, sd_d, L, sd_l)
    if s <= 0:
        return float(inv_position < d_bar * L)
    return float(1 - norm.cdf((inv_position - d_bar * L) / s))


def fill_rate(service_level, d_bar, sd_d, L, sd_l, Q) -> float:
    """Approximate fill rate (beta service) implied by a cycle-service (alpha) target."""
    s = sigma_ltd(d_bar, sd_d, L, sd_l)
    esc = s * normal_loss(z_value(service_level))
    return float(max(0.0, 1 - esc / Q)) if Q > 0 else 0.0


def classify(inv_position: float, ss: float, rop: float, ltd_mean: float) -> str:
    if inv_position < ltd_mean:
        return "Critical Stockout Risk"   # cannot cover expected lead-time demand
    if inv_position <= rop:
        return "Reorder Now"
    if inv_position <= rop + 0.5 * ss + 0.15 * ltd_mean:
        return "Monitor"
    return "Healthy"


def action(status: str, rop: float, ip: float, Q: float) -> str:
    if status == "Critical Stockout Risk":
        return f"Expedite order of {Q:,.0f} u now; consider emergency supplier"
    if status == "Reorder Now":
        return f"Place order of {Q:,.0f} u today"
    if status == "Monitor":
        return f"Review in 1 week — {ip - rop:,.0f} u above ROP"
    return "No action — stock above reorder point"


def simulated_on_hand(skus, rops, seed: int = 7) -> dict:
    """SIMULATED scenario stock levels — the source dataset contains no inventory records.
    Generated as ROP x U(0.35, 2.2) with a fixed seed so the demo is reproducible."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.35, 2.2, size=len(skus))
    return {s: float(np.round(r * k)) for s, r, k in zip(skus, rops, f)}
