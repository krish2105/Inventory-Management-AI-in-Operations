"""AI automation layer: 'before AI' vs 'after AI' replay on real demand, forecast-drift monitoring,
ML anomaly detection, ML SKU segmentation, an autonomous replenishment agent, portfolio-wide
Bayesian policy tuning and an auto-generated exception digest.

Before AI (classic textbook practice)            After AI (this system)
- static mean & sigma of all history             - forecast refit every week on new data (continuous learning)
- one service level for every SKU                - service level set per ML segment (ABC-XYZ + K-means)
- formula buffer, set once, never revised        - buffer from live forecast-error sigma, recomputed weekly
- no monitoring                                  - tracking-signal drift detection triggers model re-selection
- manual what-if sweeps                          - Bayesian optimisation of (s, Q)
"""
from __future__ import annotations
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from . import data as D, forecasting as F, inventory as I, simulation as S, optimization as O

SEG_SERVICE = {"AX": 0.98, "AY": 0.97, "AZ": 0.95, "BX": 0.96, "BY": 0.95, "BZ": 0.93,
               "CX": 0.94, "CY": 0.92, "CZ": 0.90}


# ------------------------------------------------------------------ segmentation (ABC-XYZ + K-means)
def segment(df: pd.DataFrame, weeks: int = 52, k: int = 4, seed: int = 0) -> pd.DataFrame:
    rows = []
    for sku, g in df.groupby("sku"):
        y = D.series(df, sku, "W").tail(weeks)
        if len(y) < 8:
            continue
        x = np.arange(len(y))
        slope = np.polyfit(x, y.values, 1)[0] / max(y.mean(), 1e-9)
        m = y.groupby(y.index.month).mean()
        rows.append({"sku": sku, "description": g["description"].iloc[0] if "description" in g else sku,
                     "revenue": float(g.loc[g.date >= g.date.max() - pd.Timedelta(weeks=weeks), "revenue"].sum())
                     if "revenue" in g else float(y.sum()),
                     "mean_wk": y.mean(), "cv": y.std() / max(y.mean(), 1e-9),
                     "trend_pct_wk": 100 * slope, "seasonality": m.std() / max(m.mean(), 1e-9),
                     "zero_weeks_pct": 100 * (y == 0).mean()})
    s = pd.DataFrame(rows).sort_values("revenue", ascending=False)
    cum = s["revenue"].cumsum() / s["revenue"].sum()
    s["ABC"] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    s["XYZ"] = np.where(s["cv"] <= 0.5, "X", np.where(s["cv"] <= 1.0, "Y", "Z"))
    s["segment"] = s["ABC"] + s["XYZ"]
    s["ai_service"] = s["segment"].map(SEG_SERVICE)
    feats = np.c_[np.log1p(s["mean_wk"]), s["cv"], s["trend_pct_wk"], s["seasonality"], s["zero_weeks_pct"]]
    Z = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=min(k, len(s)), n_init=10, random_state=seed).fit(Z)
    s["cluster"] = km.labels_
    names = {}
    cen = pd.DataFrame(StandardScaler().fit(feats).inverse_transform(km.cluster_centers_),
                       columns=["logmean", "cv", "trend", "seas", "zero"])
    for c, r in cen.iterrows():
        tags = []
        tags.append("High-volume" if r.logmean > np.log1p(s["mean_wk"]).median() else "Low-volume")
        tags.append("volatile" if r.cv > s["cv"].median() else "stable")
        if r.seas > s["seasonality"].quantile(0.6):
            tags.append("seasonal")
        if r.trend > 0.3:
            tags.append("growing")
        elif r.trend < -0.3:
            tags.append("declining")
        names[c] = " · ".join(tags)
    s["cluster_name"] = s["cluster"].map(names)
    return s.reset_index(drop=True)


# ------------------------------------------------------------------ anomaly detection
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Isolation Forest over weekly demand features of every SKU (level shift, spike, drop)."""
    rows = []
    for sku in df["sku"].unique():
        y = D.series(df, sku, "W")
        med = y.rolling(8, min_periods=4).median().shift(1)
        mad = (y - med).abs().rolling(8, min_periods=4).median().shift(1) + 1e-9
        f = pd.DataFrame({"sku": sku, "week": y.index, "demand": y.values,
                          "robust_z": ((y - med) / (1.4826 * mad)).values,
                          "ratio": (y / med.replace(0, np.nan)).values,
                          "chg": y.pct_change().replace([np.inf, -np.inf], np.nan).values})
        rows.append(f.dropna())
    A = pd.concat(rows, ignore_index=True)
    X = np.c_[np.clip(A["robust_z"], -20, 20), np.log1p(A["ratio"].clip(0, 50)), np.clip(A["chg"], -5, 20)]
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X)
    A["score"] = -iso.score_samples(X)
    A["anomaly"] = iso.predict(X) == -1
    A["type"] = np.where(A["robust_z"] > 0, "Spike", "Drop")
    return A


# ------------------------------------------------------------------ drift (tracking signal)
def tracking_signal(errors) -> float:
    e = np.asarray(errors, float)
    mad = np.mean(np.abs(e)) if len(e) else 0
    return float(np.sum(e) / mad) if mad > 0 else 0.0


# ------------------------------------------------------------------ rolling AI forecasts (continuous learning)
def _gbm_forecast(y: pd.Series, h: int):
    return F.fit_predict("Gradient Boosting (ML)", y, h, "W")


def rolling_ai_forecasts(df: pd.DataFrame, sku: str, holdout_weeks: int = 26, horizon: int = 4,
                         retrain_every: int = 4, ts_limit: float = 4.0, cooldown: int = 3) -> tuple[pd.DataFrame, list]:
    """Walk forward through the hold-out one week at a time, as the system would in production.
    Every week: ETS refit on all data so far (continuous updating). Every `retrain_every` weeks or on a
    drift alarm (|tracking signal| > ts_limit): champion/challenger re-selection between ETS and the
    gradient-boosting model on the last 8 weeks of one-step errors."""
    y = D.series(df, sku, "W")
    start = len(y) - holdout_weeks
    out, log, champion = [], [], "ETS (Holt-Winters)"
    errs = {"ETS (Holt-Winters)": [], "Gradient Boosting (ML)": []}
    last_retrain = -999
    for t in range(start, len(y)):
        hist = y.iloc[:t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_ets = F.fit_predict("ETS (Holt-Winters)", hist, horizon, "W")
            p_gbm = _gbm_forecast(hist, horizon) if (t - start) % 2 == 0 or champion != "ETS (Holt-Winters)" else None
        actual = y.iloc[t]
        errs["ETS (Holt-Winters)"].append(actual - p_ets[0])
        if p_gbm is not None:
            errs["Gradient Boosting (ML)"].append(actual - p_gbm[0])
        ch_err = errs[champion][-8:]
        ts = tracking_signal(ch_err)
        reason = None
        if t - last_retrain >= retrain_every:
            reason = "scheduled retrain"
        if abs(ts) > ts_limit and len(ch_err) >= 6 and t - last_retrain >= cooldown:
            reason = f"drift alarm (TS = {ts:+.1f})"
        if reason:
            scores = {m: np.mean(np.abs(e[-8:])) for m, e in errs.items() if len(e) >= 3}
            new = min(scores, key=scores.get) if scores else champion
            errs = {m: e for m, e in errs.items()}
            log.append({"week": y.index[t].date(), "sku": sku, "event": reason,
                        "champion": new, "switched": new != champion,
                        "mae_ets": scores.get("ETS (Holt-Winters)", np.nan), "mae_gbm": scores.get("Gradient Boosting (ML)", np.nan)})
            champion, last_retrain = new, t
        pred = p_ets if champion == "ETS (Holt-Winters)" or p_gbm is None else p_gbm
        e_hist = np.array(errs[champion][-13:])
        sd = float(np.sqrt(np.mean(e_hist ** 2))) if len(e_hist) >= 4 else float(hist.tail(26).std())
        out.append({"week": y.index[t], "sku": sku, "actual": actual, "forecast": float(np.mean(pred)),
                    "forecast_1": float(pred[0]), "err_sd": sd, "model": champion, "tracking_signal": ts})
    return pd.DataFrame(out), log


def rolling_ai_all(df, holdout_weeks=26, progress=None):
    frames, logs = [], []
    skus = df["sku"].unique()
    for i, s in enumerate(skus):
        f, l = rolling_ai_forecasts(df, s, holdout_weeks)
        frames.append(f); logs += l
        if progress:
            progress(i + 1, len(skus), s)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(logs)


# ------------------------------------------------------------------ replay on REAL daily demand
def replay(daily: np.ndarray, s_weekly: np.ndarray, Q_weekly: np.ndarray, lt_days: np.ndarray, cp: O.CostParams):
    """Run an (s, Q) policy over the actual daily demand of the hold-out. s and Q may change every week
    (dynamic policy) or stay fixed (static policy). Lead times are pre-drawn so both policies face the
    same supplier behaviour. Lost sales."""
    T = len(daily)
    on_hand = float(s_weekly[0] + Q_weekly[0] * 0.5)
    pipeline = np.zeros(T + 200)
    on_order = 0.0
    hold = short = orders = dem = 0.0
    k = 0
    stockout_days = 0
    inv_path = np.zeros(T)
    for t in range(T):
        a = pipeline[t]; on_hand += a; on_order -= a
        d = daily[t]
        sold = min(on_hand, d); on_hand -= sold
        short += d - sold; dem += d
        stockout_days += (d - sold) > 1e-9
        w = min(t // 7, len(s_weekly) - 1)
        s, Q = s_weekly[w], max(Q_weekly[w], 1.0)
        ip = on_hand + on_order
        if ip <= s:
            n = np.floor((s - ip) / Q) + 1
            lt = int(lt_days[k % len(lt_days)]); k += 1
            pipeline[t + lt] += n * Q; on_order += n * Q; orders += 1
        hold += on_hand
        inv_path[t] = on_hand
    yrs = T / 365
    avg = hold / T
    c_hold = avg * cp.unit_cost * cp.holding_rate
    return {"fill_rate": 1 - short / max(dem, 1e-9), "stockout_days": int(stockout_days), "units_short": short,
            "avg_inventory": avg, "orders": orders, "holding_cost": c_hold / 1 * yrs,
            "ordering_cost": orders * cp.order_cost, "shortage_cost": short * cp.shortage_cost,
            "total_cost": c_hold * yrs + orders * cp.order_cost + short * cp.shortage_cost,
            "inv_path": inv_path, "inv_value": avg * cp.unit_cost}


def before_after(df, roll: pd.DataFrame, service: float, lt_mean_d: float, lt_sd_d: float,
                 cost_pct=0.6, holding_rate=0.25, order_cost=40.0, short_mult=1.0, seg: pd.DataFrame | None = None,
                 seed: int = 11, holdout_weeks: int = 26) -> pd.DataFrame:
    rows, paths = [], {}
    rng = np.random.default_rng(seed)
    ld = S.lead_time_dist("Gamma", lt_mean_d, max(lt_sd_d, 1e-3))
    L, sL = lt_mean_d / 7, lt_sd_d / 7
    for sku, r in roll.groupby("sku"):
        y = D.series(df, sku, "W")
        train = y.iloc[: len(y) - holdout_weeks]
        start = r["week"].min() - pd.Timedelta(days=6)
        g = df[df.sku == sku].set_index("date")["demand"].resample("D").sum()
        daily = g[(g.index >= start) & (g.index <= r["week"].max())].values
        price = df.loc[df.sku == sku, "avg_price"].mean() if "avg_price" in df else 1.0
        cp = O.CostParams(unit_cost=price * cost_pct, holding_rate=holding_rate, order_cost=order_cost,
                          shortage_cost=price * short_mult)
        lts = np.maximum(1, np.rint(ld.rvs(size=400, random_state=rng))).astype(int)
        h_unit = cp.unit_cost * holding_rate
        # BEFORE AI: static textbook policy from historical statistics, single service level
        d0, s0 = train.mean(), train.std()
        s_b = d0 * L + I.z_value(service) * np.sqrt(s0 ** 2 * L + d0 ** 2 * sL ** 2)
        q_b = max(1.0, I.eoq(d0 * 52, order_cost, h_unit))
        nW = len(r)
        B = replay(daily, np.full(nW, s_b), np.full(nW, q_b), lts, cp)
        # AFTER AI: (1) weekly re-forecast with drift-triggered model re-selection (rolling_ai_forecasts)
        #           (2) self-correction: recent positive forecast bias is added back (tracking-signal style)
        #           (3) cost-aware service level: lost-sales critical ratio 1 - hQ/(pD), floored at the ML-segment alpha
        #           (4) conformal buffer: alpha-quantile of the system's OWN realised lead-time forecast errors
        a_floor = float(seg.set_index("sku").loc[sku, "ai_service"]) if seg is not None and sku in set(seg["sku"]) else service
        f, sd, act = r["forecast"].values, r["err_sd"].values, r["actual"].values
        e1 = act - r["forecast_1"].values
        bias = pd.Series(e1).shift(1).rolling(6, min_periods=2).mean().fillna(0).clip(lower=0).values
        fa = f + bias
        q_a = np.maximum(1.0, np.sqrt(2 * fa * 52 * order_cost / max(h_unit, 1e-9)))
        alpha = np.clip(1 - h_unit * q_a / (cp.shortage_cost * np.maximum(fa, 1e-9) * 52), a_floor, 0.995)
        Lr = max(1, int(round(L)))
        n = len(fa)
        res = np.array([act[k:k + Lr].sum() - fa[k] * Lr if k + Lr <= n else np.nan for k in range(n)])
        s_a = np.zeros(n)
        for t in range(n):
            known = res[:max(0, t - Lr + 1)]
            known = known[~np.isnan(known)][-26:]
            z = I.z_value(alpha[t])
            q_norm = z * sd[t] * np.sqrt(L)
            q = max(np.quantile(known, alpha[t]), q_norm) if len(known) >= 6 else 1.3 * q_norm
            s_a[t] = fa[t] * L + np.sqrt(q ** 2 + (z * fa[t] * sL) ** 2)
        svc_ai = float(alpha.mean())
        A = replay(daily, s_a, q_a, lts, cp)
        paths[sku] = {"dates": pd.date_range(start, periods=len(daily), freq="D"), "daily": daily,
                      "before": B.pop("inv_path"), "after": A.pop("inv_path"), "s_before": s_b, "s_after": s_a,
                      "weeks": r["week"].values}
        rows.append({"sku": sku, "ai_service": svc_ai,
                     **{f"before_{k}": v for k, v in B.items()}, **{f"after_{k}": v for k, v in A.items()},
                     "wape_before": 100 * np.abs(r["actual"] - d0).sum() / max(r["actual"].sum(), 1e-9),
                     "wape_after": 100 * np.abs(r["actual"] - r["forecast_1"]).sum() / max(r["actual"].sum(), 1e-9)})
    return pd.DataFrame(rows), paths


# ------------------------------------------------------------------ portfolio Bayesian tuning
def tune_portfolio(P: pd.DataFrame, L_wk, sdL_wk, holding_rate, order_cost, short_mult, target_fill=0.95,
                   trials=20, reps=60, seed=0, progress=None) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(P.itertuples()):
        cp = O.CostParams(unit_cost=r.unit_cost, holding_rate=holding_rate, order_cost=order_cost,
                          shortage_cost=r.price * short_mult, target_fill=target_fill, moq=1)
        sim = O.PolicySimulator(r.d_bar, max(r.sd_d, 1e-6), L_wk, max(sdL_wk, 1e-3), reps=reps, days=182, seed=seed + i)
        base = sim.run(r.rop, max(r.eoq, 1), cp)
        res = O.tpe_bayes_opt(sim, cp, {"s": (0, max(3 * r.rop, 10)), "Q": (1, max(4 * r.eoq, 10))},
                              n_trials=trials, seed=seed, penalty_scale=base["total_cost"])
        b = res["best"]
        rows.append({"sku": r.sku, "description": r.description, "s_formula": r.rop, "Q_eoq": r.eoq,
                     "s_opt": b["s"], "Q_opt": b["Q"], "cost_formula": base["total_cost"], "cost_opt": b["total_cost"],
                     "fill_formula": base["fill_rate"], "fill_opt": b["fill_rate"], "feasible": b["feasible"]})
        if progress:
            progress(i + 1, len(P), r.sku)
    out = pd.DataFrame(rows)
    out["saving_pct"] = 100 * (1 - out["cost_opt"] / out["cost_formula"])
    return out


# ------------------------------------------------------------------ autonomous replenishment agent
def agent_decisions(P: pd.DataFrame, L_wk: float, sdL_wk: float, n_mc: int = 5000, seed: int = 3,
                    today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list]:
    """Perceive → reason → act loop over every SKU. Uses a Monte Carlo (Gamma) lead-time-demand quantile
    as the buffer so skew is respected, then drafts purchase orders with a human-readable rationale."""
    log = [{"step": "Perceive", "detail": f"Loaded {len(P)} SKUs: forecast, forecast-error σ, stock position, ML segment"}]
    rng_seed = seed
    today = today or pd.Timestamp.today().normalize()
    recs = []
    for i, r in enumerate(P.itertuples()):
        mc = S.simulate_ltd(n_mc, "Gamma", max(r.d_bar, 1e-6), max(r.sd_d, 1e-6), "Gamma", L_wk, max(sdL_wk, 1e-3),
                            0.0, rng_seed + i)["ltd"].values
        svc = getattr(r, "svc", 0.95)
        rop_mc = float(np.quantile(mc, svc))
        p_so = float(np.mean(mc > r.inv_position))
        days_cover = r.on_hand / max(r.d_bar / 7, 1e-9)
        if r.inv_position < np.mean(mc):
            act, urg = "EXPEDITE", 3
        elif r.inv_position <= rop_mc:
            act, urg = "ORDER", 2
        elif r.inv_position <= rop_mc + 0.25 * r.eoq:
            act, urg = "WATCH", 1
        elif days_cover > 84:
            act, urg = "HOLD · overstock", 0
        else:
            act, urg = "OK", 0
        qty = 0.0
        if act in ("EXPEDITE", "ORDER"):
            qty = (np.floor((rop_mc - r.inv_position) / max(r.eoq, 1)) + 1) * r.eoq
        conf = max(0.0, min(1.0, 1 - r.fc_wape / 100))
        why = (f"Position {r.inv_position:,.0f} vs MC reorder point {rop_mc:,.0f} (α {svc:.0%}, segment {getattr(r, 'segment', '—')}); "
               f"P(stockout in lead time) {p_so:.0%}; {days_cover:.0f} days of cover.")
        recs.append({"sku": r.sku, "description": r.description, "segment": getattr(r, "segment", ""), "service": svc,
                     "action": act, "urgency": urg, "order_qty": round(qty), "rop_mc": rop_mc, "rop_formula": r.rop,
                     "p_stockout": p_so, "days_cover": days_cover, "confidence": conf,
                     "po_value": qty * r.unit_cost, "eta": (today + pd.Timedelta(days=round(L_wk * 7))).date() if qty else None,
                     "rationale": why})
    R = pd.DataFrame(recs).sort_values(["urgency", "p_stockout"], ascending=False)
    n_po = int((R["order_qty"] > 0).sum())
    log += [{"step": "Reason", "detail": f"Simulated {n_mc:,} lead-time scenarios per SKU ({n_mc * len(P):,} total) to size buffers"},
            {"step": "Decide", "detail": f"{(R.action == 'EXPEDITE').sum()} expedite · {(R.action == 'ORDER').sum()} order · "
                                          f"{(R.action == 'WATCH').sum()} watch · {R.action.str.startswith('HOLD').sum()} overstock"},
            {"step": "Act", "detail": f"Drafted {n_po} purchase orders worth £{R['po_value'].sum():,.0f} (awaiting human approval)"}]
    for l in log:
        l["time"] = datetime.now().strftime("%H:%M:%S")
    return R, log


def digest(R: pd.DataFrame, anomalies: pd.DataFrame, drift_log: pd.DataFrame, seg: pd.DataFrame, when: str) -> str:
    ex = R[R.action.isin(["EXPEDITE", "ORDER"])]
    lines = [f"# AI Inventory Exception Digest — {when}", "",
             f"**{len(ex)} SKUs need replenishment** (£{ex['po_value'].sum():,.0f} of draft POs); "
             f"{(R.action == 'EXPEDITE').sum()} are critical.", ""]
    if len(ex):
        lines.append("## Priority actions")
        for r in ex.head(8).itertuples():
            lines.append(f"- **{r.action}** {r.sku} {r.description[:30]} — order {r.order_qty:,.0f} u; {r.rationale}")
    recent = anomalies[anomalies.anomaly].sort_values("week").groupby("sku").tail(1)
    recent = recent[recent.week >= anomalies.week.max() - pd.Timedelta(weeks=4)]
    lines += ["", "## Demand anomalies (last 4 weeks, Isolation Forest)"]
    lines += [f"- {r.type} on {r.sku}: {r.demand:,.0f} u in week {r.week:%d %b %Y} (robust z = {r.robust_z:+.1f})"
              for r in recent.itertuples()] or ["- None detected"]
    if drift_log is not None and len(drift_log):
        dr = drift_log[drift_log.event.str.startswith("drift")]
        lines += ["", "## Model drift & auto-retraining",
                  f"- {len(dr)} drift alarms across {dr.sku.nunique()} SKUs triggered automatic model re-selection; "
                  f"{int(drift_log.switched.sum())} champion switches."]
    over = R[R.action.str.startswith("HOLD")]
    if len(over):
        lines += ["", "## Overstock", f"- {len(over)} SKUs have > 12 weeks of cover: {', '.join(over.sku.head(6))}"]
    lines += ["", "## Segment policy", "- Service levels auto-assigned by ABC-XYZ segment: " +
              ", ".join(f"{k} {v:.0%}" for k, v in SEG_SERVICE.items() if k in set(seg.segment))]
    return "\n".join(lines)
