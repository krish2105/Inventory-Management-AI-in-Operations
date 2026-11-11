"""AI-Powered Inventory Intelligence & Reorder Point Optimization — Streamlit dashboard.

AI in Operations · Session II Laboratory Exercise
Krishna Mathur · AS25DXB018 · MAIB September 2025 · SP Jain School of Global Management
Run:  streamlit run apps/streamlit/app.py
"""
from __future__ import annotations
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib  # noqa: E402
from inventory_core import data as D, forecasting as F, inventory as I, simulation as S, optimization as O, automation as A  # noqa: E402
import inventory_ui as ui  # noqa: E402  (renamed from ui.py so a redeploy can never reuse a stale module)

# Streamlit re-executes app.py on every rerun but keeps imported modules cached in the server process.
# After a redeploy (git push) the old ui/engine modules could stay in memory and no longer match app.py,
# so reload them whenever their source file changed since they were loaded.
for _m in (D, F, I, S, O, A, ui):
    try:
        _mt = Path(_m.__file__).stat().st_mtime
        if getattr(_m, "_loaded_mtime", None) != _mt:   # first sight in this process, or file changed
            importlib.reload(_m)
        _m._loaded_mtime = Path(_m.__file__).stat().st_mtime
    except Exception:
        pass

st.set_page_config(page_title="AI Inventory Intelligence | Krishna Mathur", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")

STUDENT = {"name": "Krishna Mathur", "id": "AS25DXB018", "prog": "MAIB", "batch": "September 2025",
           "school": "SP Jain School of Global Management", "course": "AI in Operations",
           "lab": "Session II – Laboratory Exercise", "instructor": "Dr. Sandip Kumar Roy"}
DEFAULTS = dict(theme_mode="Dark", service=0.95, lt_days=14.0, lt_sd_days=4.0, basis="AI forecast (dynamic)",
                window=52, cost_pct=0.60, holding_rate=0.25, order_cost=40.0, short_mult=1.0, stock_seed=7, ai_svc=True)

# ------------------------------------------------------------------ session state
ss = st.session_state
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)
if "df" not in ss:
    ss.df = D.load_default()
    ss.data_source = "UCI Online Retail II (processed: 40 SKUs, daily demand)"
    ss.last_ingest = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ss.ingest_log = [{"time": ss.last_ingest, "event": "Loaded default dataset", "rows": len(ss.df)}]
ss.setdefault("stock_edits", {})

# ------------------------------------------------------------------ theme
mode = ss.theme_mode
T = ui.THEMES[mode]
try:  # make native widgets follow the selected theme
    from streamlit import config as _cfg
    want = {"theme.base": mode.lower(), "theme.primaryColor": T["primary"], "theme.backgroundColor": T["bg"],
            "theme.secondaryBackgroundColor": T["surface"] if mode == "Light" else "#111A2E",
            "theme.textColor": T["text"]}
    if ss.get("_applied_theme") != mode:
        for k, v in want.items():
            _cfg.set_option(k, v)
        ss._applied_theme = mode
        st.rerun()
except Exception:
    pass
st.markdown(ui.css(T, mode), unsafe_allow_html=True)
SC = ui.status_colors(T)


def fig_(fig, h=360, legend=True, **kw):
    """Render a Plotly figure. The title is rendered as HTML above the chart so it can never collide with the legend."""
    title = fig.layout.title.text
    if title:
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
        fig.update_layout(title=None)
    return st.plotly_chart(ui.style_fig(fig, T, h, legend), width="stretch",
                           config={"displaylogo": False, "toImageButtonOptions": {"scale": 2}}, **kw)


def fmt(x, d=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{d}f}"


# ------------------------------------------------------------------ cached computations
@st.cache_data(show_spinner=False)
def portfolio_base(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, g in df.groupby("sku"):
        y = D.series(df, sku, "W")
        if len(y) < 20:
            continue
        q = F.quick_portfolio_forecast(y)
        p = g.sort_values("date")["avg_price"].tail(120).mean() if "avg_price" in g else np.nan
        rows.append({"sku": sku, "description": g["description"].dropna().iloc[0] if "description" in g and g["description"].notna().any() else sku,
                     "weeks": len(y), "price": p, "fc_mean": q["forecast_mean"], "fc_total4": q["forecast_total"],
                     "fc_err_sd": q["err_sd"], "fc_wape": q["wape"], "last_week": float(y.iloc[-1]),
                     **{f"hist_mean_{w}": float(y.tail(w).mean()) for w in (13, 26, 52, 104)},
                     **{f"hist_sd_{w}": float(y.tail(w).std()) for w in (13, 26, 52, 104)}})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_backtest(df, sku, freq, h, models, folds):
    y = D.series(df, sku, freq)
    pr = D.price_series(df, sku, freq, y.index)
    return y, pr, F.backtest(y, freq, h, pr, folds=folds, models=list(models))


@st.cache_data(show_spinner=False)
def run_forecast(df, sku, freq, h, model, resid_sd, level):
    y = D.series(df, sku, freq)
    pr = D.price_series(df, sku, freq, y.index)
    return F.forecast(y, freq, h, model, resid_sd, pr, level)


@st.cache_data(show_spinner=False)
def run_importance(df, sku, freq):
    y = D.series(df, sku, freq)
    return F.feature_importance(y, freq, D.price_series(df, sku, freq, y.index))


@st.cache_data(show_spinner=False)
def run_mc(n, dd, dm, dsd, ld, lm, lsd, rho, seed, emp):
    return S.simulate_ltd(n, dd, dm, dsd, ld, lm, lsd, rho, seed, np.array(emp) if emp is not None else None)


@st.cache_data(show_spinner=False)
def stl(y: pd.Series, period: int):
    from statsmodels.tsa.seasonal import STL
    r = STL(y, period=period, robust=True).fit()
    return pd.DataFrame({"observed": y, "trend": r.trend, "seasonal": r.seasonal, "resid": r.resid})


@st.cache_data(show_spinner=False)
def run_segment(df):
    return A.segment(df)


@st.cache_data(show_spinner=False)
def run_anomalies(df):
    return A.detect_anomalies(df)


@st.cache_data(show_spinner=False)
def load_rolling(df):
    """Walk-forward AI forecasts. Uses the pre-computed file when the data is the original dataset,
    otherwise recomputes (weekly refit + drift-triggered re-selection) for the current data."""
    f, l = ROOT / "data/processed/ai_rolling_forecasts.csv", ROOT / "data/processed/ai_retrain_log.csv"
    orig = D.load_default()
    if f.exists() and len(df) == len(orig) and float(df["demand"].sum()) == float(orig["demand"].sum()):
        r = pd.read_csv(f, dtype={"sku": str}, parse_dates=["week"])
        lg = pd.read_csv(l, dtype={"sku": str}) if l.exists() else pd.DataFrame()
        return r, lg, "pre-computed walk-forward run on the original dataset"
    bar = st.progress(0.0, text="Re-running walk-forward AI forecasts on the updated data…")
    r, lg = A.rolling_ai_all(df, 26, progress=lambda i, n, s_: bar.progress(i / n, text=f"Walk-forward retraining {i}/{n} · {s_}"))
    bar.empty()
    return r, lg, "recomputed after data update"


@st.cache_data(show_spinner=False)
def run_before_after(df, roll, service, lt, ltsd, cost_pct, hr, oc, sm, seg, use_seg):
    return A.before_after(df, roll, service, lt, ltsd, cost_pct, hr, oc, sm, seg if use_seg else None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"""<div class='side-brand'><div class='n'>📦 InvenTwin AI</div>
      <div class='m'>{STUDENT['course']} · Session II Lab<br>{STUDENT['name']} · {STUDENT['id']}<br>
      {STUDENT['prog']} · {STUDENT['batch']}</div></div>""", unsafe_allow_html=True)
    st.segmented_control("Appearance", ["Light", "Dark"], key="theme_mode", selection_mode="single")
    if ss.theme_mode is None:
        ss.theme_mode = mode
    base = portfolio_base(ss.df)
    sku_opts = base.sort_values("fc_total4", ascending=False)["sku"].tolist()
    labels = dict(zip(base["sku"], base["description"]))
    if ss.get("focus_sku") not in sku_opts:
        ss.focus_sku = sku_opts[0]
    st.selectbox("Focus SKU", sku_opts, key="focus_sku", format_func=lambda s: f"{s} · {str(labels.get(s, ''))[:28]}")
    st.markdown("**Service & supply assumptions**")
    st.slider("Target cycle service level (α)", 0.80, 0.999, key="service", step=0.005, format="%.3f",
              help="Probability of no stockout during a replenishment lead time.")
    st.toggle("AI service level per ML segment", key="ai_svc",
              help="ON: the AI assigns α per ABC-XYZ segment (A-items 95–98 %, C-items 90–94 %). OFF: one α for every SKU (classic practice).")
    st.slider("Mean supplier lead time (days)", 1.0, 60.0, key="lt_days", step=1.0,
              help="Not present in the dataset → configurable scenario assumption.")
    st.slider("Lead-time std. deviation (days)", 0.0, 20.0, key="lt_sd_days", step=0.5)
    st.radio("Demand basis for safety stock", ["AI forecast (dynamic)", "Historical (window)"], key="basis",
             help="AI: mean = 4-week ETS forecast, σ = forecast-error RMSE on an 8-week hold-out. Historical: weekly mean/σ over the window.")
    st.select_slider("History window (weeks)", [13, 26, 52, 104], key="window")
    with st.expander("💷 Cost parameters", expanded=False):
        st.slider("Unit cost (% of selling price)", 0.2, 0.9, key="cost_pct", step=0.05)
        st.slider("Annual holding rate", 0.05, 0.50, key="holding_rate", step=0.01)
        st.number_input("Ordering cost per PO (£)", 1.0, 1000.0, key="order_cost", step=5.0)
        st.slider("Stockout penalty (× selling price)", 0.25, 3.0, key="short_mult", step=0.25)
    with st.expander("🧪 Inventory scenario", expanded=False):
        st.number_input("Simulated stock seed", 0, 9999, key="stock_seed",
                        help="The dataset has no stock-on-hand. Scenario stock = baseline ROP × U(0.35, 2.2) with this seed.")
        st.caption("Edit individual stock levels in **AI Autopilot → Edit scenario stock**.")
    if st.button("↺ Reset scenario", width="stretch"):
        for k, v in DEFAULTS.items():
            if k != "theme_mode":
                ss[k] = v
        ss.stock_edits = {}
        for k in [k for k in ss.keys() if str(k).startswith(("ssc_", "mc_", "bo_"))]:
            del ss[k]
        st.rerun()
    st.caption("🔒 Single-user, local session. Uploaded data stays in this browser session — no peer sharing.")

# ------------------------------------------------------------------ portfolio (all SKUs) with current params
L_wk, sdL_wk = ss.lt_days / 7, ss.lt_sd_days / 7
svc = ss.service
w = ss.window
P = base.copy()
SEG = run_segment(ss.df)
P = P.merge(SEG[["sku", "segment", "cluster_name", "ai_service"]], on="sku", how="left")
P["svc"] = P["ai_service"].fillna(svc) if ss.ai_svc else svc
if ss.basis.startswith("AI"):
    P["d_bar"], P["sd_d"] = P["fc_mean"], P["fc_err_sd"]
else:
    P["d_bar"], P["sd_d"] = P[f"hist_mean_{w}"], P[f"hist_sd_{w}"]
P["unit_cost"] = P["price"] * ss.cost_pct
P["ltd_mean"] = P["d_bar"] * L_wk
P["sigma_ltd"] = np.sqrt(P["sd_d"] ** 2 * L_wk + P["d_bar"] ** 2 * sdL_wk ** 2)
P["ss"] = norm.ppf(P["svc"]) * P["sigma_ltd"]
P["rop"] = P["ltd_mean"] + P["ss"]
P["eoq"] = [max(1.0, I.eoq(d * 52, ss.order_cost, c * ss.holding_rate)) for d, c in zip(P["d_bar"], P["unit_cost"])]
# simulated stock is anchored to a FIXED baseline ROP (95%, 14 d ± 4 d, 52-wk history) so sliders move ROP, not stock
base_rop = (base["hist_mean_52"] * 2 + I.z_value(0.95) * np.sqrt(base["hist_sd_52"] ** 2 * 2 + base["hist_mean_52"] ** 2 * (4 / 7) ** 2))
sim_stock = I.simulated_on_hand(base["sku"], base_rop, ss.stock_seed)
P["on_hand"] = [ss.stock_edits.get(s, {}).get("on_hand", sim_stock[s]) for s in P["sku"]]
P["on_order"] = [ss.stock_edits.get(s, {}).get("on_order", 0.0) for s in P["sku"]]
P["backorders"] = [ss.stock_edits.get(s, {}).get("backorders", 0.0) for s in P["sku"]]
P["inv_position"] = P["on_hand"] + P["on_order"] - P["backorders"]
P["risk"] = [I.stockout_risk(ip, d, sd, L_wk, sdL_wk) for ip, d, sd in zip(P["inv_position"], P["d_bar"], P["sd_d"])]
P["status"] = [I.classify(ip, s_, r, m) for ip, s_, r, m in zip(P["inv_position"], P["ss"], P["rop"], P["ltd_mean"])]
P["weeks_cover"] = P["on_hand"] / P["d_bar"].clip(lower=1e-6)
# (s, Q) policy: when IP <= ROP order the smallest multiple of Q that lifts IP above ROP
_n = np.floor((P["rop"] - P["inv_position"]) / P["eoq"]) + 1
P["order_qty"] = np.where(P["inv_position"] <= P["rop"], _n * P["eoq"], 0.0)
P["action"] = [I.action(s_, r, ip, max(q, e)) for s_, r, ip, q, e in
               zip(P["status"], P["rop"], P["inv_position"], P["order_qty"], P["eoq"])]
P["stock_value"] = P["on_hand"] * P["unit_cost"]
P["status_rank"] = P["status"].map({s: i for i, s in enumerate(I.STATUS_ORDER)})

FOCUS = ss.focus_sku
FR = P.set_index("sku").loc[FOCUS]

# ------------------------------------------------------------------ hero
n_alert = int(P["status"].isin(["Reorder Now", "Critical Stockout Risk"]).sum())
st.markdown(f"""
<div class='hero'>
  <div class='eyebrow'>{STUDENT['course']} · {STUDENT['lab']}</div>
  <h1>AI-Powered Inventory Intelligence &amp; Reorder Point Optimization</h1>
  <div class='sub'>From a static textbook reorder-point calculator to an autonomous AI system: self-retraining ML forecasts ·
  drift &amp; anomaly detection · ML segmentation · Monte Carlo safety stock · Bayesian-optimised reorder points · an
  auto-replenishment agent.</div>
  <div class='chips'>
    <span class='chip'>👤 <b>{STUDENT['name']}</b> · {STUDENT['id']}</span>
    <span class='chip'>🎓 {STUDENT['prog']} · {STUDENT['batch']}</span>
    <span class='chip'>🏛 {STUDENT['school']}</span>
    <span class='chip'>📚 Instructor: {STUDENT['instructor']}</span>
    <span class='chip live'>Data refreshed {ss.last_ingest} · on-demand batch</span>
    <span class='chip'>🗃 {len(ss.df):,} SKU-days · {ss.df['sku'].nunique()} SKUs</span>
  </div>
</div>""", unsafe_allow_html=True)

COLS = ["sku", "description", "segment", "status", "fc_total4", "d_bar", "on_hand", "on_order", "inv_position", "ss", "rop",
        "order_qty", "risk", "weeks_cover", "stock_value", "action"]
tabs = st.tabs([":material/compare_arrows: Before vs After AI", ":material/smart_toy: AI Autopilot",
                ":material/insights: 1 · Demand Forecasting", ":material/shield: 2 · Safety Stock & Monte Carlo",
                ":material/psychology: 3 · Bayesian Optimization", ":material/database: Data & Reports"])

# ================================================================== TAB 1 · BEFORE vs AFTER AI
with tabs[0]:
    st.markdown(ui.section("Before AI vs After AI", "Same real demand, same suppliers, same costs — only the decision method changes", "head-to-head"), unsafe_allow_html=True)
    st.markdown(f"""<div class='compare-grid'><div class='card compare before'><h4>⏮ Before AI — classic textbook reorder point</h4><ul>
      <li>Average &amp; σ of <b>all past demand</b>, computed once</li>
      <li>One service level for <b>every</b> SKU</li>
      <li>Formula SS = Z√(σ²L + D̄²σ²<sub>L</sub>) set once, <b>never revised</b></li>
      <li>EOQ order quantity; manual what-if sweeps in a spreadsheet</li>
      <li>No monitoring — errors are found after the stockout</li>
      <li>Planner reviews every SKU by hand</li></ul></div><div class='card compare after'><h4>⏭ After AI — what the AI does here</h4><ul>
      <li><b>Learns continuously</b>: forecast refit every week; drift alarms trigger champion/challenger re-selection (ETS ↔ gradient boosting)</li>
      <li><b>Corrects itself</b>: recent under-forecasting bias is detected and added back</li>
      <li><b>Learns its own uncertainty</b>: conformal buffer = α-quantile of its realised lead-time forecast errors</li>
      <li><b>Decides service economically</b>: cost-optimal α per SKU, floored by the ML segment (ABC-XYZ + K-means)</li>
      <li><b>Optimises</b> (s, Q) with Bayesian optimisation instead of sweeps</li>
      <li><b>Acts</b>: agent flags anomalies, drafts POs and writes the daily digest</li></ul></div></div>""", unsafe_allow_html=True)

    roll, rlog, roll_src = load_rolling(ss.df)
    BA, PATHS = run_before_after(ss.df, roll, svc, ss.lt_days, ss.lt_sd_days, ss.cost_pct, ss.holding_rate, ss.order_cost,
                                 ss.short_mult, SEG, ss.ai_svc)
    tot = lambda c: BA[c].sum()
    dem_total = roll.groupby("sku")["actual"].sum().sum()
    fr_b, fr_a = 1 - tot("before_units_short") / dem_total, 1 - tot("after_units_short") / dem_total
    cb, ca = tot("before_total_cost"), tot("after_total_cost")
    wape_b = float((BA["wape_before"] * roll.groupby("sku")["actual"].sum().reindex(BA["sku"]).values).sum() / dem_total)
    wape_a = float((BA["wape_after"] * roll.groupby("sku")["actual"].sum().reindex(BA["sku"]).values).sum() / dem_total)
    n_drift = int(rlog["event"].astype(str).str.startswith("drift").sum()) if len(rlog) else 0
    n_switch = int(rlog["switched"].sum()) if len(rlog) else 0

    def delta(b, a, good_low=True, pct=True, unit=""):
        better = (a < b) if good_low else (a > b)
        cls = "up" if better else "down"
        d = (a - b) / abs(b) * 100 if pct and b else a - b
        return f"<span class='{cls}'>{d:+.1f}{'%' if pct else unit}</span> vs before"
    st.markdown(ui.kpi_row([
        ui.kpi_ba("Fill rate", f"{fr_b:.1%}", f"{fr_a:.1%}", delta(fr_b * 100, fr_a * 100, False, False, " pp"), "◐", T["green"]),
        ui.kpi_ba("Stockout days", f"{tot('before_stockout_days'):,}", f"{tot('after_stockout_days'):,}", delta(tot('before_stockout_days'), tot('after_stockout_days')), "⚠", T["red"]),
        ui.kpi_ba("Total cost · 26 wks", f"£{cb:,.0f}", f"£{ca:,.0f}", delta(cb, ca), "£", T["primary"]),
        ui.kpi_ba("Forecast WAPE", f"{wape_b:.0f}%", f"{wape_a:.0f}%", delta(wape_b, wape_a), "◎", T["blue"]),
    ]) + ui.kpi_row([
        ui.kpi_ba("Lost sales (units)", f"{tot('before_units_short'):,.0f}", f"{tot('after_units_short'):,.0f}", delta(tot('before_units_short'), tot('after_units_short')), "↯", T["orange"]),
        ui.kpi_ba("Avg stock value", f"£{tot('before_inv_value'):,.0f}", f"£{tot('after_inv_value'):,.0f}", delta(tot('before_inv_value'), tot('after_inv_value')), "▣", T["amber"]),
        ui.kpi("Automatic retrains", f"{len(rlog):,}", f"{n_drift} drift-triggered · {n_switch} model switches", "⟳", T["primary"]),
        ui.kpi("AI cheaper on", f"{(BA['after_total_cost'] < BA['before_total_cost']).sum()}/{len(BA)} SKUs", f"fill rate up on {(BA['after_fill_rate'] > BA['before_fill_rate'] + 1e-9).sum()} SKUs", "✓", T["green"]),
    ]), unsafe_allow_html=True)
    st.caption(f"Replay on the **last 26 weeks of actual daily demand** ({roll['week'].min():%d %b %Y} – {roll['week'].max():%d %b %Y}) for all {len(BA)} SKUs, "
               f"lost-sales (s, Q) policies, identical seeded Gamma lead times ({ss.lt_days:.0f} ± {ss.lt_sd_days:.0f} d) for both methods. "
               f"Before: α = {svc:.1%} for all SKUs, all-history mean/σ. After: cost-optimal α (avg {BA['ai_service'].mean():.1%}) floored at {'the ML-segment α' if ss.ai_svc else f'{svc:.1%}'} · forecasts: {roll_src}. Change any sidebar slider to re-run.")

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        pth = PATHS[FOCUS]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pth["dates"], y=pth["daily"], name="Actual daily demand", marker_color=T["grid"].replace("0.14", "0.45").replace("0.08", "0.35"), opacity=0.9), secondary_y=True)
        fig.add_trace(go.Scatter(x=pth["dates"], y=pth["before"], name="Stock · before AI", line=dict(color=T["muted"], width=2)))
        fig.add_trace(go.Scatter(x=pth["dates"], y=pth["after"], name="Stock · after AI", line=dict(color=T["primary"], width=2.5)))
        wk = pd.to_datetime(pth["weeks"]) - pd.Timedelta(days=6)
        fig.add_trace(go.Scatter(x=wk, y=pth["s_after"], name="AI reorder point (weekly)", line=dict(color=T["amber"], dash="dot", shape="hv")))
        fig.add_hline(y=pth["s_before"], line=dict(color=T["muted"], dash="dash"), annotation_text="static ROP", annotation_font_color=T["muted"])
        so_b = pth["dates"][(pth["before"] <= 0) & (pth["daily"] > 0)]
        fig.add_trace(go.Scatter(x=so_b, y=np.zeros(len(so_b)), mode="markers", name="Stockout · before", marker=dict(color=T["red"], symbol="x", size=7)))
        fig.update_layout(title=f"{FOCUS} · {FR['description'][:32]} — inventory replay on real demand", barmode="overlay")
        fig.update_yaxes(title_text="units on hand", secondary_y=False)
        fig.update_yaxes(showgrid=False, title_text="daily demand", secondary_y=True)
        fig_(fig, 420)
    with c2:
        r_ = roll[roll.sku == FOCUS]
        y_tr = D.series(ss.df, FOCUS, "W")
        static_mean = y_tr.iloc[: len(y_tr) - 26].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_["week"], y=r_["actual"], name="Actual", line=dict(color=T["blue"], width=2)))
        fig.add_trace(go.Scatter(x=r_["week"], y=[static_mean] * len(r_), name="Before: historical mean", line=dict(color=T["muted"], dash="dash")))
        fig.add_trace(go.Scatter(x=r_["week"], y=r_["forecast_1"], name="After: AI 1-week-ahead", line=dict(color=T["primary"], width=2.5)))
        if len(rlog):
            ev = rlog[rlog.sku == FOCUS]
            evd = pd.to_datetime(ev["week"])
            yv = r_.set_index("week")["actual"].reindex(evd).values
            fig.add_trace(go.Scatter(x=evd, y=yv, mode="markers", name="Auto-retrain / drift",
                                     marker=dict(symbol=np.where(ev["event"].str.startswith("drift"), "star", "circle-open"), size=12, color=T["amber"]),
                                     hovertext=ev["event"] + " → " + ev["champion"]))
        fig.update_layout(title="Forecast: static average vs self-retraining AI")
        fig_(fig, 420)

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        d = BA.assign(desc=BA["sku"].map(labels)).sort_values("before_total_cost")
        fig = go.Figure()
        for r in d.itertuples():
            fig.add_trace(go.Scatter(x=[r.before_total_cost, r.after_total_cost], y=[r.sku, r.sku], mode="lines",
                                     line=dict(color=T["green"] if r.after_total_cost < r.before_total_cost else T["red"], width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["before_total_cost"], y=d["sku"], mode="markers", name="Before AI", marker=dict(color=T["muted"], size=8)))
        fig.add_trace(go.Scatter(x=d["after_total_cost"], y=d["sku"], mode="markers", name="After AI", marker=dict(color=T["primary"], size=9)))
        fig.update_layout(title="Cost per SKU over 26 weeks (green = AI cheaper)", xaxis_tickprefix="£", xaxis_type="log")
        fig.update_yaxes(tickfont=dict(size=9))
        fig_(fig, 640)
    with c2:
        fig = px.scatter(BA.assign(desc=BA["sku"].map(labels)), x="before_fill_rate", y="after_fill_rate", hover_name="sku",
                         hover_data={"desc": True}, color_discrete_sequence=[T["primary"]])
        fig.add_shape(type="line", x0=0.5, y0=0.5, x1=1, y1=1, line=dict(color=T["muted"], dash="dot"))
        fig.update_traces(marker=dict(size=10, line=dict(color=T["bg"], width=1)))
        fig.update_layout(title="Fill rate per SKU (above the line = AI better)", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Before AI", yaxis_title="After AI")
        fig_(fig, 330, legend=False)
        st.markdown(ui.section("Work automated", "manual task → what the system does now"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ["Update demand averages", "Monthly, by hand", "Every data load, automatic"],
            ["Choose forecasting method", "Fixed", "Champion/challenger on drift"],
            ["Spot unusual demand", "Not done", "Isolation Forest alerts"],
            ["Set service levels", "One for all", "Per ML segment"],
            ["Size buffers", "Formula once", "Conformal + Monte Carlo"],
            ["Correct forecast bias", "Not done", "Automatic (tracking signal)"],
            ["Tune s and Q", "Spreadsheet sweep", "Bayesian optimisation"],
            ["Decide who to reorder", f"Review {len(P)} SKUs", "Agent drafts POs + reasons"],
        ], columns=["Task", "Before AI", "After AI"]), hide_index=True, width="stretch")
    with st.expander("Per-SKU before/after table & honest reading of the result"):
        show = BA[["sku", "ai_service", "before_fill_rate", "after_fill_rate", "before_stockout_days", "after_stockout_days",
                   "before_total_cost", "after_total_cost", "before_inv_value", "after_inv_value", "wape_before", "wape_after"]].copy()
        show.insert(1, "description", show["sku"].map(labels))
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ["ai_service", "before_fill_rate", "after_fill_rate"]} | {
            c: st.column_config.NumberColumn(format="£%.0f") for c in ["before_total_cost", "after_total_cost", "before_inv_value", "after_inv_value"]} | {
            c: st.column_config.NumberColumn(format="%.1f") for c in ["wape_before", "wape_after"]})
        st.download_button("⬇ Download before/after results (CSV)", show.to_csv(index=False).encode(), "before_vs_after_ai.csv", "text/csv")
        st.markdown("""**Honest reading.** The hold-out (Jun – Dec 2011) contains the pre-Christmas surge. The classic policy's σ of
        *all* history already includes last year's seasonal swing, so it carries a large, blunt buffer. The AI policy follows the surge
        week by week, cuts lost sales and stockout days and raises the fill rate at a similar or lower total cost — but it is not
        better on every SKU, and it may hold more stock where shortages are expensive. The largest cost saving comes from the
        Bayesian optimisation of (s, Q) (Workstream 3 tab). All SKUs are listed so the trade-off is transparent.""")

# ================================================================== TAB 2 · AI AUTOPILOT
with tabs[1]:
    st.markdown(ui.section("AI Autopilot · command centre", "Runs automatically on every data load — or on demand", "autonomous"), unsafe_allow_html=True)
    ANOM = run_anomalies(ss.df)
    roll, rlog, _ = load_rolling(ss.df)
    sig = (len(ss.df), float(ss.df["demand"].sum()), svc, ss.lt_days, ss.lt_sd_days, ss.ai_svc, ss.basis, ss.window,
           ss.cost_pct, ss.order_cost, ss.holding_rate, ss.stock_seed, str(ss.stock_edits))
    c1, c2 = st.columns([3, 1], gap="medium", vertical_alignment="center")
    with c2:
        manual = st.button("▶ Run AI Autopilot now", type="primary", width="stretch")
    if manual or ss.get("agent_sig") != sig:
        with st.spinner("Agent: perceive → reason → decide → act…"):
            R, alog = A.agent_decisions(P, L_wk, sdL_wk, today=ss.df["date"].max() + pd.Timedelta(days=1))
        ss.agent = {"R": R, "log": alog, "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "trigger": "manual run" if manual else ("data / settings change" if ss.get("agent_sig") else "app start")}
        ss.agent_sig = sig
    R, alog = ss.agent["R"], ss.agent["log"]
    recent_anom = ANOM[(ANOM.anomaly) & (ANOM.week >= ANOM.week.max() - pd.Timedelta(weeks=4))]
    n_drift = int(rlog["event"].astype(str).str.startswith("drift").sum()) if len(rlog) else 0
    with c1:
        st.markdown(f"<div style='font-size:.86rem;color:var(--muted)'>Last autonomous run <b style='color:var(--text)'>{ss.agent['at']}</b> · trigger: {ss.agent['trigger']} · "
                    "on-demand batch — re-runs automatically whenever data or settings change.</div>", unsafe_allow_html=True)
    steps = [("Ingest", f"{len(ss.df):,} rows"), ("Anomalies", f"{len(recent_anom)} new"),
             ("Drift & retrain", f"{len(rlog)} retrains"), ("Forecast", f"{len(P)} SKUs"),
             ("Segment", f"{SEG['segment'].nunique()} segments"), ("Simulate", f"{5000 * len(P) / 1e3:,.0f}k scenarios"),
             ("Decide", f"{(R.order_qty > 0).sum()} POs"), ("Report", "digest")]
    st.markdown("<div class='pipeline'>" + "".join(f"<div class='step'><div class='n'>{i + 1}</div><div><b>{a_}</b><span>{b_}</span></div></div>"
                for i, (a_, b_) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
    exp, ordr = (R.action == "EXPEDITE").sum(), (R.action == "ORDER").sum()
    st.markdown(ui.kpi_row([
        ui.kpi("SKUs monitored", fmt(len(P)), f"{ss.df['date'].min():%b %Y} → {ss.df['date'].max():%b %Y}", "▦", T["primary"]),
        ui.kpi("Forecast · 4 wks", fmt(P["fc_total4"].sum()), f"median WAPE {P['fc_wape'].median():.0f}%", "↗", T["blue"], "units"),
        ui.kpi("Draft purchase orders", f"{exp + ordr}", f"<span class='down'>{exp} expedite</span> · {ordr} order · £{R['po_value'].sum():,.0f}", "⚑", T["orange"]),
        ui.kpi("Avg stockout risk", f"{R['p_stockout'].mean():.1%}", f"{(R.p_stockout > 0.2).sum()} SKUs above 20%", "⚠", T["red"]),
    ]) + ui.kpi_row([
        ui.kpi("Avg safety stock / SKU", fmt(P["ss"].mean()), f"α {P['svc'].min():.0%}–{P['svc'].max():.0%} by segment" if ss.ai_svc else f"α = {svc:.1%} for all", "⛨", T["primary"], "units"),
        ui.kpi("Anomalies · 4 wks", f"{len(recent_anom)}", f"{ANOM.anomaly.sum()} in full history (Isolation Forest)", "◉", T["amber"]),
        ui.kpi("Drift alarms", f"{n_drift}", f"{len(rlog)} automatic retrains in walk-forward", "⟳", T["blue"]),
        ui.kpi("Inventory on hand", fmt(P["on_hand"].sum()), f"£{P['stock_value'].sum():,.0f} at cost · <b>simulated</b>", "▣", T["amber"], "units"),
    ]), unsafe_allow_html=True)

    st.markdown(ui.section("Replenishment agent decisions", "MC-sized reorder points · draft POs need human approval"), unsafe_allow_html=True)
    c = st.columns([1.3, 1, 1], gap="medium")
    acts = ["EXPEDITE", "ORDER", "WATCH", "HOLD · overstock", "OK"]
    fa = c[0].multiselect("Action filter", acts, ["EXPEDITE", "ORDER", "WATCH"], key="ag_f")
    qs = c[1].text_input("Search SKU / product", key="ag_q")
    segf = c[2].multiselect("Segment", sorted(R["segment"].dropna().unique()), key="ag_seg")
    V = R[R.action.isin(fa)] if fa else R
    if qs:
        V = V[V.sku.str.contains(qs, case=False) | V.description.str.contains(qs, case=False)]
    if segf:
        V = V[V.segment.isin(segf)]
    st.dataframe(V[["action", "sku", "description", "segment", "service", "order_qty", "po_value", "eta", "p_stockout", "days_cover", "rop_mc", "rop_formula", "confidence", "rationale"]],
                 hide_index=True, width="stretch", height=380, column_config={
                     "action": "Action", "sku": "SKU", "description": "Product", "segment": "Seg.",
                     "service": st.column_config.NumberColumn("α", format="percent"),
                     "order_qty": st.column_config.NumberColumn("Order qty", format="%.0f"),
                     "po_value": st.column_config.NumberColumn("PO value", format="£%.0f"), "eta": "ETA",
                     "p_stockout": st.column_config.ProgressColumn("P(stockout)", min_value=0, max_value=1, format="percent"),
                     "days_cover": st.column_config.NumberColumn("Days cover", format="%.0f"),
                     "rop_mc": st.column_config.NumberColumn("ROP (MC)", format="%.0f"),
                     "rop_formula": st.column_config.NumberColumn("ROP (formula)", format="%.0f"),
                     "confidence": st.column_config.ProgressColumn("Forecast conf.", min_value=0, max_value=1, format="percent"),
                     "rationale": st.column_config.TextColumn("Why", width="large")})
    po = R[R.order_qty > 0][["sku", "description", "order_qty", "po_value", "eta", "action", "rationale"]]
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Draft purchase orders (CSV)", po.to_csv(index=False).encode(), "draft_purchase_orders.csv", "text/csv", type="primary", width="stretch")
    d2.download_button("⬇ All agent decisions (CSV)", R.to_csv(index=False).encode(), "agent_decisions.csv", "text/csv", width="stretch")
    dg = A.digest(R, ANOM, rlog, SEG, ss.agent["at"])
    d3.download_button("⬇ Exception digest (Markdown)", dg.encode(), "ai_exception_digest.md", "text/markdown", width="stretch")

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        cnt = R["action"].value_counts().reindex(acts).fillna(0)
        colmap = {"EXPEDITE": T["red"], "ORDER": T["orange"], "WATCH": T["amber"], "HOLD · overstock": T["blue"], "OK": T["green"]}
        fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=0.66, sort=False, textinfo="value",
                               marker=dict(colors=[colmap[a] for a in cnt.index], line=dict(color=T["bg"], width=3))))
        fig.add_annotation(text=f"<b>{int(cnt.get('OK', 0))}/{len(R)}</b><br><span style='font-size:11px'>no action</span>", showarrow=False, font=dict(size=18, color=T["text"]))
        fig.update_layout(title="Agent action mix", legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle"))
        fig_(fig, 340)
    with c2:
        st.markdown(ui.section("Agent reasoning log"), unsafe_allow_html=True)
        st.markdown("".join(f"<div class='reco'><div class='dot' style='background:{c_}'></div><div><div class='t'>{l['step']} · {l['time']}</div><div class='d'>{l['detail']}</div></div></div>"
                            for l, c_ in zip(alog, [T["blue"], T["primary"], T["amber"], T["green"]])), unsafe_allow_html=True)

    st.markdown(ui.section("Anomaly & drift alerts", "Isolation Forest on weekly demand · tracking-signal drift on forecasts", "ML"), unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        yv = D.series(ss.df, FOCUS, "W")
        an_ = ANOM[(ANOM.sku == FOCUS) & ANOM.anomaly]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yv.index, y=yv, name="Weekly demand", line=dict(color=T["blue"], width=1.8)))
        fig.add_trace(go.Scatter(x=an_["week"], y=an_["demand"], mode="markers", name="Anomaly",
                                 marker=dict(size=12, color=np.where(an_["type"] == "Spike", T["red"], T["amber"]), symbol="circle-open", line=dict(width=3))))
        fig.update_layout(title=f"{FOCUS} · anomalies detected by Isolation Forest")
        fig_(fig, 340)
    with c2:
        al = ANOM[ANOM.anomaly].sort_values("week", ascending=False).head(25).assign(description=lambda d: d.sku.map(labels))
        st.dataframe(al[["week", "sku", "description", "type", "demand", "robust_z", "score"]], hide_index=True, width="stretch", height=340,
                     column_config={"week": st.column_config.DateColumn("Week", format="DD MMM YY"), "demand": st.column_config.NumberColumn(format="%.0f"),
                                    "robust_z": st.column_config.NumberColumn("Robust z", format="%.1f"), "score": st.column_config.ProgressColumn("Anomaly score", min_value=0.3, max_value=0.8, format="%.2f")})

    st.markdown(ui.section("ML segmentation & AI service levels", "ABC (revenue) × XYZ (variability) + K-means demand clusters"), unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        fig = px.scatter(SEG, x="mean_wk", y="cv", color="cluster_name", size="revenue", size_max=30, hover_name="sku",
                         hover_data={"description": True, "segment": True, "ai_service": ":.0%"}, log_x=True,
                         color_discrete_sequence=ui.colorway(T))
        fig.add_hline(y=0.5, line=dict(color=T["muted"], dash="dot"), annotation_text="X | Y")
        fig.add_hline(y=1.0, line=dict(color=T["muted"], dash="dot"), annotation_text="Y | Z")
        fig.update_layout(title="K-means demand clusters (bubble = revenue)", xaxis_title="Mean weekly demand (log)", yaxis_title="Coefficient of variation",
                          legend_title_text="")
        fig_(fig, 400)
    with c2:
        mat = SEG.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        txt_ = [[f"{mat.loc[a, x]} SKUs<br>α {A.SEG_SERVICE[a + x]:.0%}" for x in mat.columns] for a in mat.index]
        fig = go.Figure(go.Heatmap(z=mat.values, x=["X · stable", "Y · variable", "Z · erratic"], y=["A · top 80% revenue", "B · next 15%", "C · last 5%"],
                                   text=txt_, texttemplate="%{text}", colorscale=[[0, T["surface2"]], [1, T["primary"]]], showscale=False, xgap=4, ygap=4))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Segment matrix → auto-assigned service level")
        fig_(fig, 400, legend=False)

    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        with st.expander("📰 Today's AI exception digest", expanded=False):
            st.markdown(dg)
    with c2:
        with st.expander("✏️ Edit scenario stock (on hand / on order / backorders)"):
            ed = st.data_editor(P[["sku", "on_hand", "on_order", "backorders"]], hide_index=True, width="stretch", height=300,
                                disabled=["sku"], key="stock_editor", column_config={c_: st.column_config.NumberColumn(min_value=0, step=1, format="%.0f") for c_ in ["on_hand", "on_order", "backorders"]})
            if st.button("Apply stock changes", type="primary", width="stretch"):
                ss.stock_edits = {r.sku: {"on_hand": float(r.on_hand), "on_order": float(r.on_order), "backorders": float(r.backorders)} for r in ed.itertuples()}
                st.rerun()
            st.caption("Stock on hand is a simulated scenario — the public dataset has no inventory records.")

# ================================================================== TAB 3 · WORKSTREAM 1 · DEMAND FORECASTING
with tabs[2]:
    st.markdown(ui.section(f"Demand forecasting · {FOCUS}", f"{FR['description']} — chronological train / validation / test with rolling-origin backtesting", "ML"), unsafe_allow_html=True)
    c = st.columns([1.1, 1, 1, 1, 1.4])
    freq = c[0].segmented_control("Granularity", ["D", "W", "M"], default="W", format_func=lambda f: D.FREQ_LABEL[f], key="fc_freq") or "W"
    hmax = {"D": 56, "W": 16, "M": 6}[freq]
    hdef = {"D": 28, "W": 8, "M": 3}[freq]
    h = c[1].slider("Horizon (periods)", 1, hmax, hdef, key=f"fc_h_{freq}")
    folds = c[2].slider("Backtest folds", 1, 4, 3, key="fc_folds")
    level = c[3].select_slider("Interval", [0.80, 0.90, 0.95, 0.99], 0.95, format_func=lambda x: f"{x:.0%}", key="fc_level")
    chosen = c[4].multiselect("Models", F.MODELS, F.MODELS, key="fc_models")
    if len(chosen) == 0:
        st.warning("No model selected — showing all models.")
        chosen = list(F.MODELS)
    with st.spinner("Training & backtesting models…"):
        y, pr, bt = run_backtest(ss.df, FOCUS, freq, h, tuple(chosen), folds)
    best = bt["best"]
    sel_model = st.selectbox("Model used for the forward forecast", chosen, index=chosen.index(best), key=f"fc_sel_{freq}",
                             help="Defaults to the best model on validation WAPE (never on the test set).")
    rsd = bt["validation"].loc[sel_model, "resid_sd"]
    fc = run_forecast(ss.df, FOCUS, freq, h, sel_model, float(rsd if np.isfinite(rsd) else y.std()), level)
    v, te = bt["validation"], bt["test"]
    naive_w = te.loc["Naive", "WAPE %"] if "Naive" in te.index else np.nan
    st.markdown(ui.kpi_row([
        ui.kpi("Best model (validation)", f"<span style='font-size:1.0rem;white-space:normal;display:block;line-height:1.25'>{best}</span>", f"WAPE {v.loc[best, 'WAPE %']:.1f}% across {folds} fold(s)", "★", T["primary"]),
        ui.kpi("Test WAPE · selected", f"{te.loc[sel_model, 'WAPE %']:.1f}%", (f"<span class='up'>{naive_w - te.loc[sel_model, 'WAPE %']:.1f} pp better</span> than naive" if np.isfinite(naive_w) and naive_w > te.loc[sel_model, 'WAPE %'] else "vs naive benchmark"), "◎", T["blue"]),
        ui.kpi("Test RMSE · selected", fmt(te.loc[sel_model, "RMSE"], 1), f"MAE {te.loc[sel_model, 'MAE']:,.1f} · bias {te.loc[sel_model, 'Bias']:+,.1f}", "σ", T["amber"], "units"),
        ui.kpi(f"Forecast · next {h} {D.FREQ_LABEL[freq].lower()[:-2] if freq != 'D' else 'day'}s", fmt(fc["forecast"].sum()), f"{level:.0%} PI per period ±{(fc['upper'] - fc['forecast']).mean():,.0f}", "↗", T["green"], "units"),
    ]), unsafe_allow_html=True)

    show_models = st.multiselect("Overlay test-period predictions", chosen, [best] + ([m for m in ["Naive"] if m in chosen and m != best]), key="fc_overlay")
    dmin, dmax = y.index.min().date(), y.index.max().date()
    rng_ = st.slider("Display window", dmin, dmax, (max(dmin, (y.index[-1] - pd.Timedelta(days={"D": 180, "W": 730, "M": 1100}[freq])).date()), dmax), key=f"fc_rng_{freq}")
    yv = y[(y.index.date >= rng_[0]) & (y.index.date <= rng_[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yv.index, y=yv, name="Actual demand", line=dict(color=T["blue"], width=1.8)))
    pal = ui.colorway(T)[2:]
    for i, mname in enumerate(show_models):
        p = bt["test_preds"][mname]
        fig.add_trace(go.Scatter(x=p.index, y=p, name=f"{mname} (test)", line=dict(color=pal[i % len(pal)], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["upper"]) + list(fc["lower"][::-1]),
                             fill="toself", fillcolor="rgba(45,212,191,0.16)", line=dict(width=0), name=f"{level:.0%} interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name=f"Forecast · {sel_model}", line=dict(color=T["primary"], width=3), mode="lines+markers"))
    fig.add_vrect(x0=bt["test_actual"].index[0], x1=bt["test_actual"].index[-1], fillcolor=T["amber"], opacity=0.07, line_width=0,
                  annotation_text="test", annotation_position="top left")
    fig.update_layout(title=f"Actual vs predicted · {D.FREQ_LABEL[freq]}")
    fig_(fig, 430)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(ui.section("Model leaderboard", "Validation = model selection · Test = unbiased final check"), unsafe_allow_html=True)
        lb = v[["WAPE %", "MAE", "RMSE", "MAPE %"]].add_prefix("Val ").join(te[["WAPE %", "RMSE", "Bias"]].add_prefix("Test "))
        lb = lb.sort_values("Val WAPE %")
        st.dataframe(lb.style.format("{:,.1f}").highlight_min(subset=["Val WAPE %", "Test WAPE %"], color="rgba(45,212,191,0.25)"),
                     width="stretch", height=290)
        st.caption("MAPE excludes zero-demand periods (undefined when actual = 0) — WAPE is the primary metric for intermittent retail demand.")
    with c2:
        dfm = lb.reset_index().melt(id_vars="Model", value_vars=["Val WAPE %", "Test WAPE %"], var_name="Set", value_name="WAPE")
        fig = px.bar(dfm, x="WAPE", y="Model", color="Set", barmode="group", orientation="h",
                     color_discrete_sequence=[T["primary"], T["amber"]])
        fig.update_traces(marker_cornerradius=4)
        fig.update_layout(title="WAPE by model (lower is better)", yaxis=dict(categoryorder="total descending"))
        fig_(fig, 330)

    c1, c2 = st.columns(2)
    with c1:
        period = {"D": 7, "W": 52, "M": 12}[freq]
        if len(y) >= 2 * period + 1:
            dec = stl(y, period)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Trend", "Seasonal", "Residual"))
            fig.add_trace(go.Scatter(x=dec.index, y=dec["trend"], line=dict(color=T["primary"])), 1, 1)
            fig.add_trace(go.Scatter(x=dec.index, y=dec["seasonal"], line=dict(color=T["blue"])), 2, 1)
            fig.add_trace(go.Bar(x=dec.index, y=dec["resid"], marker_color=T["muted"]), 3, 1)
            fig.update_layout(title=f"STL seasonal decomposition (period {period})")
            fig_(fig, 420, legend=False)
        else:
            st.info(f"STL needs ≥ {2 * period + 1} periods; {len(y)} available at {D.FREQ_LABEL[freq]} granularity. Seasonality is still modelled via calendar features in the ML models.")
            prof = ss.df[ss.df.sku == FOCUS].assign(m=lambda d: d.date.dt.month).groupby("m")["demand"].mean()
            fig = go.Figure(go.Bar(x=[pd.Timestamp(2000, i, 1).strftime("%b") for i in prof.index], y=prof.values, marker_color=T["blue"]))
            fig.update_layout(title="Average daily demand by month")
            fig_(fig, 300, legend=False)
    with c2:
        imp = run_importance(ss.df, FOCUS, freq)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=imp.values, colorscale=[[0, T["surface2"]], [1, T["primary"]]], cornerradius=4)))
        fig.update_layout(title="Gradient-boosting feature importance")
        fig_(fig, 420, legend=False)

    with st.expander("Forecast table & export"):
        out = fc.copy(); out.index.name = "period"
        st.dataframe(out.style.format("{:,.1f}"), width="stretch")
        st.download_button("⬇ Download forecast CSV", out.to_csv().encode(), f"forecast_{FOCUS}_{freq}.csv", "text/csv")
    with st.expander("ℹ️ Why no LSTM? · promotions & external factors"):
        st.markdown(f"""
- **LSTM** was evaluated as a design option but deliberately **not deployed**: this SKU has only **{len(D.series(ss.df, FOCUS, 'W'))} weekly** observations.
  Recurrent networks need thousands of sequences to beat strong statistical baselines; on short retail series they overfit and
  lose to ETS/ARIMA (cf. M4/M5 competition findings). A lighter neural network (**MLP**) on engineered lag features is included instead
  and compared fairly — it is selected automatically whenever it wins on validation.
- **Promotions:** the source data contains **no promotion flag** — none was fabricated. The observed **average selling price**
  (discounting acts as a promotion proxy) is used as a regressor; future price is held at its last observed value.
- **External factors / seasonality:** derived calendar features (week-of-year & day-of-year Fourier terms, day-of-week, a Sep–Nov
  pre-Christmas wholesale peak flag) capture seasonality without inventing external data.
- **Continuous updating:** upload new observations in **Data Manager** → validation, reconciliation, re-training, backtesting and
  re-computation of safety stock, ROP and recommendations happen automatically on the next run (on-demand batch refresh).""")

# ================================================================== TAB 4 · WORKSTREAM 2 · SAFETY STOCK & MONTE CARLO
with tabs[3]:
    w2 = st.segmented_control("Workstream 2 view", ["Analytical safety stock & ROP", "Monte Carlo risk simulation"],
                              default="Analytical safety stock & ROP", key="w2_view", label_visibility="collapsed") or "Analytical safety stock & ROP"
    if w2.startswith("Analytical"):
        st.markdown(ui.section(f"Safety stock & reorder-point calculator · {FOCUS}", "Inputs pre-filled from the AI forecast / sidebar — override any value", "analytical"), unsafe_allow_html=True)

        def ninput(col, label, default, key, step, fmt_="%.2f", mn=0.0, mx=None, help=None):
            return col.number_input(label, min_value=mn, max_value=mx, value=float(round(default, 3)), step=step, format=fmt_,
                                    key=f"ssc_{key}_{FOCUS}_{round(float(default), 3)}", help=help)
        c = st.columns(4)
        d_bar = ninput(c[0], "Mean demand D̄ (units/week)", FR["d_bar"], "dbar", 1.0)
        sd_d = ninput(c[1], "Demand σ_D (units/week)", FR["sd_d"], "sdd", 1.0)
        L_d = ninput(c[2], "Mean lead time L (days)", ss.lt_days, "L", 1.0, "%.1f", 0.1)
        sdL_d = ninput(c[3], "Lead-time σ_L (days)", ss.lt_sd_days, "sdL", 0.5, "%.1f")
        c = st.columns(4)
        svc_c = c[0].slider("Service level α", 0.80, 0.999, float(svc), 0.005, format="%.3f", key=f"ssc_svc_{svc}")
        oh = ninput(c[1], "Current stock on hand", FR["on_hand"], "oh", 10.0, "%.0f", help="Simulated scenario value unless edited.")
        oo = ninput(c[2], "On-order quantity", FR["on_order"], "oo", 10.0, "%.0f")
        bo = ninput(c[3], "Backorders", FR["backorders"], "bo", 10.0, "%.0f")
        Lw, sLw = L_d / 7, sdL_d / 7
        comp = I.ss_components(svc_c, d_bar, sd_d, Lw, sLw)
        ip = oh + oo - bo
        risk = I.stockout_risk(ip, d_bar, sd_d, Lw, sLw)
        status = I.classify(ip, comp["ss"], comp["rop"], comp["ltd_mean"])
        q = max(1.0, I.eoq(d_bar * 52, ss.order_cost, FR["unit_cost"] * ss.holding_rate))
        n_ord = np.floor((comp["rop"] - ip) / q) + 1 if ip <= comp["rop"] else 0
        st.markdown(ui.kpi_row([
            ui.kpi("Z factor", f"{comp['z']:.3f}", f"Φ⁻¹({svc_c:.3f})", "Z", T["blue"]),
            ui.kpi("Expected LT demand", fmt(comp["ltd_mean"]), f"D̄·L = {d_bar:,.1f} × {Lw:.2f} wk", "∑", T["primary"], "units"),
            ui.kpi("Safety stock", fmt(comp["ss"]), f"σ_LTD = {comp['sigma_ltd']:,.1f}", "⛨", T["primary"], "units"),
            ui.kpi("Reorder point", fmt(comp["rop"]), f"{comp['rop'] / max(d_bar, 1e-9):.1f} weeks of demand", "◆", T["amber"], "units"),
        ], 4) + ui.kpi_row([
            ui.kpi("Inventory position", fmt(ip), "on hand + on order − backorders", "▣", T["blue"], "units"),
            ui.kpi("Stockout risk now", f"{risk:.1%}", "P(LTD > position) · Normal approx.", "⚠", T["red"] if risk > 1 - svc_c else T["green"]),
            ui.kpi("Status", f"<span style='font-size:1.0rem;color:{SC[status]};white-space:normal;display:block;line-height:1.25'>{status}</span>", f"EOQ = {q:,.0f} units", "⚑", SC[status]),
            ui.kpi("Recommendation", f"<span style='font-size:1.0rem;white-space:normal;display:block;line-height:1.25'>{'Order ' + fmt(n_ord * q) + ' u' if n_ord else 'Hold'}</span>",
                   (f"IP ≤ ROP → order {int(n_ord)}×EOQ" if n_ord else f"{ip - comp['rop']:,.0f} u above ROP"), "➜", T["green"]),
        ], 4), unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='chart-title'>Formula (from the assignment)</div>", unsafe_allow_html=True)
            st.latex(r"SS = Z\sqrt{\sigma_D^2\,L + \bar D^2\,\sigma_L^2}\qquad ROP = \bar D\,L + SS")
            st.latex(rf"SS = {comp['z']:.3f}\sqrt{{{sd_d:,.1f}^2\times{Lw:.3f} + {d_bar:,.1f}^2\times{sLw:.3f}^2}} = \mathbf{{{comp['ss']:,.1f}}}")
            st.latex(rf"ROP = {d_bar:,.1f}\times{Lw:.3f} + {comp['ss']:,.1f} = \mathbf{{{comp['rop']:,.1f}}}")
            st.markdown(f"""<div style='font-size:.84rem;color:var(--muted)'>Units: demand per <b>week</b>, lead time converted
            days → weeks (L = {L_d:.1f}/7 = {Lw:.3f} wk). Variance share: <b style='color:{T['primary']}'>{comp['share_demand']:.0%} demand</b> ·
            <b style='color:{T['amber']}'>{comp['share_leadtime']:.0%} lead time</b>. Assumes i.i.d. weekly demand, lead time independent of
            demand and approximately Normal lead-time demand — the Monte Carlo comparison below relaxes these.</div>""", unsafe_allow_html=True)
        with c2:
            sl = np.linspace(0.80, 0.999, 120)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sl, y=[I.safety_stock(x, d_bar, sd_d, Lw, sLw) for x in sl], name="Safety stock",
                                     line=dict(color=T["primary"], width=3), fill="tozeroy", fillcolor="rgba(45,212,191,0.10)"))
            fig.add_trace(go.Scatter(x=[svc_c], y=[comp["ss"]], mode="markers+text", text=[f"  {svc_c:.1%} → {comp['ss']:,.0f}"],
                                     textposition="middle right", marker=dict(size=13, color=T["amber"]), name="Current"))
            fig.update_layout(title="Cost of service: safety stock vs service level", xaxis_tickformat=".0%")
            fig_(fig, 330)

        c1, c2 = st.columns(2)
        with c1:
            x = np.linspace(max(0, comp["ltd_mean"] - 4 * comp["sigma_ltd"]), comp["ltd_mean"] + 4.5 * comp["sigma_ltd"], 400)
            pdf = norm.pdf(x, comp["ltd_mean"], comp["sigma_ltd"])
            mc = run_mc(20000, "Gamma", d_bar, max(sd_d, 1e-6), "Gamma", Lw, max(sLw, 1e-6), 0.0, 11, None)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=mc["ltd"], histnorm="probability density", nbinsx=70, name="Monte Carlo (Gamma/Gamma)",
                                       marker_color=T["blue"], opacity=0.35))
            fig.add_trace(go.Scatter(x=x, y=pdf, name="Normal approximation", line=dict(color=T["primary"], width=2.5)))
            xt = x[x >= comp["rop"]]
            fig.add_trace(go.Scatter(x=np.r_[xt, xt[::-1]], y=np.r_[norm.pdf(xt, comp["ltd_mean"], comp["sigma_ltd"]), np.zeros(len(xt))],
                                     fill="toself", fillcolor="rgba(248,113,113,0.35)", line=dict(width=0), name=f"Stockout zone ({1 - svc_c:.1%})"))
            fig.add_vline(x=comp["rop"], line=dict(color=T["amber"], dash="dash"), annotation_text="ROP")
            fig.add_vline(x=ip, line=dict(color=T["text"], dash="dot"), annotation_text="Position", annotation_position="bottom right")
            fig.update_layout(title="Lead-time demand distribution", bargap=0.02)
            fig_(fig, 360)
        with c2:
            mults = np.round(np.linspace(0.5, 2.0, 7), 2)
            sls = np.linspace(0, max(2 * sdL_d, 6), 7)
            Z = [[I.safety_stock(svc_c, d_bar, sd_d * m, Lw, s_ / 7) for m in mults] for s_ in sls]
            fig = go.Figure(go.Heatmap(z=Z, x=[f"{m:.2f}×" for m in mults], y=[f"{s_:.1f} d" for s_ in sls], texttemplate="%{z:,.0f}",
                                       colorscale=[[0, T["surface2"]], [0.5, T["blue"]], [1, T["red"]]], colorbar=dict(thickness=10),
                                       hovertemplate="σ_D %{x} · σ_L %{y}<br>SS %{z:,.0f}<extra></extra>"))
            fig.update_layout(title="How variability inflates the buffer", xaxis_title="Demand σ multiplier", yaxis_title="Lead-time σ (days)")
            fig_(fig, 360, legend=False)

        c1, c2 = st.columns([1, 1.1])
        with c1:
            base_ss = comp["ss"]
            items = {"Demand σ_D": ("sd", sd_d), "Mean demand D̄": ("d", d_bar), "Lead time L": ("L", Lw), "Lead-time σ_L": ("sL", sLw)}
            rows = []
            for lab, (k, v) in items.items():
                lo_hi = []
                for f in (0.8, 1.2):
                    a = dict(d=d_bar, sd=sd_d, L=Lw, sL=sLw); a[k] = v * f
                    lo_hi.append(I.safety_stock(svc_c, a["d"], a["sd"], a["L"], a["sL"]) - base_ss)
                rows.append((lab, *lo_hi))
            rows.sort(key=lambda r: abs(r[2] - r[1]))
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", name="−20%", marker_color=T["blue"]))
            fig.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h", name="+20%", marker_color=T["amber"]))
            fig.update_layout(barmode="overlay", title="Tornado: Δ safety stock for ±20% input change")
            fig_(fig, 320)
        with c2:
            st.markdown(ui.section("Analytical vs simulation", "20,000 draws each · same inputs", "MC"), unsafe_allow_html=True)
            rows = [{"Method": "Analytical formula (Normal)", "E[LTD]": comp["ltd_mean"], "ROP": comp["rop"], "Safety stock": comp["ss"]}]
            for lab, dd, ld, rho in [("MC · Normal demand, Normal LT", "Normal (truncated at 0)", "Normal (truncated)", 0.0),
                                     ("MC · Gamma demand, Gamma LT", "Gamma", "Gamma", 0.0),
                                     ("MC · Gamma, skewed LT (triangular)", "Gamma", "Triangular (right-skewed)", 0.0),
                                     ("MC · joint: LT ↔ demand ρ = 0.5", "Gamma", "Gamma", 0.5)]:
                r_ = run_mc(20000, dd, d_bar, max(sd_d, 1e-6), ld, Lw, max(sLw, 1e-6), rho, 21, None)["ltd"]
                q_ = float(np.quantile(r_, svc_c))
                rows.append({"Method": lab, "E[LTD]": r_.mean(), "ROP": q_, "Safety stock": q_ - r_.mean()})
            cmp_ = pd.DataFrame(rows)
            cmp_["Δ ROP vs formula"] = cmp_["ROP"] - comp["rop"]
            ss["ssc_compare"] = cmp_
            st.dataframe(cmp_.style.format({c: "{:,.1f}" for c in cmp_.columns if c != "Method"}), hide_index=True, width="stretch")
            st.caption("Right-skewed or demand-correlated lead times push the true quantile above the Normal formula → the analytical SS under-protects; use the simulated buffer when independence/normality is doubtful.")
    else:
        st.markdown(ui.section(f"Monte Carlo risk simulation · {FOCUS}", "Lead-time demand = Σ weekly demand over a random lead time · reproducible seed", "probabilistic"), unsafe_allow_html=True)
        yw = D.series(ss.df, FOCUS, "W")
        with st.form("mc_form"):
            c = st.columns(4)
            dd = c[0].selectbox("Demand distribution", S.DEMAND_DISTS, index=1, help="Gamma is non-negative and right-skewed — a natural fit for retail demand. Empirical bootstraps the observed weekly history.")
            ld = c[1].selectbox("Lead-time distribution", S.LEAD_DISTS, index=1)
            n_sim = c[2].select_slider("Simulations", [1000, 5000, 10000, 25000, 50000, 100000], 10000)
            seed = c[3].number_input("Random seed", 0, 99999, 42)
            c = st.columns(4)
            dm = c[0].number_input("Mean demand (units/wk)", 0.0, value=float(round(FR["d_bar"], 1)), step=1.0)
            dsd = c[1].number_input("Demand σ (units/wk)", 0.0, value=float(round(FR["sd_d"], 1)), step=1.0)
            lm = c[2].number_input("Mean lead time (days)", 0.5, value=float(ss.lt_days), step=1.0)
            lsd = c[3].number_input("Lead-time σ (days)", 0.0, value=float(ss.lt_sd_days), step=0.5)
            c = st.columns(4)
            msvc = c[0].slider("Service target α", 0.80, 0.999, float(svc), 0.005, format="%.3f")
            rho = c[1].slider("Lead time ↔ demand correlation ρ", 0.0, 0.9, 0.0, 0.05, help="Joint simulation (Gaussian copula). ρ>0: busy periods coincide with slower suppliers — violates the formula's independence assumption.")
            spike = c[2].slider("Stress: demand spike", 0.0, 1.0, 0.30, 0.05, format="%.2f")
            delay = c[3].slider("Stress: supplier delay", 0.0, 1.5, 0.50, 0.05, format="%.2f")
            test_rop = st.slider("Reorder point to evaluate (units)", 0.0, float(max(10.0, FR["rop"] * 2.5)), float(round(FR["rop"])), 1.0)
            go_ = st.form_submit_button("▶ Run Monte Carlo simulation", type="primary", width="stretch")
        params = (n_sim, dd, dm, max(dsd, 1e-6), ld, lm / 7, max(lsd / 7, 1e-6), rho, int(seed))
        if go_ or ss.get("mc_params") is None or ss.get("mc_sku") != FOCUS:
            ss.mc_params, ss.mc_sku = params, FOCUS
            ss.mc_run_at = datetime.now().strftime("%H:%M:%S")
        n_sim, dd, dm, dsd, ld, lmw, lsdw, rho, seed = ss.mc_params
        emp = tuple(yw.tail(104).values) if dd.startswith("Empirical") else None
        with st.spinner(f"Simulating {n_sim:,} lead-time cycles…"):
            mc = run_mc(n_sim, dd, dm, dsd, ld, lmw, lsdw, rho, seed, emp)
        q_eoq = float(FR["eoq"])
        summ = S.summarize(mc["ltd"].values, msvc, test_rop, q_eoq)
        an = I.ss_components(msvc, dm, dsd, lmw, lsdw)
        st.markdown(ui.kpi_row([
            ui.kpi("Stockout probability", f"{summ['stockout_prob']:.2%}", f"at ROP {test_rop:,.0f} · per cycle (1 − α)", "⚠", T["red"] if summ["stockout_prob"] > 1 - msvc else T["green"]),
            ui.kpi("Cycle service level α", f"{summ['cycle_service']:.2%}", f"target {msvc:.1%}", "✓", T["green"]),
            ui.kpi("Fill rate β", f"{summ['fill_rate']:.2%}", f"units served / demanded · Q = EOQ {q_eoq:,.0f}", "◐", T["blue"]),
            ui.kpi("Expected shortage", fmt(summ["expected_shortage"], 1), "units short per replenishment cycle", "↯", T["orange"], "units"),
        ]) + ui.kpi_row([
            ui.kpi("Required buffer (MC)", fmt(summ["ss_sim"]), f"Analytical SS {an['ss']:,.0f} · Δ {summ['ss_sim'] - an['ss']:+,.0f}", "⛨", T["primary"], "units"),
            ui.kpi("Required ROP (MC)", fmt(summ["rop_sim"]), f"{msvc:.1%} quantile of LTD", "◆", T["amber"], "units"),
            ui.kpi("Mean LT demand", f"{summ['mean_ltd']:,.0f}", f"σ {summ['sd_ltd']:,.0f} · skew {summ['skew']:.2f}", "∑", T["blue"], "units"),
            ui.kpi("P95 / P99 LT demand", f"{summ['p95_ltd']:,.0f}", f"P99 {summ['p99_ltd']:,.0f}", "▲", T["red"], "units"),
        ]), unsafe_allow_html=True)
        st.caption(f"Run at {ss.get('mc_run_at')} · seed **{seed}** · n = **{n_sim:,}** · demand **{dd}** (μ={dm:,.1f}, σ={dsd:,.1f}/wk) · lead time **{ld}** (μ={lmw * 7:.1f}, σ={lsdw * 7:.1f} days) · ρ = {rho:.2f}. "
                   "α (cycle service) = P(no stockout in a cycle); β (fill rate) = share of demand met from stock — they are different metrics.")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=mc["ltd"], nbinsx=80, histnorm="probability", marker_color=T["blue"], opacity=0.8, name="Simulated LTD"))
            for xv, lab, col in [(summ["mean_ltd"], "Mean", T["muted"]), (an["rop"], "ROP formula", T["primary"]),
                                 (summ["rop_sim"], "ROP MC", T["amber"]), (test_rop, "Evaluated ROP", T["red"])]:
                fig.add_vline(x=xv, line=dict(color=col, dash="dash", width=2), annotation_text=lab, annotation_font_color=col)
            fig.update_layout(title="Lead-time demand probability histogram", bargap=0.03)
            fig_(fig, 360, legend=False)
        with c2:
            xs = np.sort(mc["ltd"].values)
            cdf = np.arange(1, len(xs) + 1) / len(xs)
            step = max(1, len(xs) // 1500)
            fig = go.Figure(go.Scatter(x=xs[::step], y=cdf[::step], line=dict(color=T["primary"], width=3), name="Empirical CDF",
                                       fill="tozeroy", fillcolor="rgba(45,212,191,0.08)"))
            fig.add_hline(y=msvc, line=dict(color=T["amber"], dash="dot"), annotation_text=f"α = {msvc:.1%}")
            fig.add_vline(x=summ["rop_sim"], line=dict(color=T["amber"], dash="dot"))
            fig.update_layout(title="Cumulative probability: service achieved by stocking level", yaxis_tickformat=".0%", xaxis_title="Units available during lead time")
            fig_(fig, 360, legend=False)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            smp = mc.sample(min(3000, len(mc)), random_state=1)
            fig = go.Figure(go.Scattergl(x=smp["lead_time"] * 7, y=smp["ltd"], mode="markers",
                                         marker=dict(size=4, color=smp["ltd"] > test_rop, colorscale=[[0, T["blue"]], [1, T["red"]]], opacity=0.55)))
            fig.add_hline(y=test_rop, line=dict(color=T["amber"], dash="dash"), annotation_text="ROP")
            fig.update_layout(title="Joint risk: lead time vs lead-time demand (red = stockout)", xaxis_title="Lead time (days)", yaxis_title="LTD (units)")
            fig_(fig, 380, legend=False)
        with c2:
            scen = S.scenario_set(dm, dsd, lmw, lsdw, spike, delay)
            rows, dists = [], {}
            for i, (nm, (a, b, c_, d_)) in enumerate(scen.items()):
                r_ = run_mc(min(n_sim, 25000), dd if not dd.startswith("Empirical") else "Gamma", a, max(b, 1e-6), ld, c_, max(d_, 1e-6), rho, seed + i, None)["ltd"].values
                sm = S.summarize(r_, msvc, test_rop, q_eoq)
                rows.append({"Scenario": nm, "Mean LTD": sm["mean_ltd"], "Required SS": sm["ss_sim"], "Required ROP": sm["rop_sim"],
                             "P(stockout) @ current ROP": sm["stockout_prob"], "Fill rate β": sm["fill_rate"], "Exp. shortage": sm["expected_shortage"]})
                dists[nm] = r_
            sc = pd.DataFrame(rows)
            ss["mc_scenarios"] = sc
            fig = go.Figure()
            cols_ = [T["primary"], T["amber"], T["orange"], T["blue"], T["red"]]
            for (nm, r_), col in zip(dists.items(), cols_):
                fig.add_trace(go.Violin(y=r_[:4000], name=nm.split(" (")[0], line_color=col, fillcolor=col, opacity=0.55, box_visible=True, meanline_visible=True, points=False))
            fig.add_hline(y=test_rop, line=dict(color=T["text"], dash="dot"), annotation_text="current ROP")
            fig.update_layout(title="Risk scenarios: lead-time demand distributions", yaxis_title="Units")
            fig_(fig, 380, legend=False)
        st.markdown(ui.section("Scenario comparison", "Buffer required to keep α at target under each stress", "scenarios"), unsafe_allow_html=True)
        st.dataframe(sc, hide_index=True, width="stretch", column_config={
            "Mean LTD": st.column_config.NumberColumn(format="%.0f"), "Required SS": st.column_config.NumberColumn(format="%.0f"),
            "Required ROP": st.column_config.NumberColumn(format="%.0f"),
            "P(stockout) @ current ROP": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Fill rate β": st.column_config.NumberColumn(format="percent"), "Exp. shortage": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("⬇ Download simulation draws (CSV)", mc.to_csv(index=False).encode(), f"montecarlo_{FOCUS}_seed{seed}.csv", "text/csv")

# ================================================================== TAB 5 · WORKSTREAM 3 · BAYESIAN OPTIMIZATION
with tabs[4]:
    st.markdown(ui.section(f"Bayesian optimisation lab · {FOCUS}", "Replaces manual parameter sweeps — a surrogate model decides where to evaluate next", "AI"), unsafe_allow_html=True)
    price = float(FR["price"]); ucost = float(FR["unit_cost"])
    rop0, eoq0 = float(FR["rop"]), float(FR["eoq"])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("<div class='chart-title'>Objective & constraints</div>", unsafe_allow_html=True)
        st.latex(r"\min_{s,\,Q}\; \mathbb{E}[C] = \underbrace{K\,\tfrac{N_{orders}}{yr}}_{ordering} + \underbrace{h\,c\,\bar I}_{holding} + \underbrace{p\,\mathbb{E}[\text{units short}]}_{shortage}")
        st.latex(r"\text{s.t. } \beta \ge \beta^{*},\;\; I^{peak}_{95\%} \le \text{Cap},\;\; c\,\bar I \le \text{Budget},\;\; Q \ge \text{MOQ}")
        st.markdown(f"""<div style='font-size:.83rem;color:var(--muted)'>Continuous-review <b>(s, Q)</b> policy simulated <b>day by day</b>
        (Gamma demand, stochastic lead times, lost sales). Every candidate is scored on the <b>same random numbers</b> (CRN) so cost
        differences reflect the policy, not noise; the winner is re-validated on an <b>independent</b> stream.
        K = £{ss.order_cost:,.0f}/PO · c = £{ucost:,.2f}/unit · h = {ss.holding_rate:.0%}/yr · p = £{price * ss.short_mult:,.2f}/unit short.</div>""", unsafe_allow_html=True)
    with c2:
        with st.form("bo_form"):
            a, b = st.columns(2)
            method = a.selectbox("Algorithm", ["GP-EI (Gaussian process)", "Optuna TPE"], help="GP-EI: Matern-5/2 GP surrogate + Expected Improvement acquisition. TPE: Optuna's Tree-structured Parzen Estimator with constraint handling.")
            n_trials = b.slider("Evaluations", 15, 80, 40, 5)
            target_fill = a.slider("Min fill rate β*", 0.80, 0.995, 0.95, 0.005, format="%.3f")
            reps = b.select_slider("Sim replications", [50, 100, 150, 250, 400], 150)
            cap = a.number_input("Warehouse capacity (units)", 10.0, value=float(round((rop0 + 2.5 * eoq0) / 10) * 10), step=50.0)
            budget = b.number_input("Avg inventory budget (£)", 10.0, value=float(round(ucost * (rop0 + eoq0))), step=50.0)
            moq = a.number_input("Supplier MOQ (units)", 1.0, value=float(max(1, round(eoq0 * 0.25))), step=10.0)
            bseed = b.number_input("Seed", 0, 9999, 2025)
            s_rng = st.slider("Search range · reorder point s", 0.0, float(round(rop0 * 3 + 10)), (0.0, float(round(rop0 * 2.2 + 5))), 1.0)
            q_rng = st.slider("Search range · order quantity Q", 1.0, float(round(max(eoq0 * 5, moq * 4) + 10)), (float(moq), float(round(max(eoq0 * 3.5, moq * 2) + 5))), 1.0)
            run_bo = st.form_submit_button("🧠 Run Bayesian optimisation", type="primary", width="stretch")
    cp = O.CostParams(unit_cost=ucost, holding_rate=ss.holding_rate, order_cost=ss.order_cost, shortage_cost=price * ss.short_mult,
                      target_fill=target_fill, capacity=cap, budget=budget, moq=moq)
    if run_bo:
        sim = O.PolicySimulator(FR["d_bar"], FR["sd_d"], L_wk, max(sdL_wk, 1e-3), "Gamma", reps=reps, seed=int(bseed))
        base_q = max(eoq0, moq)
        base_res = sim.run(rop0, base_q, cp)
        prog = st.progress(0.0, text="Initialising surrogate…")
        def cb(i, n, r):
            prog.progress(i / n, text=f"Evaluation {i}/{n} · s={r['s']:,.0f}, Q={r['Q']:,.0f} → £{r['total_cost']:,.0f}{'' if r['feasible'] else ' (infeasible)'}")
        bounds = {"s": s_rng, "Q": (max(q_rng[0], moq), max(q_rng[1], moq + 1))}
        if method.startswith("GP"):
            res = O.gp_bayes_opt(sim, cp, bounds, n_init=min(10, n_trials // 3), n_iter=n_trials - min(10, n_trials // 3), seed=int(bseed), penalty_scale=base_res["total_cost"], progress=cb)
        else:
            res = O.tpe_bayes_opt(sim, cp, bounds, n_trials=n_trials, seed=int(bseed), penalty_scale=base_res["total_cost"], progress=cb)
        prog.empty()
        val = O.PolicySimulator(FR["d_bar"], FR["sd_d"], L_wk, max(sdL_wk, 1e-3), "Gamma", reps=max(reps, 300), seed=int(bseed) + 7919)
        ss.bo = {"sim_args": (FR["d_bar"], FR["sd_d"], L_wk, max(sdL_wk, 1e-3), "Gamma"), "sim_kw": {"reps": reps, "seed": int(bseed)},
                 "bounds": bounds, "res": res, "base": base_res, "val_best": val.run(res["best"]["s"], res["best"]["Q"], cp),
                 "val_base": val.run(rop0, base_q, cp), "sku": FOCUS, "cp": cp, "at": datetime.now().strftime("%H:%M:%S"),
                 "ltd_mean": float(FR["ltd_mean"]), "method": method}
    bo = ss.get("bo")
    if not bo or bo["sku"] != FOCUS:
        st.info("Configure the scenario and press **Run Bayesian optimisation**. Baseline policy = analytical ROP (sidebar service level) with EOQ order quantity.")
    else:
        res, b0, vb, v0 = bo["res"], bo["base"], bo["val_best"], bo["val_base"]
        bst = res["best"]
        imp = (v0["total_cost"] - vb["total_cost"]) / max(v0["total_cost"], 1e-9)
        st.markdown(ui.kpi_row([
            ui.kpi("Optimal reorder point s*", fmt(bst["s"]), f"baseline {b0['s']:,.0f}", "◆", T["primary"], "units"),
            ui.kpi("Implied safety stock", fmt(bst["s"] - bo["ltd_mean"]), f"s* − E[LTD] ({bo['ltd_mean']:,.0f})", "⛨", T["blue"], "units"),
            ui.kpi("Optimal order qty Q*", fmt(bst["Q"]), f"baseline EOQ {b0['Q']:,.0f}", "▣", T["amber"], "units"),
            ui.kpi("Annual cost (validated)", f"£{vb['total_cost']:,.0f}", (f"<span class='up'>−{imp:.1%}</span>" if imp > 0 else f"<span class='down'>{-imp:+.1%}</span>") + f" vs baseline £{v0['total_cost']:,.0f}", "£", T["green"] if imp > 0 else T["red"]),
        ]) + ui.kpi_row([
            ui.kpi("Fill rate β (validated)", f"{vb['fill_rate']:.2%}", f"target ≥ {bo['cp'].target_fill:.1%} · baseline {v0['fill_rate']:.2%}", "◐", T["green"] if vb["fill_rate"] >= bo["cp"].target_fill else T["red"]),
            ui.kpi("Orders per year", f"{vb['orders_per_year']:,.1f}", f"baseline {v0['orders_per_year']:,.1f}", "⇄", T["blue"]),
            ui.kpi("Peak stock (P95)", fmt(vb["peak_on_hand_p95"]), f"capacity {bo['cp'].capacity:,.0f}", "▥", T["green"] if vb["peak_on_hand_p95"] <= bo["cp"].capacity else T["red"], "units"),
            ui.kpi("Feasible trials", f"{res['trials']['feasible'].mean():.0%}", f"{len(res['trials'])} evals · {res['seconds']:.1f}s · {bo['method'].split(' (')[0]}", "✓", T["primary"]),
        ]), unsafe_allow_html=True)
        tr = res["trials"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr["trial"], y=tr["objective"], mode="markers", name="Trial objective",
                                     marker=dict(size=8, color=np.where(tr["feasible"], T["blue"], T["red"]), symbol=np.where(tr["feasible"], "circle", "x"))))
            fig.add_trace(go.Scatter(x=tr["trial"], y=tr["best_so_far"], mode="lines", line=dict(color=T["primary"], width=3, shape="hv"), name="Best so far"))
            fig.add_hline(y=b0["total_cost"], line=dict(color=T["amber"], dash="dash"), annotation_text="baseline")
            fig.update_layout(title="Optimisation convergence", xaxis_title="Evaluation", yaxis_title="Penalised annual cost (£)", yaxis_type="log")
            fig_(fig, 380)
        with c2:
            fig = go.Figure()
            if res.get("surface") is not None:
                sf = res["surface"]
                fig.add_trace(go.Contour(x=sf["s"], y=sf["Q"], z=np.log10(sf["cost"]), colorscale=[[0, T["primary"]], [0.5, T["blue"]], [1, T["surface2"]]],
                                         contours=dict(showlines=False), colorbar=dict(title="log₁₀ £", thickness=10), opacity=0.85,
                                         hovertemplate="s %{x:,.0f} · Q %{y:,.0f}<br>GP mean £%{customdata:,.0f}<extra></extra>", customdata=sf["cost"]))
            fig.add_trace(go.Scatter(x=tr["s"], y=tr["Q"], mode="markers+text", text=tr["trial"].astype(str), textfont=dict(size=8, color=T["muted"]),
                                     textposition="top center", marker=dict(size=8, color=np.where(tr["feasible"], T["text"], T["red"]), line=dict(color=T["bg"], width=1)), name="Evaluated"))
            fig.add_trace(go.Scatter(x=[b0["s"]], y=[b0["Q"]], mode="markers", marker=dict(symbol="diamond", size=15, color=T["amber"]), name="Baseline"))
            fig.add_trace(go.Scatter(x=[bst["s"]], y=[bst["Q"]], mode="markers", marker=dict(symbol="star", size=20, color=T["green"], line=dict(color=T["bg"], width=1)), name="Optimum"))
            fig.update_layout(title="Search space · GP surrogate & sampled points" if res.get("surface") is not None else "Search space · sampled points",
                              xaxis_title="Reorder point s", yaxis_title="Order quantity Q")
            fig_(fig, 380)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            comp_ = pd.DataFrame({"Policy": ["Baseline", "Optimised"], "Ordering": [v0["ordering_cost"], vb["ordering_cost"]],
                                  "Holding": [v0["holding_cost"], vb["holding_cost"]], "Shortage": [v0["shortage_cost"], vb["shortage_cost"]]})
            fig = go.Figure()
            for col, cc in zip(["Ordering", "Holding", "Shortage"], [T["blue"], T["primary"], T["red"]]):
                fig.add_trace(go.Bar(x=comp_["Policy"], y=comp_[col], name=col, marker_color=cc, marker_cornerradius=4))
            fig.update_layout(barmode="stack", title="Annual cost breakdown (validated)", yaxis_tickprefix="£")
            fig_(fig, 340)
        with c2:
            fig = px.scatter(tr, x="fill_rate", y="total_cost", color="feasible", color_discrete_map={True: T["primary"], False: T["red"]},
                             hover_data={"s": ":.0f", "Q": ":.0f"})
            fig.add_vline(x=bo["cp"].target_fill, line=dict(color=T["amber"], dash="dash"), annotation_text="β*")
            fig.update_layout(title="Cost vs service trade-off", xaxis_tickformat=".1%", yaxis_tickprefix="£")
            fig_(fig, 340)
        with c3:
            chk = pd.DataFrame([
                {"Constraint": "Fill rate ≥ target", "Limit": f"{bo['cp'].target_fill:.1%}", "Optimised": f"{vb['fill_rate']:.2%}", "Baseline": f"{v0['fill_rate']:.2%}", "OK": vb["fill_rate"] >= bo["cp"].target_fill},
                {"Constraint": "Peak stock ≤ capacity", "Limit": f"{bo['cp'].capacity:,.0f}", "Optimised": f"{vb['peak_on_hand_p95']:,.0f}", "Baseline": f"{v0['peak_on_hand_p95']:,.0f}", "OK": vb["peak_on_hand_p95"] <= bo["cp"].capacity},
                {"Constraint": "Avg value ≤ budget", "Limit": f"£{bo['cp'].budget:,.0f}", "Optimised": f"£{vb['avg_inventory_value']:,.0f}", "Baseline": f"£{v0['avg_inventory_value']:,.0f}", "OK": vb["avg_inventory_value"] <= bo["cp"].budget},
                {"Constraint": "Q ≥ MOQ", "Limit": f"{bo['cp'].moq:,.0f}", "Optimised": f"{bst['Q']:,.0f}", "Baseline": f"{b0['Q']:,.0f}", "OK": bst["Q"] >= bo["cp"].moq},
            ])
            st.markdown(ui.section("Constraint validation", "independent stream"), unsafe_allow_html=True)
            st.dataframe(chk, hide_index=True, width="stretch", column_config={"OK": st.column_config.CheckboxColumn("✓")})
            st.caption(f"Run {bo['at']}. Validation re-simulates both policies on a fresh seed with ≥300 replications.")
        with st.expander("Baseline vs optimised — interpretation"):
            st.markdown(f"""
The **baseline** sets s to the analytical ROP ({b0['s']:,.0f}) and Q to the EOQ ({b0['Q']:,.0f}). It treats service level as an input and
ignores the cost of shortages, MOQ, capacity and budget. The **optimised** policy lets the Bayesian optimiser trade holding against
ordering and shortage cost directly: s* = {bst['s']:,.0f}, Q* = {bst['Q']:,.0f}. Larger Q raises cycle stock but reduces the number of
exposures to lead-time risk per year, which is why fill rate (β) can rise even when s falls. Validated annual cost changes by
**{-imp:+.1%}** on an independent simulation.""")
        st.download_button("⬇ Download trial history (CSV)", tr.to_csv(index=False).encode(), f"bayes_opt_trials_{FOCUS}.csv", "text/csv")

    # ---------------- Before AI: manual grid sweep vs Bayesian optimisation
    st.markdown(ui.section("Manual sweep (before AI) vs Bayesian optimisation", "Same simulator, same random numbers, same bounds", "before / after"), unsafe_allow_html=True)
    bo = ss.get("bo")
    if not bo or bo["sku"] != FOCUS:
        st.info("Run the Bayesian optimisation above first; the sweep reuses its scenario.")
    else:
        c1, c2 = st.columns([1, 2.2], gap="medium", vertical_alignment="center")
        g = c1.select_slider("Grid resolution (points per axis)", [5, 8, 10, 12, 15], 10, key="bo_grid")
        if c1.button(f"Run manual sweep ({g * g} evaluations)", width="stretch"):
            sim = O.PolicySimulator(*bo["sim_args"], **bo["sim_kw"])
            S_ = np.linspace(*bo["bounds"]["s"], g); Q_ = np.linspace(*bo["bounds"]["Q"], g)
            rows, bar = [], st.progress(0.0)
            for i, s_ in enumerate(S_):
                for q_ in Q_:
                    r_ = sim.run(round(s_), round(max(q_, bo["cp"].moq)), bo["cp"])
                    rows.append({**r_, "objective": O.penalised(r_, bo["base"]["total_cost"])})
                bar.progress((i + 1) / g)
            bar.empty()
            gs = pd.DataFrame(rows); gs["eval"] = np.arange(1, len(gs) + 1); gs["best_so_far"] = gs["objective"].cummin()
            ss.bo_grid_res = {"df": gs, "sku": FOCUS, "g": g}
        gr = ss.get("bo_grid_res")
        if gr and gr["sku"] == FOCUS:
            gs, tr = gr["df"], bo["res"]["trials"]
            gb, bb = gs[gs.feasible].total_cost.min(), bo["res"]["best"]["total_cost"]
            hit = tr.index[tr["best_so_far"] <= gs["objective"].min() * 1.001]
            n_bo = int(tr.loc[hit[0], "trial"]) if len(hit) else None
            c2.markdown(ui.kpi_row([
                ui.kpi("Manual sweep best", f"£{gb:,.0f}", f"after {len(gs)} evaluations", "▦", T["muted"]),
                ui.kpi("Bayesian best", f"£{bb:,.0f}", f"after {len(tr)} evaluations", "★", T["primary"]),
                ui.kpi("Evaluations saved", f"{(1 - len(tr) / len(gs)):.0%}", (f"BO matched the sweep at evaluation {n_bo}" if n_bo else "BO found a different optimum") , "⚡", T["green"]),
            ], 3), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gs["eval"], y=gs["best_so_far"], name=f"Manual grid sweep ({gr['g']}×{gr['g']})", line=dict(color=T["muted"], width=2.5, shape="hv")))
            fig.add_trace(go.Scatter(x=tr["trial"], y=tr["best_so_far"], name="Bayesian optimisation", line=dict(color=T["primary"], width=3, shape="hv")))
            fig.update_layout(title="Best cost found vs number of evaluations", xaxis_title="Evaluations", yaxis_title="Penalised annual cost (£)", yaxis_type="log")
            fig_(fig, 340)

    # ---------------- Automation: tune every SKU
    st.markdown(ui.section("Auto-tune the whole portfolio", "Bayesian optimisation (TPE) of (s, Q) for every SKU in one click", "automation"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2], gap="medium", vertical_alignment="bottom")
    pt_trials = c1.select_slider("Evaluations per SKU", [10, 15, 20, 30], 20, key="pt_trials")
    pt_fill = c2.slider("Min fill rate", 0.85, 0.99, 0.95, 0.01, key="pt_fill")
    if c3.button(f"⚙ Auto-tune all {len(P)} SKUs", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Optimising…")
        ss.portfolio_tune = A.tune_portfolio(P, L_wk, sdL_wk, ss.holding_rate, ss.order_cost, ss.short_mult, pt_fill, pt_trials,
                                             progress=lambda i, n, s_: bar.progress(i / n, text=f"Bayesian optimisation {i}/{n} · {s_}"))
        bar.empty()
    pt = ss.get("portfolio_tune")
    if pt is not None:
        st.markdown(ui.kpi_row([
            ui.kpi("Cost · formula policy", f"£{pt.cost_formula.sum():,.0f}", "analytical ROP + EOQ (per year)", "▦", T["muted"]),
            ui.kpi("Cost · AI-tuned", f"£{pt.cost_opt.sum():,.0f}", f"<span class='up'>−{1 - pt.cost_opt.sum() / pt.cost_formula.sum():.1%}</span>", "★", T["primary"]),
            ui.kpi("SKUs improved", f"{(pt.saving_pct > 0.5).sum()}/{len(pt)}", f"median saving {pt.saving_pct.median():.1f}%", "✓", T["green"]),
            ui.kpi("Constraint-feasible", f"{pt.feasible.mean():.0%}", f"fill rate ≥ {pt_fill:.0%}", "◐", T["blue"]),
        ]), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=pt.sort_values("saving_pct")["sku"], y=pt.sort_values("saving_pct")["saving_pct"],
                               marker=dict(color=np.where(pt.sort_values("saving_pct")["saving_pct"] > 0, T["primary"], T["red"]), cornerradius=4),
                               hovertext=pt.sort_values("saving_pct")["description"]))
        fig.update_layout(title="Annual cost saving per SKU from Bayesian tuning (%)", xaxis_tickangle=-60)
        fig_(fig, 340, legend=False)
        st.download_button("⬇ Tuned policies for all SKUs (CSV)", pt.to_csv(index=False).encode(), "ai_tuned_policies.csv", "text/csv")

# ================================================================== TAB 6 · DATA & REPORTS
with tabs[5]:
    v6 = st.segmented_control("Data & reports view", ["Data manager", "Reports & exports", "Assignment map & method"],
                              default="Data manager", key="v6_view", label_visibility="collapsed") or "Data manager"
    if v6 == "Data manager":
        st.markdown(ui.section("Data manager", "Add new data → validate → reconcile → re-forecast → recompute buffers & recommendations", "ingestion"), unsafe_allow_html=True)
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown(f"""<div class='card'><h4>Active dataset</h4>
            <div style='font-size:.86rem'><b>{ss.data_source}</b><br>
            {len(ss.df):,} rows · {ss.df['sku'].nunique()} SKUs · {ss.df['date'].min():%d %b %Y} → {ss.df['date'].max():%d %b %Y}<br>
            Last successful ingestion: <b>{ss.last_ingest}</b> (on-demand batch refresh — not a real-time feed)</div>
            <div style='font-size:.8rem;color:var(--muted);margin-top:8px'>Source: Chen, D. (2019) <i>Online Retail II</i>, UCI Machine Learning Repository,
            doi:10.24432/C5CG6D, CC BY 4.0 — 1,067,371 transactions of a UK-based online gift wholesaler, Dec 2009 – Dec 2011. Cleaned (cancellations,
            returns, non-product codes and 99.9th-pct bulk outliers removed) and aggregated to daily demand for the 40 highest-volume SKUs sold on ≥400 days.</div></div>""", unsafe_allow_html=True)
            prov = pd.DataFrame([
                ["Daily demand (units), SKU, description, date", "Observed", "UCI Online Retail II invoices"],
                ["Average selling price", "Observed", "Invoice line price, quantity-weighted"],
                ["Weekly/monthly series, calendar & Q4-peak features", "Derived", "Aggregation & feature engineering"],
                ["Forecasts, σ of forecast error, SS, ROP, EOQ, risk", "Derived", "Models & formulas in this app"],
                ["Service level, lead time & σ, costs, capacity, budget, MOQ", "Configured", "Sidebar / tab controls"],
                ["Stock on hand, on order, backorders", "Simulated", "Seeded scenario — not in source data"],
                ["Promotions", "Not available", "No promo flag in source; price used as proxy, nothing fabricated"],
            ], columns=["Field", "Type", "Origin"])
            st.dataframe(prov, hide_index=True, width="stretch")
        with c2:
            weekly_all = ss.df.groupby([pd.Grouper(key="date", freq="W-SUN"), "sku"])["demand"].sum().reset_index()
            cov = weekly_all.groupby("date")["sku"].nunique()
            fig = go.Figure(go.Bar(x=cov.index, y=cov.values, marker_color=T["primary"]))
            fig.update_layout(title="Data coverage: SKUs with sales per week")
            fig_(fig, 300, legend=False)
            tmpl = pd.DataFrame({"date": ["2011-12-12", "2011-12-12"], "sku": ["85099B", "84879"], "demand": [120, 64],
                                 "description": ["JUMBO BAG RED RETROSPOT", "ASSORTED COLOUR BIRD ORNAMENT"], "avg_price": [1.95, 1.69]})
            st.download_button("⬇ Download CSV template", tmpl.to_csv(index=False).encode(), "new_data_template.csv", "text/csv", width="stretch")

        st.markdown(ui.section("Upload new observations", "CSV · automatic column detection with manual override"), unsafe_allow_html=True)
        up = st.file_uploader("CSV file (transactions or daily demand)", type=["csv"], key="uploader")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:
                raw = None
                st.error(f"Could not read CSV: {e}")
            if raw is not None:
                guess = D.detect_columns(raw.columns)
                st.caption(f"{len(raw):,} rows · columns: {', '.join(map(str, raw.columns[:12]))}")
                cc = st.columns(5)
                opts = ["— none —"] + list(raw.columns)
                mapping = {}
                for i, fld in enumerate(D.REQUIRED + D.OPTIONAL):
                    g = guess.get(fld)
                    sel = cc[i].selectbox(f"{fld}{' *' if fld in D.REQUIRED else ''}", opts, index=opts.index(g) if g in opts else 0, key=f"map_{fld}")
                    if sel != "— none —":
                        mapping[fld] = sel
                new, rep = D.validate(raw, mapping)
                if not rep.ok:
                    st.error(" · ".join(rep.messages) or "Validation failed")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Valid rows", f"{rep.rows_valid:,}", f"of {rep.rows_in:,} input")
                    m2.metric("SKUs", rep.skus, f"{len(set(new['sku']) - set(ss.df['sku']))} new")
                    m3.metric("Date range", rep.date_min, rep.date_max, delta_color="off")
                    m4.metric("Overlapping SKU-days", f"{len(ss.df.merge(new[['date', 'sku']], on=['date', 'sku'])):,}")
                    for msg in rep.messages:
                        st.warning(msg)
                    st.dataframe(new.head(50), width="stretch", height=200)
                    mode_ = st.radio("On overlap with existing (date, SKU)", ["replace", "add"], horizontal=True,
                                     format_func=lambda m: "Replace with new values (corrections)" if m == "replace" else "Add to existing (late transactions)")
                    if st.button("✅ Ingest & refresh all models", type="primary"):
                        merged, info = D.merge_new(ss.df, new, mode_)
                        ss.df = merged
                        ss.last_ingest = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ss.data_source = ss.data_source.split(" + ")[0] + " + user upload"
                        ss.ingest_log.append({"time": ss.last_ingest, "event": f"Upload '{up.name}' ({mode_})", "rows": rep.rows_valid, **info})
                        ss.pop("bo", None); ss.pop("mc_params", None)
                        st.success(f"Ingested {info['added_rows']:,} new and reconciled {info['reconciled_rows']:,} existing rows. Forecasts, safety stock, ROP and recommendations refreshed.")
                        st.rerun()

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown(ui.section("Add a single observation", "e.g. today's sales for a SKU"), unsafe_allow_html=True)
            with st.form("manual_add", clear_on_submit=True):
                a, b, c_ = st.columns(3)
                dt_ = a.date_input("Date", value=(ss.df["date"].max() + pd.Timedelta(days=1)).date())
                sk_ = b.selectbox("SKU", sku_opts, format_func=lambda s: f"{s} · {str(labels.get(s, ''))[:22]}")
                qt_ = c_.number_input("Units sold", 0.0, value=0.0, step=1.0)
                if st.form_submit_button("➕ Add observation & refresh", type="primary"):
                    new = pd.DataFrame({"date": [pd.Timestamp(dt_)], "sku": [sk_], "demand": [qt_], "description": [labels.get(sk_)],
                                        "avg_price": [ss.df.loc[ss.df.sku == sk_, "avg_price"].iloc[-1]]})
                    ss.df, info = D.merge_new(ss.df, new, "replace")
                    ss.last_ingest = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ss.ingest_log.append({"time": ss.last_ingest, "event": f"Manual entry {sk_} {dt_} = {qt_:,.0f}", "rows": 1, **info})
                    ss.pop("bo", None)
                    st.rerun()
        with c2:
            st.markdown(ui.section("Ingestion log", "this session only"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(ss.ingest_log).iloc[::-1], hide_index=True, width="stretch", height=180)
            if st.button("↺ Restore original dataset", width="stretch"):
                ss.df = D.load_default(); ss.data_source = "UCI Online Retail II (processed: 40 SKUs, daily demand)"
                ss.last_ingest = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ss.ingest_log.append({"time": ss.last_ingest, "event": "Restored original dataset", "rows": len(ss.df)})
                ss.pop("bo", None); ss.pop("mc_params", None)
                st.rerun()
    elif v6 == "Reports & exports":
        st.markdown(ui.section("Reports & model diagnostics", "All outputs downloadable · computed live"), unsafe_allow_html=True)
        y_, pr_, bt_ = run_backtest(ss.df, FOCUS, "W", 8, tuple(F.MODELS), 3)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            pw = P.sort_values("fc_wape")
            fig = go.Figure(go.Bar(x=pw["sku"], y=pw["fc_wape"], marker=dict(color=pw["fc_wape"], colorscale=[[0, T["primary"]], [0.5, T["amber"]], [1, T["red"]]], cornerradius=4),
                                   hovertext=pw["description"]))
            fig.add_hline(y=pw["fc_wape"].median(), line=dict(color=T["text"], dash="dot"), annotation_text=f"median {pw['fc_wape'].median():.1f}%")
            fig.update_layout(title="Forecast accuracy across portfolio (8-week hold-out WAPE, damped ETS)", xaxis_tickangle=-60)
            fig_(fig, 360, legend=False)
        with c2:
            resid = bt_["test_actual"] - bt_["test_preds"][bt_["best"]]
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Test residuals over time", "Residual distribution"))
            fig.add_trace(go.Bar(x=resid.index, y=resid.values, marker_color=np.where(resid.values > 0, T["blue"], T["amber"])), 1, 1)
            fig.add_trace(go.Histogram(x=resid.values, nbinsx=10, marker_color=T["primary"]), 1, 2)
            fig.update_layout(title=f"{FOCUS} · {bt_['best']} residual diagnostics")
            fig_(fig, 360, legend=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(ui.section("Forecast accuracy report", f"{FOCUS} · weekly · 8-week test"), unsafe_allow_html=True)
            rep_fc = bt_["validation"].drop(columns="resid_sd").add_prefix("val_").join(bt_["test"].add_prefix("test_")).round(2)
            st.dataframe(rep_fc, width="stretch")
        with c2:
            st.markdown(ui.section("Safety-stock comparison", "analytical vs Monte Carlo (Safety Stock tab inputs)"), unsafe_allow_html=True)
            cmp_ = ss.get("ssc_compare")
            if cmp_ is not None:
                st.dataframe(cmp_.round(1), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(ui.section("Monte Carlo scenario results"), unsafe_allow_html=True)
            if ss.get("mc_scenarios") is not None:
                st.dataframe(ss["mc_scenarios"].round(3), hide_index=True, width="stretch")
        with c2:
            st.markdown(ui.section("Optimisation summary"), unsafe_allow_html=True)
            bo = ss.get("bo")
            if bo:
                st.dataframe(pd.DataFrame({"Baseline": bo["val_base"], "Optimised": bo["val_best"]}).drop(index=["viol_fill", "viol_capacity", "viol_budget"]).astype(str),
                             width="stretch", height=300)
            else:
                st.info("Run the Bayesian Optimization tab to populate this section.")
        st.markdown(ui.section("Dataset quality report", "per SKU · daily level"), unsafe_allow_html=True)
        dq = D.dataset_quality(ss.df)
        st.dataframe(dq, hide_index=True, width="stretch", height=300)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({"Field": ["Student", "ID", "Programme", "Course", "Generated", "Service level", "Lead time (d)", "Lead-time σ (d)", "Demand basis"],
                          "Value": [STUDENT["name"], STUDENT["id"], f"{STUDENT['prog']} {STUDENT['batch']}", f"{STUDENT['course']} — {STUDENT['lab']}",
                                    datetime.now().strftime("%Y-%m-%d %H:%M"), svc, ss.lt_days, ss.lt_sd_days, ss.basis]}).to_excel(xw, sheet_name="Cover", index=False)
            P[COLS].to_excel(xw, sheet_name="Inventory", index=False)
            rep_fc.to_excel(xw, sheet_name="Forecast accuracy")
            if cmp_ is not None:
                cmp_.to_excel(xw, sheet_name="SS comparison", index=False)
            if ss.get("mc_scenarios") is not None:
                ss["mc_scenarios"].to_excel(xw, sheet_name="MC scenarios", index=False)
            if bo:
                bo["res"]["trials"].to_excel(xw, sheet_name="BO trials", index=False)
            dq.to_excel(xw, sheet_name="Data quality", index=False)
        c = st.columns(4)
        c[0].download_button("⬇ Full report (Excel)", buf.getvalue(), f"AI_Inventory_Report_{STUDENT['id']}.xlsx", type="primary", width="stretch")
        c[1].download_button("⬇ Inventory CSV", P[COLS].to_csv(index=False).encode(), "inventory.csv", width="stretch")
        c[2].download_button("⬇ Forecast accuracy CSV", rep_fc.to_csv().encode(), f"forecast_accuracy_{FOCUS}.csv", width="stretch")
        c[3].download_button("⬇ Data quality CSV", dq.to_csv(index=False).encode(), "data_quality.csv", width="stretch")
    else:
        st.markdown(ui.section("Assignment → implementation map", "Session II: AI Applications in Reorder Point & Inventory Management — Exercise"), unsafe_allow_html=True)
        st.markdown("""<div class='card'><b>Objective (from the brief):</b> enhance the reorder-point calculator and sensitivity analysis by introducing
        predictive intelligence, adaptive decision-making and real-time optimisation.</div>""", unsafe_allow_html=True)
        req = pd.DataFrame([
            ["1. Demand forecasting", "ML models (time series, regression, LSTM)", "Naive, Seasonal Naive, MA, ETS, ARIMA/SARIMA, Gradient Boosting regression, MLP neural net (LSTM assessed — see note)", "Demand Forecasting"],
            ["1. Demand forecasting", "Historical data, seasonality, promotions, external factors", "2 years of real invoices; STL + Fourier calendar features; price as promo proxy; Q4 peak flag", "Demand Forecasting"],
            ["1. Demand forecasting", "Possibility of adding new data", "CSV upload with auto/manual column mapping, validation, dedupe, reconciliation; manual entry", "Data Manager"],
            ["1. Demand forecasting", "Continuously update forecasts", "Content-hashed caches → every ingestion retrains, re-backtests and refreshes all KPIs; timestamped", "All tabs"],
            ["2. Dynamic safety stock", "Probabilistic model: demand & lead-time variability", "SS = Z√(σ²_D·L + D̄²·σ²_L) with σ_D from live forecast error (dynamic)", "Safety Stock & ROP"],
            ["2. Dynamic safety stock", "Monte Carlo risk scenarios & buffers", "10k–100k draws, 5 demand × 5 lead-time distributions, copula dependence, 5 stress scenarios", "Monte Carlo Risk"],
            ["3. AI sensitivity analysis", "Bayesian optimisation of ROP under constraints", "GP-EI (Matern-5/2) and Optuna TPE over (s, Q) with fill-rate, capacity, budget, MOQ constraints; CRN + independent validation", "Bayesian Optimization"],
            ["AI & automation", "Show what AI adds vs the pre-AI method", "Replay on 26 real weeks: textbook static policy vs self-retraining, bias-correcting, conformal, cost-aware AI policy", "Before vs After AI"],
            ["AI & automation", "Adaptive decision-making", "Drift detection (tracking signal) → automatic champion/challenger retraining; Isolation Forest anomalies; ABC-XYZ + K-means segment service levels", "AI Autopilot"],
            ["AI & automation", "Real-time optimisation / automation", "Replenishment agent drafts POs with rationale on every data load; portfolio-wide Bayesian tuning; auto exception digest", "AI Autopilot, Bayesian Optimization"],
            ["Condition", "No sharing of data with peers", "Local single-user session; uploads held in session memory only; no telemetry; no external AI service", "Sidebar note"],
        ], columns=["Workstream", "Requirement", "Implementation", "Where"])
        st.dataframe(req, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Key formulas")
            st.latex(r"Z = \Phi^{-1}(\alpha)\qquad \sigma_{LTD}=\sqrt{\sigma_D^2L+\bar D^2\sigma_L^2}")
            st.latex(r"SS = Z\,\sigma_{LTD}\qquad ROP=\bar D L + SS\qquad EOQ=\sqrt{2DK/h}")
            st.latex(r"P(\text{stockout}) = 1-\Phi\!\left(\tfrac{IP-\bar D L}{\sigma_{LTD}}\right)\qquad \beta = 1-\tfrac{\sigma_{LTD}\,G(Z)}{Q}")
            st.latex(r"\text{WAPE}=\tfrac{\sum|y-\hat y|}{\sum|y|}\qquad \text{EI}(x)=\mathbb{E}[\max(f^{*}-f(x),0)]")
        with c2:
            st.markdown("""#### Assumptions & limitations
    - Time unit for inventory maths = **week**; lead time entered in days and converted (÷7).
    - Analytical formula assumes i.i.d. demand, lead time independent of demand, Normal LTD → relaxed by Monte Carlo (Gamma, skewed lead times, ρ > 0).
    - Source data has **no stock, supplier lead-time, PO or promotion fields** — these are configured or simulated and labelled so.
    - Policy simulation: continuous review (s, Q), daily periods, lost sales, Gamma demand, integer-day lead times.
    - Refresh is **on-demand batch**, not a real-time stream.
    - LSTM not deployed: ≈105 weekly points per SKU is too short; MLP neural net included and compared fairly.""")
            st.markdown("""#### References
    - Chen, D. (2019). *Online Retail II*. UCI ML Repository. doi:10.24432/C5CG6D (CC BY 4.0)
    - Silver, Pyke & Thomas (2017). *Inventory and Production Management in Supply Chains*, 4th ed.
    - Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd ed.
    - Makridakis, Spiliotis & Assimakopoulos (2022). M5 accuracy competition. *IJF* 38(4).
    - Jones, Schonlau & Welch (1998). Efficient global optimisation of expensive black-box functions. *J. Global Optim.*
    - Akiba et al. (2019). Optuna: a next-generation hyperparameter optimisation framework. *KDD*.""")

st.markdown(f"""<div class='footer'><span>© 2026 · <b>{STUDENT['course']}</b> — done by <b>{STUDENT['name']}</b> · {STUDENT['id']} · {STUDENT['prog']} {STUDENT['batch']} · {STUDENT['school']}</span>
<span>Data: UCI Online Retail II (CC BY 4.0) · Individual work — no data shared with peers</span></div>""", unsafe_allow_html=True)
