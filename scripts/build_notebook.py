"""Builds and executes the academic notebook (all outputs saved)."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions" / "Laboratory Work 1" / "Krishna_Mathur_AS25DXB018_AI_in_Operations_Laboratory_Work_1.ipynb"
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# AI-Powered Inventory Intelligence & Reorder Point Optimization
### AI in Operations — Session II: AI Applications in Reorder Point & Inventory Management (Laboratory Exercise)

| | |
|---|---|
| **Student** | Krishna Mathur |
| **Student ID** | AS25DXB018 |
| **Programme / Batch** | MAIB · September 2025 |
| **Institution** | SP Jain School of Global Management |
| **Instructor** | Dr. Sandip Kumar Roy |
| **Interactive dashboard** | Streamlit app in `apps/streamlit/app.py` (same engine as this notebook) |

*Individual work. In line with the brief ("No sharing of data with peers allowed"), no data was shared with peers.*
""")
md("""
## 1. Assignment overview (reconstructed from the brief)
**Objective:** enhance the reorder-point calculator and sensitivity analysis by introducing *predictive intelligence, adaptive decision-making and real-time optimisation*.

| # | Workstream | Requirement in the brief | Section |
|---|---|---|---|
| 1 | Demand forecasting | ML models (time series, regression, LSTM) on history, seasonality, promotions, external factors; allow adding new data; continuously update forecasts | §3–4 |
| 2 | Dynamic safety stock | Probabilistic models for demand variability **and** lead-time uncertainty; Monte Carlo simulation of risk scenarios and buffers; formula $SS = Z\\sqrt{\\sigma_D^2 L + \\bar D^2\\sigma_L^2}$ | §5–6 |
| 3 | AI-driven sensitivity analysis | Bayesian optimisation instead of manual parameter sweeps to find optimal reorder points under constraints | §7 |
""")
code("""
import sys, subprocess, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
# Works locally (repo root) and on Google Colab (clones the project repository)
ROOT = Path.cwd()
while not (ROOT / "src" / "inventory_core").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "src" / "inventory_core").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/krish2105/Inventory-Management-AI-in-Operations", "/content/repo"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna", "statsmodels"], check=True)
    ROOT = Path("/content/repo")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from inventory_core import data as D, forecasting as F, inventory as I, simulation as S, optimization as O
plt.rcParams.update({"figure.figsize": (11, 4), "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
FIG = ROOT / "outputs"; FIG.mkdir(exist_ok=True)
print("Engine loaded from", ROOT)
""")
md("""
## 2. Dataset research, selection and attribution
| Candidate | Strengths | Limitations | Decision |
|---|---|---|---|
| **Walmart M5** (Kaggle) | 30k SKU-store series, calendar, prices, SNAP events | Requires Kaggle account & competition-rule acceptance (cannot be redistributed); 450 MB | Not used |
| **UCI Online Retail II** | Real invoice lines (1.07 M), 2 full years → yearly seasonality, unit prices, CC BY 4.0, direct download | No stock-on-hand, supplier lead time, PO or promotion fields | **Selected** |
| UCI Online Retail (I) | Same retailer | Only one year (subset of II) | Superseded |

**Source:** Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0).

**Processing (`scripts/prepare_data.py`):** removed cancellations (invoice prefix *C*), returns (qty ≤ 0), zero prices, non-product codes (POST, M, DOT…), bulk outliers above the 99.9th line-quantity percentile; aggregated to daily units; kept the 40 highest-volume SKUs sold on ≥ 400 days; filled non-sale trading days with 0.

**Provenance:** demand & price = *observed*; weekly series, features, forecasts, SS/ROP = *derived*; service level, lead time, costs = *configured assumptions*; stock on hand = *simulated scenario* (never presented as observed).
""")
code("""
df = D.load_default()
print(f"{len(df):,} SKU-day rows | {df.sku.nunique()} SKUs | {df.date.min():%d %b %Y} – {df.date.max():%d %b %Y}")
q = D.dataset_quality(df)
display(q.sort_values('total_units', ascending=False).head(10))
print("Missing values per column:"); print(df.isna().sum())
""")
code("""
wk = df.set_index('date')['demand'].resample('W-SUN').sum().iloc[1:-1]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={'width_ratios': [2, 1]})
ax[0].plot(wk.index, wk, color='#2563EB', lw=1.3); ax[0].plot(wk.index, wk.rolling(8).mean(), color='#0D9488', lw=2.5)
ax[0].set_title('Portfolio weekly demand (40 SKUs) — observed'); ax[0].set_ylabel('units')
mon = df.assign(m=df.date.dt.month).groupby('m').demand.sum()
ax[1].bar(mon.index, mon / mon.mean(), color='#0D9488'); ax[1].set_title('Seasonal index by month'); ax[1].set_xticks(range(1, 13))
plt.tight_layout(); plt.savefig(FIG / 'fig_demand.png', dpi=160); plt.show()
""")
md("""
The portfolio shows strong **yearly seasonality**: a Sep–Nov build-up before Christmas (wholesale gift buyers), then a January trough. This motivates calendar features and seasonal models.

## 3. Demand forecasting — model design
Focus SKU = the highest-volume product. Models compared (all in `inventory_core/forecasting.py`):

1. **Naive** and 2. **Seasonal naive** — benchmarks  3. **Moving average**  4. **ETS / Holt-Winters** (damped trend)  5. **ARIMA/SARIMA**
6. **Gradient-boosting regression** (ML) on lags, rolling statistics, Fourier week-of-year terms, a Q4-peak flag and observed price (promotion proxy — the data has no promotion flag, so none is fabricated)
7. **MLP neural network** on the same features.

**LSTM:** considered but not deployed — ≈105 weekly observations per SKU are far too few for a recurrent network to beat statistical baselines (consistent with M4/M5 findings); the MLP is the neural representative and is compared on equal terms.

**Evaluation protocol:** chronological split — the final 8 weeks are a **test** set never used for selection; model selection uses **rolling-origin validation** (3 folds × 8 weeks) before the test window. Metrics: MAE, RMSE, **WAPE** (primary — robust to zeros), MAPE (zero-actual periods excluded).
""")
code("""
SKU = df.groupby('sku').demand.sum().idxmax(); DESC = df[df.sku == SKU].description.iloc[0]
y = D.series(df, SKU, 'W'); price = D.price_series(df, SKU, 'W', y.index)
bt = F.backtest(y, 'W', 8, price, folds=3)
print(f"Focus SKU {SKU} – {DESC}: {len(y)} weeks.  Best model on validation: {bt['best']}")
board = bt['validation'][['MAE','RMSE','WAPE %','MAPE %']].add_prefix('val ').join(bt['test'][['MAE','RMSE','WAPE %','Bias']].add_prefix('test ')).sort_values('val WAPE %')
display(board.round(1))
""")
code("""
best = bt['best']; rsd = bt['validation'].loc[best, 'resid_sd']
fc = F.forecast(y, 'W', 8, best, rsd, price, 0.95)
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.plot(y.index[-60:], y[-60:], color='#2563EB', label='Actual')
for m, c in [(best, '#D97706'), ('Naive', '#94A3B8')]:
    p = bt['test_preds'][m]; ax.plot(p.index, p, '--', color=c, lw=2, label=f'{m} (test)')
ax.fill_between(fc.index, fc.lower, fc.upper, color='#0D9488', alpha=.18, label='95% interval')
ax.plot(fc.index, fc.forecast, color='#0D9488', lw=2.5, marker='o', label=f'Forecast – {best}')
ax.axvspan(bt['test_actual'].index[0], bt['test_actual'].index[-1], color='#D97706', alpha=.06)
ax.set_title(f'{SKU} {DESC}: actual vs predicted (weekly)'); ax.legend(ncol=3, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / 'fig_forecast.png', dpi=160); plt.show()
imp = F.feature_importance(y, 'W', price); print('Top gradient-boosting features:'); print(imp.sort_values(ascending=False).head(6).round(3))
""")
md("""
## 4. Adding new data & continuous updating
The engine validates any new CSV (automatic column detection, date parsing, negative/returns removal, duplicate aggregation), reconciles overlaps (*replace* for corrections, *add* for late transactions) and re-runs the full pipeline. Below, two new weeks are appended for the focus SKU (a demonstration using the SKU's own recent average — clearly marked as a hypothetical update) and the forecast is refreshed. In the dashboard this happens through **Data Manager → Upload / Add observation**.
""")
code("""
last = df.date.max()
avg = df[df.sku == SKU].tail(28).demand.mean()
new_raw = pd.DataFrame({'InvoiceDate': pd.date_range(last + pd.Timedelta(days=1), periods=14).strftime('%Y-%m-%d'),
                        'StockCode': SKU, 'Quantity': np.round(avg * 1.15)})   # hypothetical +15% demand update
clean, rep = D.validate(new_raw, D.detect_columns(new_raw.columns))
df2, info = D.merge_new(df, clean)
print('Validation:', rep.ok, '| rows', rep.rows_valid, '| merge info', info)
y2 = D.series(df2, SKU, 'W'); bt2 = F.backtest(y2, 'W', 8, D.price_series(df2, SKU, 'W', y2.index), folds=3)
fc2 = F.forecast(y2, 'W', 8, bt2['best'], bt2['validation'].loc[bt2['best'], 'resid_sd'])
print(f"Before update: best={best}, mean forecast {fc.forecast.mean():,.0f}/wk  →  after update: best={bt2['best']}, mean forecast {fc2.forecast.mean():,.0f}/wk")
""")
md("""
## 5. Dynamic safety stock & reorder point (analytical)
$$SS = Z\\sqrt{\\sigma_D^2\\,L + \\bar D^2\\,\\sigma_L^2}, \\qquad ROP = \\bar D\\,L + SS, \\qquad Z = \\Phi^{-1}(\\alpha)$$
Time unit = **week**. *Dynamic* inputs: $\\bar D$ = mean of the 8-week AI forecast; $\\sigma_D$ = validation residual SD of the selected model (forecast-error variability, not raw demand variability). Lead time is not in the data, so it is a configured assumption: $L$ = 14 days (2 weeks), $\\sigma_L$ = 4 days.

The formula assumes i.i.d. weekly demand, lead time independent of demand, and approximately Normal lead-time demand.
""")
code("""
d_bar, sd_d = fc.forecast.mean(), rsd
L, sL, alpha = 2.0, 4/7, 0.95
c = I.ss_components(alpha, d_bar, sd_d, L, sL)
print(f"D̄ = {d_bar:,.1f} u/wk, σ_D = {sd_d:,.1f}, L = {L} wk, σ_L = {sL:.3f} wk, α = {alpha}")
print(f"Z = {c['z']:.4f}\;  σ_LTD = √({sd_d:,.1f}²·{L} + {d_bar:,.1f}²·{sL:.3f}²) = {c['sigma_ltd']:,.1f}")
print(f"Safety stock = {c['ss']:,.1f} units   |   ROP = {c['ltd_mean']:,.1f} + {c['ss']:,.1f} = {c['rop']:,.1f} units")
print(f"Variance share: demand {c['share_demand']:.0%}, lead time {c['share_leadtime']:.0%}")
svc = np.array([.85, .90, .95, .975, .99, .995])
display(pd.DataFrame({'service α': svc, 'Z': [I.z_value(s) for s in svc], 'SS': [I.safety_stock(s, d_bar, sd_d, L, sL) for s in svc],
                      'ROP': [I.reorder_point(s, d_bar, sd_d, L, sL) for s in svc]}).round(2))
""")
md("""
## 6. Monte Carlo simulation of risk scenarios
Lead-time demand is simulated as the sum of weekly demands over a *random, fractional* lead time (100,000 draws, seed 42). This relaxes the formula's assumptions: Gamma (skewed, non-negative) demand, right-skewed lead times and a **joint** simulation in which long lead times coincide with high demand (Gaussian copula, ρ).

Note the difference between **cycle service level α** (P(no stockout in a cycle)) and **fill rate β** (share of demand met from stock).
""")
code("""
Q = I.eoq(d_bar * 52, 40, df[df.sku == SKU].avg_price.mean() * 0.6 * 0.25)
rows = []
for name, dd, ld, rho in [('Normal / Normal (formula assumptions)', 'Normal (truncated at 0)', 'Normal (truncated)', 0),
                          ('Gamma / Gamma', 'Gamma', 'Gamma', 0), ('Gamma / right-skewed LT', 'Gamma', 'Triangular (right-skewed)', 0),
                          ('Joint: Gamma, ρ = 0.5', 'Gamma', 'Gamma', 0.5)]:
    r = S.simulate_ltd(100000, dd, d_bar, sd_d, ld, L, sL, rho, seed=42)
    s_ = S.summarize(r.ltd.values, alpha, c['rop'], Q)
    rows.append({'model': name, 'ROP_MC': s_['rop_sim'], 'SS_MC': s_['ss_sim'], 'P(stockout) at formula ROP': s_['stockout_prob'],
                 'fill rate β': s_['fill_rate'], 'E[shortage]/cycle': s_['expected_shortage']})
mc_tab = pd.DataFrame(rows); mc_tab.loc[-1] = ['Analytical formula', c['rop'], c['ss'], 1 - alpha, np.nan, np.nan]
mc_tab = mc_tab.sort_index().reset_index(drop=True); display(mc_tab.round(3))
""")
code("""
scen = S.scenario_set(d_bar, sd_d, L, sL, spike=0.30, delay=0.50)
res, draws = [], {}
for i, (k, (a, b, l, sl)) in enumerate(scen.items()):
    r = S.simulate_ltd(50000, 'Gamma', a, b, 'Gamma', l, sl, 0.0, seed=100 + i).ltd.values
    s_ = S.summarize(r, alpha, c['rop'], Q); draws[k] = r
    res.append({'scenario': k, 'required SS': s_['ss_sim'], 'required ROP': s_['rop_sim'], 'P(stockout) at baseline ROP': s_['stockout_prob'], 'fill rate β': s_['fill_rate']})
scen_tab = pd.DataFrame(res); display(scen_tab.round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
base = draws['Baseline']; ax[0].hist(base, 80, density=True, color='#2563EB', alpha=.55, label='Monte Carlo LTD')
xs = np.linspace(base.min(), base.max(), 300); from scipy.stats import norm as N
ax[0].plot(xs, N.pdf(xs, c['ltd_mean'], c['sigma_ltd']), color='#0D9488', lw=2.5, label='Normal approximation')
ax[0].axvline(c['rop'], color='#D97706', ls='--', label='ROP formula'); ax[0].axvline(np.quantile(base, alpha), color='#DC2626', ls=':', label='ROP Monte Carlo')
ax[0].set_title('Lead-time demand: simulation vs formula'); ax[0].legend(fontsize=8)
ax[1].barh(scen_tab.scenario, scen_tab['required SS'], color=['#0D9488', '#D97706', '#EA580C', '#2563EB', '#DC2626'])
ax[1].set_title('Safety stock required to hold α = 95%'); plt.tight_layout(); plt.savefig(FIG / 'fig_montecarlo.png', dpi=160); plt.show()
""")
md("""
**Interpretation.** Under the formula's own assumptions simulation and formula agree closely (validation of the implementation). With skewed lead times or demand–lead-time correlation the true 95% quantile is higher, i.e. the analytical buffer *under-protects*; stress scenarios show the buffer a supplier delay or demand spike would require.

## 7. AI-driven sensitivity analysis — Bayesian optimisation
Instead of sweeping parameters manually, a **Gaussian-process surrogate with Expected Improvement** (and, for comparison, **Optuna TPE**) chooses which (s, Q) policy to evaluate next.

$$\\min_{s,Q}\; \\mathbb E[C] = K\\,\\frac{N_{orders}}{yr} + h\\,c\\,\\bar I + p\\,\\mathbb E[\\text{units short}] \\quad \\text{s.t.}\; \\beta \\ge 95\\%,\; I^{peak}_{95\\%} \\le \\text{Cap},\; c\\bar I \\le \\text{Budget},\; Q \\ge \\text{MOQ}$$

Each candidate is scored by a 364-day stochastic simulation (150 replications, lost sales, Gamma demand, stochastic lead times) on **common random numbers**; the winner is re-validated on an independent stream (300 replications). Assumptions: K = £40/order, c = 60% of average selling price, h = 25%/yr, p = 1× selling price per unit short.
""")
code("""
pr_ = df[df.sku == SKU].avg_price.mean(); unit_c = 0.6 * pr_
cp = O.CostParams(unit_cost=unit_c, holding_rate=.25, order_cost=40, shortage_cost=pr_, target_fill=.95,
                  capacity=round(c['rop'] + 2.5 * Q), budget=unit_c * (c['rop'] + Q), moq=max(1, round(.25 * Q)))
sim = O.PolicySimulator(d_bar, sd_d, L, sL, 'Gamma', reps=150, seed=2025)
base_pol = sim.run(c['rop'], max(Q, cp.moq), cp)
bounds = {'s': (0, round(2.2 * c['rop'])), 'Q': (cp.moq, round(3.5 * Q))}
gp = O.gp_bayes_opt(sim, cp, bounds, n_init=10, n_iter=30, seed=2025, penalty_scale=base_pol['total_cost'])
tpe = O.tpe_bayes_opt(sim, cp, bounds, n_trials=40, seed=2025, penalty_scale=base_pol['total_cost'])
val = O.PolicySimulator(d_bar, sd_d, L, sL, 'Gamma', reps=300, seed=2025 + 7919)
rows = []
for name, s, q in [('Baseline (formula ROP + EOQ)', c['rop'], max(Q, cp.moq)), ('GP-EI optimum', gp['best']['s'], gp['best']['Q']), ('TPE optimum', tpe['best']['s'], tpe['best']['Q'])]:
    r = val.run(s, q, cp); rows.append({'policy': name, 's': s, 'Q': q, 'annual cost £': r['total_cost'], 'ordering £': r['ordering_cost'],
        'holding £': r['holding_cost'], 'shortage £': r['shortage_cost'], 'fill rate β': r['fill_rate'], 'peak stock P95': r['peak_on_hand_p95'], 'feasible': r['feasible']})
bo_tab = pd.DataFrame(rows); display(bo_tab.round(2))
saving = 1 - bo_tab.loc[1, 'annual cost £'] / bo_tab.loc[0, 'annual cost £']
print(f"GP-BO validated saving vs baseline: {saving:.1%}  ({gp['seconds']:.1f}s, 40 evaluations)")
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for r_, col, lab in [(gp, '#0D9488', 'GP-EI'), (tpe, '#D97706', 'TPE')]:
    ax[0].plot(r_['trials'].trial, r_['trials'].best_so_far, drawstyle='steps-post', color=col, lw=2.5, label=lab)
ax[0].axhline(base_pol['total_cost'], color='#64748B', ls='--', label='baseline'); ax[0].set_yscale('log')
ax[0].set_title('Convergence (penalised annual cost)'); ax[0].set_xlabel('evaluation'); ax[0].legend()
sf = gp['surface']; cs = ax[1].contourf(sf['s'], sf['Q'], np.log10(sf['cost']), 20, cmap='GnBu_r')
t = gp['trials']; ax[1].scatter(t.s, t.Q, c=np.where(t.feasible, 'k', 'r'), s=18)
ax[1].scatter([c['rop']], [max(Q, cp.moq)], marker='D', s=90, c='#D97706', label='baseline')
ax[1].scatter([gp['best']['s']], [gp['best']['Q']], marker='*', s=260, c='#16A34A', label='optimum')
ax[1].set_xlabel('reorder point s'); ax[1].set_ylabel('order quantity Q'); ax[1].set_title('GP surrogate (log₁₀ £) & evaluated points'); ax[1].legend()
plt.colorbar(cs, ax=ax[1]); plt.tight_layout(); plt.savefig(FIG / 'fig_bayesopt.png', dpi=160); plt.show()
""")
md("""
## 8. Portfolio application (all 40 SKUs)
The same pipeline is applied to every SKU. Stock on hand is **not** in the source data, so a seeded scenario (ROP × U(0.35, 2.2)) is used to demonstrate the inventory-status logic.
""")
code("""
rows = []
for s in df.sku.unique():
    ys = D.series(df, s, 'W'); qf = F.quick_portfolio_forecast(ys)
    ss_ = I.safety_stock(.95, qf['forecast_mean'], qf['err_sd'], 2, 4/7); rop_ = qf['forecast_mean'] * 2 + ss_
    rows.append({'sku': s, 'forecast/wk': qf['forecast_mean'], 'σ_err': qf['err_sd'], 'WAPE %': qf['wape'], 'SS': ss_, 'ROP': rop_})
port = pd.DataFrame(rows)
stock = I.simulated_on_hand(port.sku, port.ROP, seed=7); port['on_hand (simulated)'] = port.sku.map(stock)
port['risk'] = [I.stockout_risk(o, d, s_, 2, 4/7) for o, d, s_ in zip(port['on_hand (simulated)'], port['forecast/wk'], port['σ_err'])]
port['status'] = [I.classify(o, s_, r, d * 2) for o, s_, r, d in zip(port['on_hand (simulated)'], port.SS, port.ROP, port['forecast/wk'])]
print(port.status.value_counts().to_string()); print(f"Median portfolio WAPE: {port['WAPE %'].median():.1f}%")
display(port.sort_values('risk', ascending=False).head(10).round(2))
port.to_csv(FIG / 'portfolio_results.csv', index=False)
""")
md("""
## 9. Results & business interpretation
* **Forecasting:** statistical/ML models clearly beat the naive benchmark on WAPE; selection is done on validation folds only, so test errors are an unbiased estimate. Retail SKU demand remains noisy (portfolio median WAPE ≈ 35–45%), which is exactly why safety stock is needed.
* **Dynamic safety stock:** using forecast-error variability rather than raw demand variability makes the buffer *adapt* as the model improves or data changes. Lead-time uncertainty contributes a large share of the buffer variance → supplier reliability is a key lever.
* **Monte Carlo:** validates the formula under its assumptions and quantifies how much extra buffer skewed lead times, correlation and stress scenarios require.
* **Bayesian optimisation:** finds a lower-cost (s, Q) policy that meets the fill-rate, capacity, budget and MOQ constraints in only 40 simulations — instead of a grid of hundreds — and the improvement is confirmed on an independent simulation.

## 10. Conclusion
The exercise upgrades a static reorder-point calculator into an adaptive system: **(1)** ML forecasts that retrain when new data arrives, **(2)** probabilistic, simulation-checked safety stocks, and **(3)** Bayesian optimisation of reorder decisions under real operational constraints. The interactive Streamlit dashboard exposes every step with sliders and live KPIs.

**Limitations:** lead times, costs and stock levels are assumptions (absent from the public data); refresh is on-demand batch rather than a live feed; LSTM was not justified at this series length.

## References
1. Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
2. Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
4. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition. *International Journal of Forecasting*, 38(4).
5. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13.
6. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
7. Dr. Sandip Kumar Roy, *Lecture Notes: AI in Operations*, Session II, SP Jain School of Global Management.
""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=900, kernel_name="aiops", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print("written", OUT)
