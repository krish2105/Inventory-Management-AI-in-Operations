"""One-page A4 summary PDF built from the executed notebook's results and figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions" / "Laboratory Work 1" / "Krishna_Mathur_AS25DXB018_AI_in_Operations_Laboratory_Work_1_Summary.pdf"
NAVY, TEAL, MUTED, AMBER = "#0B1E3A", "#0D9488", "#475569", "#D97706"
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(8.27, 11.69))
def txt(x, y, s, **k): fig.text(x, y, s, va="top", **k)

fig.patches.append(plt.Rectangle((0, 0.905), 1, 0.095, transform=fig.transFigure, color=NAVY, zorder=-1))
txt(0.04, 0.988, "AI IN OPERATIONS  ·  SESSION II LABORATORY EXERCISE", color="#7EF0DD", fontsize=7.5, weight="bold")
txt(0.04, 0.972, "AI-Powered Inventory Intelligence & Reorder Point Optimization", color="white", fontsize=14, weight="bold")
txt(0.04, 0.945, "Krishna Mathur · AS25DXB018 · MAIB September 2025 · SP Jain School of Global Management · Instructor: Dr. Sandip Kumar Roy",
    color="#C9D6EA", fontsize=7.2)
txt(0.04, 0.927, "Data: UCI Online Retail II (Chen 2019, CC BY 4.0) — 1,067,371 real invoice lines, Dec 2009–Dec 2011 → 40 SKUs, daily demand. Individual work; no data shared with peers.",
    color="#C9D6EA", fontsize=6.5)

def section(y, title):
    txt(0.04, y, title, fontsize=9.5, weight="bold", color=TEAL)
body = dict(fontsize=7.1, color="#0F172A", linespacing=1.45)

section(0.893, "Objective & method")
txt(0.04, 0.876, "Upgrade a static reorder-point calculator with (1) ML demand forecasting, (2) probabilistic dynamic safety stock with Monte Carlo\n"
    "simulation and (3) Bayesian optimisation of reorder decisions under constraints. One shared Python engine powers both the executed\n"
    "notebook and an interactive 9-tab Streamlit dashboard (sliders, live KPIs, CSV upload & re-training, light/dark mode). 14 unit tests pass.", **body)

section(0.828, "1 · Demand forecasting (focus SKU 85099B Jumbo Bag Red Retrospot, 105 weeks)")
txt(0.04, 0.811, "7 models: Naive, Seasonal naive, Moving avg, ETS, ARIMA, Gradient boosting (lags, Fourier seasonality, Q4 flag, price), MLP neural net.\n"
    "Chronological split: 3 rolling-origin validation folds for selection + untouched 8-week test. Best on validation: ARIMA (WAPE 31.9% vs\n"
    "naive 43.4%); test WAPE 30.9%. New data is validated, reconciled and triggers re-training (demo: model switched to ETS after an update).\n"
    "LSTM assessed but not justified for ~105 weekly points; MLP is the neural benchmark. No promotion flag exists → price used as proxy.", **body)
ax = fig.add_axes([0.03, 0.585, 0.46, 0.17]); ax.imshow(mpimg.imread(ROOT / "outputs/fig_forecast.png")); ax.axis("off")
ax = fig.add_axes([0.51, 0.585, 0.46, 0.17]); ax.imshow(mpimg.imread(ROOT / "outputs/fig_montecarlo.png")); ax.axis("off")

section(0.575, "2 · Dynamic safety stock & Monte Carlo (α = 95%, L = 14 ± 4 days — configured, not in data)")
txt(0.04, 0.558, "SS = Z·√(σ²_D·L + D̄²·σ²_L) with D̄ = AI forecast (864 u/wk) and σ_D = forecast-error SD (381) → SS = 1,202 u, ROP = 2,930 u;\n"
    "lead-time uncertainty drives 46% of buffer variance. 100,000-draw simulation reproduces the formula under its assumptions (ROP 2,972),\n"
    "but Gamma demand gives 3,047 and correlated lead time/demand (ρ = 0.5) 3,326 → the formula under-protects (stockout 9.2% vs 5%).", **body)
rows = [["Scenario", "Req. SS", "P(stockout)*", "Fill rate β"],
        ["Baseline", "1,301", "6.1%", "99.2%"], ["Demand spike +30%", "1,706", "20.9%", "95.8%"],
        ["Supplier delay +50%", "2,174", "33.7%", "90.5%"], ["High variability σ×2", "2,821", "17.2%", "93.5%"],
        ["Combined stress", "3,383", "53.6%", "73.0%"]]
ax = fig.add_axes([0.04, 0.395, 0.44, 0.105]); ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.42, 0.18, 0.22, 0.18]); t.auto_set_font_size(False); t.set_fontsize(6.3); t.scale(1, 1.18)
ax.text(0, -0.08, "*at baseline ROP 3,026 u (50,000 draws each)", fontsize=5.5, color=MUTED, transform=ax.transAxes)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#E2E8F0")
    if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
ax = fig.add_axes([0.52, 0.375, 0.44, 0.15]); ax.imshow(mpimg.imread(ROOT / "outputs/fig_bayesopt.png")); ax.axis("off")

section(0.365, "3 · Bayesian optimisation of the (s, Q) policy (replaces manual sweeps)")
txt(0.04, 0.348, "Minimise annual ordering + holding + shortage cost s.t. fill rate ≥ 95%, peak stock ≤ capacity, avg value ≤ budget, Q ≥ MOQ.\n"
    "Each candidate = 364-day stochastic simulation (150 reps, common random numbers); winner re-validated on an independent stream.", **body)
rows = [["Policy", "s", "Q", "Annual cost", "Shortage cost", "Fill rate", "Feasible"],
        ["Baseline: formula ROP + EOQ", "2,930", "3,441", "£2,149", "£741", "99.0%", "yes"],
        ["GP-EI Bayesian optimum", "3,740", "3,791", "£1,756", "£92", "99.8%", "yes"],
        ["Optuna TPE optimum", "3,472", "3,104", "£1,824", "£246", "99.6%", "yes"]]
ax = fig.add_axes([0.04, 0.245, 0.92, 0.075]); ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.3, 0.09, 0.09, 0.13, 0.14, 0.12, 0.11]); t.auto_set_font_size(False); t.set_fontsize(6.8); t.scale(1, 1.25)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#E2E8F0")
    if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    if r == 2: cell.set_facecolor("#CCFBF1")
txt(0.04, 0.236, "→ GP-EI cuts validated annual cost by 18.3% (40 evaluations, 1.2 s) mainly by trading shortage cost (−£648) for safety stock.", fontsize=7.3, weight="bold", color=NAVY)

section(0.212, "Portfolio results (40 SKUs) & recommendations")
txt(0.04, 0.195, "Median forecast WAPE 44.7% — retail demand is noisy, so buffers must be sized on forecast error, not averages. With a seeded stock\n"
    "scenario: 18 Healthy, 6 Monitor, 11 Reorder Now, 5 Critical.  Recommendations: (1) replace static ROPs with forecast-error-based dynamic\n"
    "SS refreshed on every data load; (2) size buffers for high-risk SKUs by simulation when lead times are skewed or demand-correlated;\n"
    "(3) negotiate lead-time reliability — it drives ~half the buffer; (4) set (s, Q) jointly by Bayesian optimisation against true costs\n"
    "and constraints instead of a fixed service target; (5) feed live stock and supplier lead-time data to remove the scenario assumptions.", **body)
section(0.112, "Conclusion")
txt(0.04, 0.095, "All three techniques in the brief are implemented on real data and work together: forecasts update with new data, safety stock adapts\n"
    "to forecast error and is stress-tested by Monte Carlo, and Bayesian optimisation finds a cheaper constraint-feasible policy. Limitations:\n"
    "lead time, costs and stock are configured/simulated (absent from public data); refresh is on-demand batch, not real-time.", **body)
fig.lines.append(plt.Line2D([0.04, 0.96], [0.035, 0.035], transform=fig.transFigure, color="#E2E8F0"))
txt(0.04, 0.03, "Deliverables: executed notebook · Streamlit dashboard (apps/streamlit/app.py) · engine src/inventory_core · tests · github.com/krish2105/Inventory-Management-AI-in-Operations",
    fontsize=6, color=MUTED)
fig.savefig(OUT, format="pdf")
print("written", OUT)
