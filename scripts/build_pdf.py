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
txt(0.04, 0.876, "Turn a static, pre-AI reorder-point calculator into an autonomous AI system: (1) self-retraining ML forecasts, (2) probabilistic dynamic\n"
    "safety stock with Monte Carlo, (3) Bayesian optimisation under constraints, plus drift/anomaly detection, ML segmentation and a replenishment\n"
    "agent. One Python engine powers the executed notebook and a 6-tab Streamlit dashboard (Before vs After AI, AI Autopilot, 3 workstreams). 18 tests pass.", **body)

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
ax = fig.add_axes([0.04, 0.405, 0.44, 0.1]); ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.42, 0.18, 0.22, 0.18]); t.auto_set_font_size(False); t.set_fontsize(6.3); t.scale(1, 1.18)
ax.text(0, -0.1, "*at baseline ROP 3,026 u (50,000 draws each)", fontsize=5.5, color=MUTED, transform=ax.transAxes)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#E2E8F0")
    if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
ax = fig.add_axes([0.52, 0.392, 0.44, 0.135]); ax.imshow(mpimg.imread(ROOT / "outputs/fig_bayesopt.png")); ax.axis("off")

section(0.383, "3 · Bayesian optimisation of the (s, Q) policy (replaces manual sweeps)")
txt(0.04, 0.367, "Minimise annual ordering + holding + shortage cost s.t. fill rate ≥ 95%, peak stock ≤ capacity, avg value ≤ budget, Q ≥ MOQ.\n"
    "Each candidate = 364-day stochastic simulation (150 reps, common random numbers); winner re-validated on an independent stream.", **body)
rows = [["Policy", "s", "Q", "Annual cost", "Shortage cost", "Fill rate", "Feasible"],
        ["Baseline: formula ROP + EOQ", "2,930", "3,441", "£2,149", "£741", "99.0%", "yes"],
        ["GP-EI Bayesian optimum", "3,740", "3,791", "£1,756", "£92", "99.8%", "yes"],
        ["Optuna TPE optimum", "3,472", "3,104", "£1,824", "£246", "99.6%", "yes"]]
ax = fig.add_axes([0.04, 0.268, 0.92, 0.07]); ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.3, 0.09, 0.09, 0.13, 0.14, 0.12, 0.11]); t.auto_set_font_size(False); t.set_fontsize(6.8); t.scale(1, 1.25)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#E2E8F0")
    if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    if r == 2: cell.set_facecolor("#CCFBF1")
txt(0.04, 0.262, "→ GP-EI cuts validated annual cost by 18.3% (40 evaluations, 1.2 s) mainly by trading shortage cost (−£648) for safety stock.", fontsize=7.3, weight="bold", color=NAVY)

section(0.243, "Before AI vs After AI — replay on the last 26 weeks of real demand, 40 SKUs, identical suppliers & costs")
rows = [["Metric", "Before AI", "After AI", "Change"],
        ["Fill rate", "98.74%", "99.02%", "+0.28 pp"], ["Lost sales (units)", "4,567", "3,526", "−22.8%"],
        ["Stockout days", "53", "33", "−37.7%"], ["Forecast WAPE", "52.0%", "41.5%", "−20.2%"],
        ["Total cost (26 wks)", "£25,386", "£24,775", "−2.4%"], ["Avg inventory value", "£75,953", "£84,895", "+11.8%"]]
ax = fig.add_axes([0.04, 0.132, 0.5, 0.09]); ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.3, 0.28, 0.26, 0.16]); t.auto_set_font_size(False); t.set_fontsize(6.2); t.scale(1, 1.05)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#E2E8F0")
    if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
txt(0.565, 0.226, "What the AI does that the old method could not:\n"
    "• refits forecasts weekly; 92 drift alarms → 295 auto-retrains,\n   52 champion/challenger model switches\n"
    "• corrects its own bias; conformal buffers from its own errors\n"
    "• cost-optimal service level per SKU, floored by ML segment\n   (ABC-XYZ + K-means)\n"
    "• Isolation Forest flags demand spikes/drops\n"
    "• agent drafts POs with reasons; writes a daily digest\n"
    "Honest note: AI holds more stock where shortages are costly;\n   it is cheaper on 19/40 SKUs; Bayesian (s,Q) adds −18%.", fontsize=6.3, color="#0F172A", linespacing=1.35)
section(0.112, "Conclusion")
txt(0.04, 0.095, "All three techniques in the brief run on real data and act autonomously: forecasts retrain on new data and drift, buffers adapt and are\n"
    "stress-tested by Monte Carlo, and Bayesian optimisation finds cheaper constraint-feasible policies. Against the pre-AI method the system\n"
    "loses 23% fewer sales. Limitations: lead time, costs and stock are configured/simulated; refresh is on-demand batch, not real-time.", **body)
fig.lines.append(plt.Line2D([0.04, 0.96], [0.035, 0.035], transform=fig.transFigure, color="#E2E8F0"))
txt(0.04, 0.03, "Deliverables: executed notebook · Streamlit dashboard (apps/streamlit/app.py) · engine src/inventory_core · tests · github.com/krish2105/Inventory-Management-AI-in-Operations",
    fontsize=6, color=MUTED)
fig.savefig(OUT, format="pdf")
print("written", OUT)
