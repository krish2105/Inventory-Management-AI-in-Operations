# AI-Powered Inventory Intelligence & Reorder Point Optimization

**AI in Operations — Session II: AI Applications in Reorder Point & Inventory Management (Laboratory Exercise)**
Done by **Krishna Mathur · AS25DXB018 · MAIB September 2025** · SP Jain School of Global Management · Instructor: Dr. Sandip Kumar Roy

An interactive Streamlit dashboard and executed notebook that turn a static, pre-AI reorder-point calculator into an **autonomous AI system**, built on **real retail data**. It covers the three workstreams of the brief and adds a measured **Before AI vs After AI** comparison and an **AI Autopilot**:

| Workstream (from the brief) | Implementation | Dashboard tab |
|---|---|---|
| 1. Demand forecasting (ML, seasonality, promotions/external factors, add new data, continuous updates) | Naive, Seasonal naive, Moving average, ETS, ARIMA/SARIMA, Gradient-boosting regression, MLP neural net · rolling-origin backtesting · CSV upload → validate → reconcile → retrain | Demand Forecasting, Data Manager |
| 2. Dynamic safety stock (probabilistic, Monte Carlo, risk scenarios) | `SS = Z·√(σ²_D·L + D̄²·σ²_L)`, `ROP = D̄·L + SS` with σ_D from live forecast error · 10k–100k-draw simulation, 5 demand × 5 lead-time distributions, copula dependence, 5 stress scenarios | Safety Stock & ROP, Monte Carlo Risk |
| 3. AI-driven sensitivity analysis (Bayesian optimisation under constraints) | Gaussian-process BO with Expected Improvement + Optuna TPE over (s, Q); cost = ordering + holding + shortage; constraints: fill rate, capacity, budget, MOQ; common random numbers + independent validation | Bayesian Optimization |

## Dashboard (6 tabs, light / dark mode)
| Tab | What it shows |
|---|---|
| **Before vs After AI** | Day-by-day replay of the last 26 weeks of real demand (40 SKUs). Classic static textbook policy vs the AI policy (weekly self-retraining forecasts, drift-triggered model switching, bias self-correction, conformal buffers, cost-aware service levels). Result: −23% lost sales, −38% stockout days, fill rate 98.7% → 99.0%, WAPE 52% → 41%, −2.4% cost, but +12% average stock |
| **AI Autopilot** | Runs on every data load: 8-step pipeline, replenishment agent (expedite / order / watch / hold with reasons and draft POs), Isolation Forest anomaly alerts, ABC-XYZ + K-means segmentation with auto service levels, auto exception digest |
| **1 · Demand Forecasting** | 7 models, rolling-origin backtest, prediction intervals, STL decomposition, feature importance |
| **2 · Safety Stock & Monte Carlo** | Assignment formula calculator, sensitivity, analytical vs simulation, 100k-draw Monte Carlo, stress scenarios |
| **3 · Bayesian Optimization** | GP-EI / TPE over (s, Q) under constraints, compared with a manual grid sweep (before AI), plus one-click auto-tuning of all SKUs |
| **Data & Reports** | CSV upload → validate → reconcile → retrain; Excel/CSV exports; assignment-to-implementation map |

Every KPI is computed live from the sidebar sliders.

## Data
Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D — CC BY 4.0.
1,067,371 invoice lines (UK online gift wholesaler, Dec 2009 – Dec 2011) → cleaned and aggregated by `scripts/prepare_data.py` into `data/processed/sku_daily_demand.csv` (40 SKUs × 604 trading days).
**Provenance:** demand & price = observed · forecasts, SS, ROP = derived · service level, lead time, costs = configured · stock on hand = *simulated scenario* (the source has no inventory, lead-time or promotion fields; nothing is fabricated as observed).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run apps/streamlit/app.py
```
Tests: `pip install -r requirements-dev.txt && pytest -q` (18 tests: formula, ROP, Monte Carlo, distributions, metrics, ingestion, backtesting, BO constraints, segmentation, anomaly detection, drift signal, replay, agent).
Rebuild data from raw: download the UCI zip into `data/raw/`, then `python scripts/prepare_data.py`. Re-run the walk-forward AI forecasts with `python scripts/precompute_ai.py` (about 2.5 min). The dashboard recomputes them automatically after a data upload.

## Structure
```
apps/streamlit/app.py, ui.py     Streamlit dashboard + theme
src/inventory_core/              Shared engine: data, forecasting, inventory, simulation, optimization, automation (AI layer)
data/processed/                  Cleaned real dataset used by app & notebook
submissions/Laboratory Work 1/   Executed notebook, one-page PDF summary, Colab link file
scripts/                         Data preparation, notebook & PDF builders
tests/                           pytest suite
outputs/                         Figures & portfolio results produced by the notebook
```

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) (Streamlit Community Cloud, step by step).

## Privacy
Individual work — "No sharing of data with peers allowed". Single-user sessions; uploaded files are held only in the user's session memory, never written to disk or shared; Streamlit usage statistics disabled.
