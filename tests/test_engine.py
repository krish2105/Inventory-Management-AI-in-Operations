"""Unit tests for the shared inventory engine."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inventory_core import data as D, forecasting as F, inventory as I, simulation as S, optimization as O


def test_safety_stock_formula_matches_hand_calculation():
    # Z(0.95)=1.6449 ; sqrt(100^2*2 + 500^2*0.5^2) = sqrt(20000+62500) = 287.228
    ss = I.safety_stock(0.95, d_bar=500, sd_d=100, L=2, sd_l=0.5)
    assert ss == pytest.approx(norm.ppf(0.95) * np.sqrt(20000 + 62500), rel=1e-9)
    assert I.reorder_point(0.95, 500, 100, 2, 0.5) == pytest.approx(1000 + ss)


def test_zero_lead_time_variability_reduces_to_classic_formula():
    assert I.safety_stock(0.9, 50, 10, 4, 0) == pytest.approx(norm.ppf(0.9) * 10 * 2)


def test_eoq_and_stockout_risk():
    assert I.eoq(52000, 50, 1.0) == pytest.approx(np.sqrt(2 * 52000 * 50 / 1.0))
    rop = I.reorder_point(0.95, 500, 100, 2, 0.5)
    assert I.stockout_risk(rop, 500, 100, 2, 0.5) == pytest.approx(0.05, abs=1e-9)


def test_classification_ordering():
    assert I.classify(100, 300, 1300, 1000) == "Critical Stockout Risk"
    assert I.classify(1200, 300, 1300, 1000) == "Reorder Now"
    assert I.classify(5000, 300, 1300, 1000) == "Healthy"


def test_monte_carlo_matches_formula_under_normal_assumptions_and_is_reproducible():
    a = S.simulate_ltd(100000, "Normal (truncated at 0)", 500, 100, "Normal (truncated)", 2, 0.3, seed=1)
    b = S.simulate_ltd(100000, "Normal (truncated at 0)", 500, 100, "Normal (truncated)", 2, 0.3, seed=1)
    assert np.array_equal(a["ltd"].values, b["ltd"].values)
    assert a["ltd"].mean() == pytest.approx(1000, rel=0.01)
    assert np.quantile(a["ltd"], 0.95) == pytest.approx(I.reorder_point(0.95, 500, 100, 2, 0.3), rel=0.03)


def test_positive_correlation_increases_tail_risk():
    ind = S.simulate_ltd(50000, "Gamma", 500, 150, "Gamma", 2, 0.5, rho=0.0, seed=3)["ltd"]
    cor = S.simulate_ltd(50000, "Gamma", 500, 150, "Gamma", 2, 0.5, rho=0.7, seed=3)["ltd"]
    assert np.quantile(cor, 0.95) > np.quantile(ind, 0.95)


def test_lead_time_distributions_match_moments():
    for d in S.LEAD_DISTS:
        ld = S.lead_time_dist(d, 2.0, 0.4)
        assert ld.mean() == pytest.approx(2.0, rel=0.05), d
        assert ld.std() == pytest.approx(0.4, rel=0.15), d


def test_metrics():
    m = F.metrics([10, 0, 20], [12, 1, 18])
    assert m["MAE"] == pytest.approx(5 / 3)
    assert m["WAPE %"] == pytest.approx(100 * 5 / 30)
    assert m["MAPE %"] == pytest.approx(100 * np.mean([0.2, 0.1]))  # zero actual excluded


def test_validation_and_merge():
    raw = pd.DataFrame({"InvoiceDate": ["2011-12-12", "2011-12-12", "bad", "2011-12-13"],
                        "StockCode": ["A", "A", "A", "B"], "Quantity": [5, 3, 2, -1]})
    mp = D.detect_columns(raw.columns)
    assert mp == {"date": "InvoiceDate", "sku": "StockCode", "demand": "Quantity"}
    clean, rep = D.validate(raw, mp)
    assert rep.ok and rep.bad_dates == 1 and rep.negative == 1 and rep.duplicates == 1
    assert clean["demand"].tolist() == [8]
    base = pd.DataFrame({"date": pd.to_datetime(["2011-12-12"]), "sku": ["A"], "demand": [1.0]})
    merged, info = D.merge_new(base, clean, "replace")
    assert merged["demand"].tolist() == [8] and info["reconciled_rows"] == 1
    merged, _ = D.merge_new(base, clean, "add")
    assert merged["demand"].tolist() == [9]


def test_missing_required_column_rejected():
    _, rep = D.validate(pd.DataFrame({"x": [1]}), {})
    assert not rep.ok and "Missing required" in rep.messages[0]


@pytest.fixture(scope="module")
def real():
    df = D.load_default()
    sku = df.groupby("sku")["demand"].sum().idxmax()
    return df, sku, D.series(df, sku, "W")


def test_real_dataset_loaded(real):
    df, _, y = real
    assert df["sku"].nunique() == 40 and len(y) > 100 and (df["demand"] >= 0).all()


def test_backtest_selects_on_validation_and_beats_naive(real):
    df, sku, y = real
    bt = F.backtest(y, "W", 8, D.price_series(df, sku, "W", y.index),
                    models=["Naive", "Moving Average", "ETS (Holt-Winters)"])
    assert bt["best"] == bt["validation"]["WAPE %"].idxmin()
    assert bt["validation"].loc[bt["best"], "WAPE %"] <= bt["validation"].loc["Naive", "WAPE %"]
    fc = F.forecast(y, "W", 8, bt["best"], bt["validation"].loc[bt["best"], "resid_sd"])
    assert len(fc) == 8 and (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()


def test_bayesian_optimisation_respects_constraints_and_beats_baseline():
    sim = O.PolicySimulator(1000, 300, 2, 0.5, reps=80, seed=5)
    cp = O.CostParams(unit_cost=1.0, order_cost=50, shortage_cost=2.0, target_fill=0.97, moq=200)
    rop = I.reorder_point(0.95, 1000, 300, 2, 0.5)
    base = sim.run(rop, max(I.eoq(52000, 50, 0.25), 200), cp)
    for fn in (O.gp_bayes_opt, O.tpe_bayes_opt):
        kw = dict(n_init=8, n_iter=17) if fn is O.gp_bayes_opt else dict(n_trials=25)
        res = fn(sim, cp, {"s": (0, 3 * rop), "Q": (200, 8000)}, seed=1, penalty_scale=base["total_cost"], **kw)
        b = res["best"]
        assert b["feasible"] and b["fill_rate"] >= 0.97 and b["Q"] >= 200
        assert b["total_cost"] <= base["total_cost"] * 1.02


def test_common_random_numbers_are_deterministic():
    sim = O.PolicySimulator(500, 100, 1.5, 0.3, reps=40, seed=9)
    cp = O.CostParams()
    assert sim.run(800, 1000, cp)["total_cost"] == sim.run(800, 1000, cp)["total_cost"]
