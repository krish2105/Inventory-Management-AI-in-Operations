"""Demand forecasting: statistical baselines, ETS, ARIMA, gradient boosting and neural-network
regression, evaluated with chronological rolling-origin backtesting."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import SEASON

MODEL_INFO = {
    "Naive": "Last observed value carried forward — the minimum benchmark.",
    "Seasonal Naive": "Value from the same period one season earlier (7 days / 52 weeks / 12 months).",
    "Moving Average": "Mean of the most recent window of periods.",
    "ETS (Holt-Winters)": "Exponential smoothing with damped additive trend (+ additive seasonality when ≥2 seasons of history).",
    "ARIMA / SARIMA": "Auto-regressive integrated moving average; seasonal terms for daily data.",
    "Gradient Boosting (ML)": "Gradient-boosted trees on lags, rolling means, calendar seasonality, Q4 peak flag and observed price. Recursive multi-step.",
    "Neural Net (MLP)": "Feed-forward neural network on the same engineered features (lightweight deep-learning model).",
}
MODELS = list(MODEL_INFO)
MA_WINDOW = {"D": 7, "W": 4, "M": 3}
LAGS = {"D": [1, 2, 3, 7, 14], "W": [1, 2, 3, 4, 8], "M": [1, 2, 3]}


# ---------------------------------------------------------------- metrics
def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = a - p
    nz = a != 0
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "WAPE %": float(100 * np.sum(np.abs(e)) / max(np.sum(np.abs(a)), 1e-9)),
        # MAPE is undefined when actual = 0, so zero-demand periods are excluded
        "MAPE %": float(100 * np.mean(np.abs(e[nz] / a[nz]))) if nz.any() else np.nan,
        "Bias": float(np.mean(p - a)),
    }


# ---------------------------------------------------------------- features
def _features(hist: np.ndarray, idx: pd.Timestamp, freq: str, price: float) -> list:
    lags = LAGS[freq]
    f = [hist[-l] if len(hist) >= l else hist[0] for l in lags]
    w = MA_WINDOW[freq]
    f.append(np.mean(hist[-w:]))
    f.append(np.std(hist[-w:]))
    if freq == "D":
        pos, per = idx.day_of_week, 7
        f += [float(idx.day_of_week == d) for d in range(7)]
    elif freq == "W":
        pos, per = idx.isocalendar().week, 52
    else:
        pos, per = idx.month, 12
    f += [np.sin(2 * np.pi * pos / per), np.cos(2 * np.pi * pos / per)]
    yr = idx.day_of_year / 365.25
    f += [np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    f.append(float(idx.month in (9, 10, 11)))  # derived: pre-Christmas wholesale peak
    f.append(price if np.isfinite(price) else 0.0)
    return f


FEATURE_NAMES = {
    fr: [f"lag_{l}" for l in LAGS[fr]] + ["roll_mean", "roll_std"]
    + ([f"dow_{d}" for d in range(7)] if fr == "D" else [])
    + ["season_sin", "season_cos", "year_sin", "year_cos", "q4_peak_flag", "avg_price"]
    for fr in ("D", "W", "M")
}


def _train_xy(y: pd.Series, price: pd.Series, freq: str):
    v = y.values
    start = max(LAGS[freq])
    X, t = [], []
    for i in range(start, len(v)):
        X.append(_features(v[:i], y.index[i], freq, price.iloc[i]))
        t.append(v[i])
    return np.array(X), np.array(t)


def _recursive(model, y: pd.Series, price: pd.Series, freq: str, h: int, fut_idx) -> np.ndarray:
    hist = list(y.values)
    last_price = price.iloc[-1] if len(price) else 0.0  # future price assumed = last observed
    out = []
    for k in range(h):
        x = np.array(_features(np.array(hist), fut_idx[k], freq, last_price)).reshape(1, -1)
        p = max(float(model.predict(x)[0]), 0.0)
        out.append(p)
        hist.append(p)
    return np.array(out)


def future_index(y: pd.Series, freq: str, h: int) -> pd.DatetimeIndex:
    off = {"D": pd.offsets.Day(1), "W": pd.offsets.Week(1), "M": pd.offsets.MonthBegin(1)}[freq]
    return pd.date_range(y.index[-1] + off, periods=h, freq=off)


# ---------------------------------------------------------------- models
def fit_predict(name: str, y: pd.Series, h: int, freq: str, price: pd.Series | None = None,
                return_model: bool = False):
    m = SEASON[freq]
    price = price if price is not None else pd.Series(0.0, index=y.index)
    fidx = future_index(y, freq, h)
    model = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "Naive":
            pred = np.repeat(y.iloc[-1], h)
        elif name == "Seasonal Naive":
            if len(y) >= m:
                last = y.values[-m:]
                pred = np.array([last[k % m] for k in range(h)])
            else:
                pred = np.repeat(y.iloc[-1], h)
        elif name == "Moving Average":
            pred = np.repeat(y.iloc[-MA_WINDOW[freq]:].mean(), h)
        elif name == "ETS (Holt-Winters)":
            seasonal = "add" if len(y) >= 2 * m + 2 else None
            model = ExponentialSmoothing(y.values, trend="add", damped_trend=True, seasonal=seasonal,
                                         seasonal_periods=m if seasonal else None,
                                         initialization_method="estimated").fit()
            pred = model.forecast(h)
        elif name == "ARIMA / SARIMA":
            seas = (1, 0, 1, 7) if freq == "D" else (0, 0, 0, 0)
            model = SARIMAX(y.values, order=(1, 1, 1), seasonal_order=seas, trend="n",
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
            pred = model.forecast(h)
        elif name in ("Gradient Boosting (ML)", "Neural Net (MLP)"):
            X, t = _train_xy(y, price, freq)
            if name == "Gradient Boosting (ML)":
                model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                  subsample=0.8, loss="huber", random_state=42)
            else:
                model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                      max_iter=3000, early_stopping=len(t) > 60, random_state=42))
            model.fit(X, t)
            pred = _recursive(model, y, price, freq, h, fidx)
        else:
            raise ValueError(name)
    pred = np.clip(np.asarray(pred, float), 0, None)
    return (pred, model) if return_model else pred


# ---------------------------------------------------------------- evaluation
def backtest(y: pd.Series, freq: str, h: int, price: pd.Series | None = None, folds: int = 3,
             models: list | None = None) -> dict:
    """Chronological evaluation.
    * Test set  = last h periods (never used for model selection).
    * Validation = `folds` rolling origins, each forecasting h periods, ending before the test set.
    Best model = lowest mean validation WAPE."""
    models = models or MODELS
    price = price if price is not None else pd.Series(0.0, index=y.index)
    n = len(y)
    test_start = n - h
    val_rows, val_preds = [], {}
    for mname in models:
        errs, a_all, p_all = [], [], []
        for f in range(folds, 0, -1):
            origin = test_start - f * h
            if origin < max(10, 2 * max(LAGS[freq])):
                continue
            tr, te = y.iloc[:origin], y.iloc[origin:origin + h]
            try:
                p = fit_predict(mname, tr, len(te), freq, price.iloc[:origin])
            except Exception:
                p = np.repeat(tr.iloc[-1], len(te))
            a_all.append(te.values)
            p_all.append(p)
            errs.extend(te.values - p)
        if a_all:
            met = metrics(np.concatenate(a_all), np.concatenate(p_all))
            met["resid_sd"] = float(np.std(errs, ddof=1)) if len(errs) > 1 else np.nan
            val_rows.append({"Model": mname, **met})
    val = pd.DataFrame(val_rows).set_index("Model")
    best = val["WAPE %"].idxmin()

    test_rows, test_preds = [], {}
    tr, te = y.iloc[:test_start], y.iloc[test_start:]
    for mname in models:
        try:
            p = fit_predict(mname, tr, len(te), freq, price.iloc[:test_start])
        except Exception:
            p = np.repeat(tr.iloc[-1], len(te))
        test_preds[mname] = pd.Series(p, index=te.index)
        test_rows.append({"Model": mname, **metrics(te.values, p)})
    test = pd.DataFrame(test_rows).set_index("Model")
    return {"validation": val, "test": test, "best": best, "test_preds": test_preds,
            "test_actual": te, "train_end": tr.index[-1]}


def forecast(y: pd.Series, freq: str, h: int, model_name: str, resid_sd: float,
             price: pd.Series | None = None, level: float = 0.95) -> pd.DataFrame:
    """Refit on all data and forecast h periods with an empirical prediction interval
    (± z · backtest residual SD, widening with sqrt of the step for multi-step uncertainty)."""
    pred, model = fit_predict(model_name, y, h, freq, price, return_model=True)
    z = norm.ppf(0.5 + level / 2)
    steps = np.arange(1, h + 1)
    width = z * resid_sd * np.sqrt(1 + (steps - 1) / max(h, 1))
    idx = future_index(y, freq, h)
    return pd.DataFrame({"forecast": pred, "lower": np.clip(pred - width, 0, None), "upper": pred + width},
                        index=idx)


def feature_importance(y: pd.Series, freq: str, price: pd.Series | None = None) -> pd.Series:
    price = price if price is not None else pd.Series(0.0, index=y.index)
    X, t = _train_xy(y, price, freq)
    m = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  loss="huber", random_state=42).fit(X, t)
    return pd.Series(m.feature_importances_, index=FEATURE_NAMES[freq]).sort_values(ascending=True)


def quick_portfolio_forecast(y: pd.Series, h: int = 4, holdout: int = 8) -> dict:
    """Fast per-SKU forecast for the portfolio view: damped ETS on weekly demand,
    with forecast-error SD measured on an `holdout`-week hold-out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr, te = y.iloc[:-holdout], y.iloc[-holdout:]
        p = fit_predict("ETS (Holt-Winters)", tr, holdout, "W")
        err_sd = float(np.sqrt(np.mean((te.values - p) ** 2)))
        fc = fit_predict("ETS (Holt-Winters)", y, h, "W")
    return {"forecast_mean": float(np.mean(fc)), "forecast_total": float(np.sum(fc)), "err_sd": err_sd,
            "wape": float(100 * np.abs(te.values - p).sum() / max(te.values.sum(), 1e-9))}
