"""Data loading, validation, ingestion of new data and aggregation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "sku_daily_demand.csv"
REQUIRED = ["date", "sku", "demand"]
OPTIONAL = ["description", "avg_price"]

# Column-name synonyms used for automatic column detection on uploaded CSVs
SYNONYMS = {
    "date": ["date", "invoicedate", "day", "ds", "timestamp", "order_date", "orderdate"],
    "sku": ["sku", "stockcode", "item", "item_id", "product", "product_id", "productid", "code"],
    "demand": ["demand", "quantity", "qty", "units", "sales", "units_sold", "y"],
    "description": ["description", "name", "product_name", "item_name"],
    "avg_price": ["avg_price", "price", "unit_price", "unitprice", "sell_price"],
}
FREQ_LABEL = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
SEASON = {"D": 7, "W": 52, "M": 12}


def load_default() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATA, dtype={"sku": str}, parse_dates=["date"])
    return df


def detect_columns(cols) -> dict:
    """Return {canonical: original} mapping guessed from column names."""
    norm = {c: c.strip().lower().replace(" ", "_") for c in cols}
    out = {}
    for canon, syns in SYNONYMS.items():
        for orig, n in norm.items():
            if n in syns and orig not in out.values():
                out[canon] = orig
                break
    return out


@dataclass
class ValidationReport:
    ok: bool
    rows_in: int = 0
    rows_valid: int = 0
    bad_dates: int = 0
    bad_demand: int = 0
    negative: int = 0
    duplicates: int = 0
    skus: int = 0
    date_min: str = ""
    date_max: str = ""
    messages: list = field(default_factory=list)


def validate(raw: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(ok=False, rows_in=len(raw))
    missing = [c for c in REQUIRED if c not in mapping or mapping[c] not in raw.columns]
    if missing:
        rep.messages.append(f"Missing required column(s): {', '.join(missing)}")
        return pd.DataFrame(), rep
    df = pd.DataFrame({c: raw[mapping[c]] for c in REQUIRED + OPTIONAL if c in mapping and mapping[c] in raw.columns})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    rep.bad_dates = int(df["date"].isna().sum())
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    rep.bad_demand = int(df["demand"].isna().sum())
    df["sku"] = df["sku"].astype(str).str.strip()
    df = df.dropna(subset=["date", "demand"])
    rep.negative = int((df["demand"] < 0).sum())
    df = df[df["demand"] >= 0]
    if "avg_price" in df:
        df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    # Aggregate duplicate (date, sku) rows — transactional files have many lines per day
    rep.duplicates = int(df.duplicated(["date", "sku"]).sum())
    agg = {"demand": "sum"}
    if "description" in df:
        agg["description"] = "first"
    if "avg_price" in df:
        agg["avg_price"] = "mean"
    df = df.groupby(["date", "sku"], as_index=False).agg(agg)
    rep.rows_valid = len(df)
    rep.skus = df["sku"].nunique()
    if len(df):
        rep.date_min, rep.date_max = str(df["date"].min().date()), str(df["date"].max().date())
        rep.ok = True
    if rep.bad_dates:
        rep.messages.append(f"{rep.bad_dates} row(s) dropped: unparseable date")
    if rep.bad_demand:
        rep.messages.append(f"{rep.bad_demand} row(s) dropped: non-numeric demand")
    if rep.negative:
        rep.messages.append(f"{rep.negative} row(s) dropped: negative demand (returns)")
    if rep.duplicates:
        rep.messages.append(f"{rep.duplicates} duplicate (date, SKU) row(s) aggregated by summing demand")
    return df, rep


def merge_new(base: pd.DataFrame, new: pd.DataFrame, mode: str = "replace") -> tuple[pd.DataFrame, dict]:
    """Append new observations. On overlap of (date, sku): 'replace' keeps the new value, 'add' sums."""
    base = base.copy()
    key = ["date", "sku"]
    overlap = base.merge(new[key], on=key, how="inner")
    if mode == "add":
        comb = pd.concat([base, new], ignore_index=True)
        agg = {"demand": "sum", "description": "first", "avg_price": "last"}
        comb = comb.groupby(key, as_index=False).agg({k: v for k, v in agg.items() if k in comb})
    else:
        comb = pd.concat([base, new], ignore_index=True).drop_duplicates(key, keep="last")
    # carry forward description / price for SKUs that already exist
    if "description" in comb:
        comb["description"] = comb.groupby("sku")["description"].transform(lambda s: s.ffill().bfill())
    if "avg_price" in comb:
        comb["avg_price"] = comb.groupby("sku")["avg_price"].transform(lambda s: s.ffill().bfill())
    comb = comb.sort_values(key).reset_index(drop=True)
    info = {"added_rows": len(new) - len(overlap), "reconciled_rows": len(overlap),
            "new_skus": int(len(set(new["sku"]) - set(base["sku"])))}
    return comb, info


def series(df: pd.DataFrame, sku: str, freq: str = "W") -> pd.Series:
    """Demand series for one SKU at D/W/M frequency. Missing calendar periods -> 0 demand.
    Partial first/last weeks/months are dropped so they do not bias the model."""
    s = df.loc[df["sku"] == sku].set_index("date")["demand"].sort_index()
    rule = {"D": "D", "W": "W-SUN", "M": "MS"}[freq]
    out = s.resample(rule).sum()
    if freq in ("W", "M") and len(out) > 3:
        first, last = s.index.min(), s.index.max()
        if freq == "W":
            if first.day_of_week != 0:  # week did not start Monday
                out = out.iloc[1:]
            if last.day_of_week != 6 and last.day_of_week < 4:
                out = out.iloc[:-1]
        else:
            if first.day > 3:
                out = out.iloc[1:]
            if last.day < 25:
                out = out.iloc[:-1]
    out.name = sku
    return out.astype(float)


def price_series(df: pd.DataFrame, sku: str, freq: str, index: pd.DatetimeIndex) -> pd.Series:
    if "avg_price" not in df:
        return pd.Series(np.nan, index=index)
    rule = {"D": "D", "W": "W-SUN", "M": "MS"}[freq]
    p = df.loc[df["sku"] == sku].set_index("date")["avg_price"].sort_index().resample(rule).mean()
    return p.reindex(index).ffill().bfill()


def dataset_quality(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sku")
    q = pd.DataFrame({
        "description": g["description"].first() if "description" in df else "",
        "first_date": g["date"].min().dt.date,
        "last_date": g["date"].max().dt.date,
        "observations": g.size(),
        "zero_demand_days_%": g["demand"].apply(lambda s: round(100 * (s == 0).mean(), 1)),
        "total_units": g["demand"].sum().round(0),
        "mean_daily": g["demand"].mean().round(2),
        "cv_daily": (g["demand"].std() / g["demand"].mean()).round(2),
        "avg_price_gbp": g["avg_price"].mean().round(2) if "avg_price" in df else np.nan,
        "missing_price": g["avg_price"].apply(lambda s: int(s.isna().sum())) if "avg_price" in df else np.nan,
    })
    return q.reset_index()
