"""Prepare the UCI Online Retail II dataset for the inventory dashboard.

Source: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D  (CC BY 4.0)

Output: data/processed/sku_daily_demand.csv  — daily demand (units) for the top-N SKUs,
with observed average unit price and derived calendar features.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUT = ROOT / "data" / "processed"
TOP_N = 40

def main():
    sheets = pd.read_excel(RAW, sheet_name=None, dtype={"StockCode": str, "Invoice": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    n_raw = len(df)
    df = df[~df["Invoice"].str.startswith("C", na=False)]          # cancellations
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["StockCode"].str.match(r"^\d{5}[A-Za-z]?$", na=False)]  # real products only
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    # Drop extreme bulk one-off orders (>99.9th pct line quantity) which are wholesale outliers
    cap = df["Quantity"].quantile(0.999)
    df = df[df["Quantity"] <= cap]

    # Choose SKUs: high volume AND sold on many days (stable demand history)
    stats = df.groupby("StockCode").agg(units=("Quantity", "sum"), days=("Date", "nunique"))
    stats = stats[stats["days"] >= 400].sort_values("units", ascending=False).head(TOP_N)
    sel = df[df["StockCode"].isin(stats.index)]

    desc = (sel.groupby("StockCode")["Description"]
              .agg(lambda s: s.dropna().astype(str).str.strip().mode().iat[0] if s.notna().any() else ""))
    daily = (sel.groupby(["StockCode", "Date"])
                .agg(demand=("Quantity", "sum"), revenue=("Quantity", lambda q: 0))
                .reset_index())
    rev = sel.assign(rev=sel["Quantity"] * sel["Price"]).groupby(["StockCode", "Date"])["rev"].sum()
    daily["revenue"] = rev.values
    daily["avg_price"] = (daily["revenue"] / daily["demand"]).round(4)

    # Complete calendar per SKU (days with no sales -> demand 0)
    trading_days = pd.DatetimeIndex(sorted(df["Date"].unique()))  # days the store traded at all
    full = []
    for sku, g in daily.groupby("StockCode"):
        g = g.set_index("Date").reindex(trading_days)
        g["StockCode"] = sku
        g["demand"] = g["demand"].fillna(0)
        g["revenue"] = g["revenue"].fillna(0)
        g["avg_price"] = g["avg_price"].ffill().bfill()
        full.append(g.rename_axis("Date").reset_index())
    out = pd.concat(full, ignore_index=True)
    out["Description"] = out["StockCode"].map(desc)
    out = out[["Date", "StockCode", "Description", "demand", "avg_price", "revenue"]]
    out = out.rename(columns={"StockCode": "sku", "Description": "description", "Date": "date"})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sku_daily_demand.csv", index=False)
    print(f"raw rows={n_raw:,} clean rows={len(df):,} skus={out['sku'].nunique()} "
          f"days={len(trading_days)} range={trading_days.min().date()}..{trading_days.max().date()} "
          f"out rows={len(out):,}")

if __name__ == "__main__":
    main()
