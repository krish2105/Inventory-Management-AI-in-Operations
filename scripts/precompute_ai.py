"""Pre-computes the walk-forward AI forecasts (26-week hold-out, weekly refit + drift-triggered
model re-selection) so the dashboard's Before-vs-After tab opens instantly. The dashboard can
recompute them itself after new data is uploaded."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inventory_core import data as D, automation as A

df = D.load_default()
roll, log = A.rolling_ai_all(df, 26, progress=lambda i, n, s: print(f"{i}/{n} {s}", flush=True))
roll.to_csv(ROOT / "data/processed/ai_rolling_forecasts.csv", index=False)
log.to_csv(ROOT / "data/processed/ai_retrain_log.csv", index=False)
print("done", len(roll), len(log))
