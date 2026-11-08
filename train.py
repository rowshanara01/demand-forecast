"""Forecasting: baseline vs ML | expanding-window backtest | prediction interval | leakage-safe features"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from features import FEATURES, build_features

raw = pd.read_csv("data/sales.csv", parse_dates=["date"])
df = build_features(raw)
last = df.date.max()
test_start = last - pd.Timedelta(days=89)          # last 90 din test (bhobishyot er moto)


def make(loss="poisson", **kw):
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=300,
                                         max_leaf_nodes=15, l2_regularization=1.0, random_state=0, **kw)


def wape(y, p):  # Weighted Absolute Percentage Error: sum|error| / sum(actual). Company te shobcheye beshi use hoy
    return float(np.abs(y - p).sum() / y.sum())


def scores(y, p):
    return {"wape": wape(y, p), "mae": float(np.abs(y - p).mean()), "bias": float((p.sum() - y.sum()) / y.sum())}


def evaluate(train_df, test_df):
    model = make().fit(train_df[FEATURES], train_df.units_sold)
    y = test_df.units_sold.values
    return {"seasonal_naive": scores(y, test_df.lag_7.values),
            "moving_avg_28": scores(y, test_df.roll_mean_28.values),
            "hist_gb_poisson": scores(y, model.predict(test_df[FEATURES]))}, model


# ---- 1. Backtest: 3ta expanding window (random shuffle KOKHONO na, time series e) ----
print("Backtest (WAPE, kom = bhalo):")
folds = []
for k in (180, 120, 60):
    start = test_start - pd.Timedelta(days=k)
    end = start + pd.Timedelta(days=59)
    tr, te = df[df.date < start], df[(df.date >= start) & (df.date <= end)]
    res, _ = evaluate(tr.dropna(subset=["lag_28"]), te)
    folds.append(res)
    print(f"  {start.date()}..{end.date()}", {m: round(v["wape"], 3) for m, v in res.items()})

# ---- 2. Final holdout test ----
train, test = df[df.date < test_start], df[df.date >= test_start]
res, model = evaluate(train.dropna(subset=["lag_28"]), test)
print("\nTest (last 90 days):")
for m, v in res.items():
    print(f"  {m:16s}", {k: round(x, 3) for k, x in v.items()})

# ---- 3. Prediction interval (quantile regression, 80% band) ----
tr = train.dropna(subset=["lag_28"])
q10 = make("quantile", quantile=0.1).fit(tr[FEATURES], tr.units_sold)
q90 = make("quantile", quantile=0.9).fit(tr[FEATURES], tr.units_sold)
lo = np.minimum(q10.predict(test[FEATURES]), model.predict(test[FEATURES]))
hi = np.maximum(q90.predict(test[FEATURES]), model.predict(test[FEATURES]))
coverage = float(((test.units_sold >= lo) & (test.units_sold <= hi)).mean())
print(f"\n80% interval coverage on test: {coverage:.1%} (ideal ~80%)")

# ---- 4. Explain ----
imp = permutation_importance(model, test[FEATURES], test.units_sold, scoring="neg_mean_absolute_error",
                             n_repeats=5, random_state=0)
importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(3).tolist()), key=lambda kv: -kv[1])[:8])
print("top features:", importance)

# ---- 5. Final: shob data dekhe retrain, save (serving er jonno) ----
full = df.dropna(subset=["lag_28"])
final = {"mean": make().fit(full[FEATURES], full.units_sold),
         "q10": make("quantile", quantile=0.1).fit(full[FEATURES], full.units_sold),
         "q90": make("quantile", quantile=0.9).fit(full[FEATURES], full.units_sold),
         "history": raw, "last_date": str(last.date())}
joblib.dump(final, "model/forecaster.joblib")
json.dump({"test": res, "backtest": folds, "interval_coverage": coverage, "top_features": importance,
           "last_date": str(last.date())}, open("model/metrics.json", "w"), indent=2)
print("saved model/forecaster.joblib")
