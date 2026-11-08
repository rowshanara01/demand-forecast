import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from features import build_features

client = TestClient(app)


def test_no_leakage_features_ignore_today_and_recent_days():
    raw = pd.read_csv("data/sales.csv", parse_dates=["date"])
    one = raw[(raw.shop_id == 1) & (raw.product_id == 1)].reset_index(drop=True)
    base = build_features(one)
    t = 200
    changed = one.copy()
    changed.loc[t - 6:t, "units_sold"] = 10_000      # ajker ar ager 6 din er sales bodli
    new = build_features(changed)
    cols = [c for c in base.columns if c.startswith(("lag_", "roll_"))]
    # t din er feature e kono poribortan hoyni, karon feature shudhu 7+ din ager data dekhe
    pd.testing.assert_series_equal(base.loc[t, cols], new.loc[t, cols])


def test_forecast_shape_and_interval_order():
    r = client.post("/forecast", json={"shop_id": 1, "product_id": 3}).json()
    assert len(r["days"]) == 7
    assert all(d["lower"] <= d["forecast"] <= d["upper"] for d in r["days"])


def test_promo_increases_forecast():
    base = client.post("/forecast", json={"shop_id": 2, "product_id": 1}).json()["total_7d"]
    promo = client.post("/forecast", json={"shop_id": 2, "product_id": 1, "promo": [1] * 7}).json()["total_7d"]
    assert promo > base


def test_unknown_series_and_validation():
    assert client.post("/forecast", json={"shop_id": 99, "product_id": 1}).status_code == 404
    assert client.post("/forecast", json={"shop_id": 1, "product_id": 1, "promo": [1, 0]}).status_code == 422


def test_model_beats_baselines():
    import json
    m = json.load(open("model/metrics.json"))["test"]
    assert m["hist_gb_poisson"]["wape"] < m["seasonal_naive"]["wape"]
    assert m["hist_gb_poisson"]["wape"] < m["moving_avg_28"]["wape"]
