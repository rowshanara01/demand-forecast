"""Forecast API: porer 7 din er demand + 80% interval, promo plan dile 'what-if'"""
import json
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import FEATURES, build_features

B = joblib.load("model/forecaster.joblib")
HISTORY = B["history"]
LAST = HISTORY.date.max()
HORIZON = 7
app = FastAPI(title="Demand Forecast API")


class Request(BaseModel):
    shop_id: int
    product_id: int
    promo: Optional[list[Literal[0, 1]]] = Field(default=None, min_length=HORIZON, max_length=HORIZON,
                                                 description="porer 7 dine kon kon din promo thakbe")


@app.get("/health")
def health():
    return {"status": "ok", "last_data_date": str(LAST.date()), "horizon_days": HORIZON}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/forecast")
def forecast(req: Request):
    hist = HISTORY[(HISTORY.shop_id == req.shop_id) & (HISTORY.product_id == req.product_id)]
    if hist.empty:
        raise HTTPException(404, "shop/product combination not found")
    hist = hist.tail(60)  # lag_28 + rolling_28 (7 din shift) er jonno 60 din jothesto
    normal_price = float(hist[hist.promo == 0].price.median())
    promo = req.promo or [0] * HORIZON
    future = pd.DataFrame({
        "date": pd.date_range(LAST + pd.Timedelta(days=1), periods=HORIZON),
        "shop_id": req.shop_id, "product_id": req.product_id,
        "price": [normal_price * (0.8 if p else 1.0) for p in promo],
        "promo": promo, "units_sold": np.nan})
    X = build_features(pd.concat([hist, future]))
    X = X[X.date > LAST]
    mean = B["mean"].predict(X[FEATURES])
    lo = np.minimum(B["q10"].predict(X[FEATURES]), mean)
    hi = np.maximum(B["q90"].predict(X[FEATURES]), mean)
    days = [{"date": str(d.date()), "forecast": round(float(m), 1), "lower": round(float(l), 1),
             "upper": round(float(h), 1), "promo": int(p)}
            for d, m, l, h, p in zip(X.date, mean, lo, hi, promo)]
    return {"shop_id": req.shop_id, "product_id": req.product_id,
            "total_7d": round(float(mean.sum()), 1), "days": days}
