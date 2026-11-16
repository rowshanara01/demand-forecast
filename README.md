#### Demand Forecasting (Sales Prediction) ####

Predicts how much a shop or company will sell over the next 7 days, so you can plan stock, staffing and supply ahead of time.

## do to Run ##

```bash
pip install -r requirements.txt
python generate_data.py          # 3 shops x 5 products x 3 years of daily sales
python train.py                  # backtest + final test + interval + feature importance
pytest -q
uvicorn app.main:app --reload    # localhost:8000/docs
```

Example API requests:

```http
POST /forecast {"shop_id": 1, "product_id": 3}
POST /forecast {"shop_id": 1, "product_id": 3, "promo": [0,0,1,1,0,0,0]}   # what-if
```

## What Companies Use in Practice (and This Project Includes) ##

- **Never use a random split on time series.** Split by time and use an expanding-window backtest.
- **Baselines first** (seasonal naive, moving average) to check whether the ML model actually beats them.
- **Leakage-safe features:** lags start from 7 days back, and this is proven by a test (`tests/test_forecast.py`).
- **Calendar features:** Eid, public holidays, and pre-Eid shopping rush.
- **Poisson loss**, since sales are a count, and **WAPE** as the metric.
- **Prediction intervals** (10%-90% quantile regression) with a coverage check.
- **What-if forecasts:** see how the forecast changes when you supply a promo plan.

## Limitations ##

The model only forecasts up to 7 days ahead (because lags start at 7). For a longer horizon you would need recursive forecasting or a separate model per horizon.

## What Companies Use in Practice ##

1. Try LightGBM / XGBoost.
2. Cold-start forecasting for new products with no history.
3. Replace the holiday list with real data.
4. Hierarchical forecasting: make the sum of all shops reconcile with the individual shop forecasts.
5. Monitor forecast error (compare against actuals as they arrive) and alert when error rises.
6. Build a stock reorder-point calculator: derive safety stock from the forecast and its interval.
