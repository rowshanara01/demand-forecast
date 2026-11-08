## Demand Forecasting (Sales Prediction) ##

Dokan ba company agami 7 din e koto bikri hobe seta age theke bole dey. Stock, staff, supply plan er jonno.

## Chalanor niyom ##
pip install -r requirements.txt
python generate_data.py          # 3 shop x 5 product x 3 bochor daily sales
python train.py                  # backtest + final test + interval + feature importance
pytest -q
uvicorn app.main:app --reload    # localhost:8000/docs
  POST /forecast {"shop_id": 1, "product_id": 3}
  POST /forecast {"shop_id": 1, "product_id": 3, "promo": [0,0,1,1,0,0,0]}   # what-if

## Company te ja ja use hoy (ei project e ache) ##
- Time series e random split KOKHONO na: shomoy onujayi split + expanding-window backtest
- Baseline age (seasonal naive, moving average), ML model oder cheye bhalo kina dekha
- Leakage-safe feature: lag shurur 7 din theke, test diye prove kora (tests/test_forecast.py)
- Calendar feature: Eid, chhutir din, Eid er age kenakata
- Poisson loss (bikri = gonona-joggo shonkha), WAPE metric
- Prediction interval (quantile regression 10%-90%) + coverage check
- What-if forecast (promo plan dile forecast kemon bodlay)

## Shimabodhdhota ##
Shudhu 7 din agey porjonto forecast kore (lag 7 theke). Ar boro horizon dorkar hole recursive
forecast ba alada model lagbe.

## Challenge
1. LightGBM/XGBoost try koro
2. Notun product (history nai) er jonno cold-start forecast
3. Holiday list ke real data diye replace koro
4. Hierarchical forecast: shob shop er sum ar alada shop forecast jeno mile
5. Forecast error monitor koro (actual ashle forecast er sathe tulona), error baralei alert
6. Stock reorder point calculator: forecast + interval theke safety stock
