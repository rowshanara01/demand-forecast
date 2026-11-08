"""Synthetic daily sales: 3 shop x 5 product x 3 bochor. Trend + weekly + yearly + Eid + promo + noise"""
import numpy as np
import pandas as pd

from calendar_utils import EID, HOLIDAYS

rng = np.random.default_rng(11)
dates = pd.date_range("2023-01-01", "2025-12-31")
shop_scale = {1: 1.0, 2: 1.4, 3: 0.7}
base = {1: 20, 2: 12, 3: 35, 4: 8, 5: 15}
list_price = {1: 120, 2: 450, 3: 60, 4: 900, 5: 200}
weekly = np.array([0.9, 0.9, 0.95, 1.0, 1.3, 1.35, 1.0])  # Fri/Sat e beshi (Mon=0)

to_eid = np.array([(EID[np.searchsorted(EID.values, d.to_datetime64())] - d).days for d in dates])
rows = []
for s in shop_scale:
    for p in base:
        t = np.arange(len(dates))
        promo = (rng.random(len(dates)) < 0.12).astype(int)
        mu = (base[p] * shop_scale[s] * (1 + 0.0004 * t) * weekly[dates.dayofweek]
              * (1 + 0.15 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365))
              * np.where((to_eid >= 1) & (to_eid <= 7), 1.5, 1.0)       # Eid er age kenakata
              * np.where(dates.isin(HOLIDAYS), 0.6, 1.0)                 # chhutir din dokan kom
              * np.where(promo == 1, 1.4, 1.0))
        units = rng.poisson(mu)
        price = np.where(promo == 1, 0.8, 1.0) * list_price[p]
        rows.append(pd.DataFrame({"date": dates, "shop_id": s, "product_id": p,
                                  "price": price, "promo": promo, "units_sold": units}))
df = pd.concat(rows).sort_values(["shop_id", "product_id", "date"])
df.to_csv("data/sales.csv", index=False)
print("rows:", len(df), "| series:", df.groupby(["shop_id", "product_id"]).ngroups,
      "| avg daily units:", round(df.units_sold.mean(), 1))
