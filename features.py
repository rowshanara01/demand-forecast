"""Feature engineering. RULE: lag minimum 7 din, tai 7 din porjonto forecast korte recursion lage na ar leakage hoy na."""
import pandas as pd

from calendar_utils import calendar_features

KEYS = ["shop_id", "product_id"]
LAGS = [7, 14, 28]
FEATURES = (["shop_id", "product_id", "price", "promo", "dow", "month", "day_of_month", "is_holiday",
             "days_to_holiday", "pre_eid"] + [f"lag_{l}" for l in LAGS] +
            ["roll_mean_7", "roll_mean_28", "roll_std_28"])


def build_features(df):
    """df: date, shop_id, product_id, price, promo, units_sold (future row e units_sold = NaN)"""
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True).copy()
    g = df.groupby(KEYS)["units_sold"]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    df["_s7"] = g.shift(7)                       # aj theke kompokkhe 7 din age porjonto-i dekhi
    gg = df.groupby(KEYS)["_s7"]
    df["roll_mean_7"] = gg.transform(lambda x: x.rolling(7, min_periods=3).mean())
    df["roll_mean_28"] = gg.transform(lambda x: x.rolling(28, min_periods=7).mean())
    df["roll_std_28"] = gg.transform(lambda x: x.rolling(28, min_periods=7).std())
    cal = calendar_features(df["date"])
    return pd.concat([df.drop(columns="_s7"), cal], axis=1)
