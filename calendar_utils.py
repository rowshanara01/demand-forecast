"""Bangladesh style calendar: Eid + fixed chhutir din. Forecast e 'agamikal chhuti ase kina' khub gurutwopurno."""
import numpy as np
import pandas as pd

EID = pd.to_datetime(["2022-05-03", "2022-07-10", "2023-04-21", "2023-06-29", "2024-04-11",
                      "2024-06-17", "2025-03-31", "2025-06-07", "2026-03-20", "2026-05-27", "2027-03-10", "2027-05-17"])
FIXED = [(2, 21), (3, 26), (4, 14), (12, 16)]  # Language Day, Independence Day, Pohela Boishakh, Victory Day
HOLIDAYS = pd.DatetimeIndex(sorted(set(EID) | {pd.Timestamp(y, m, d) for y in range(2022, 2028) for m, d in FIXED}))


def _days_to_next(dates, ref, cap=30):
    idx = np.minimum(np.searchsorted(ref.values, dates.values, side="left"), len(ref) - 1)
    return np.clip((ref.values[idx] - dates.values).astype("timedelta64[D]").astype(int), 0, cap)


def calendar_features(dates):
    dates = pd.DatetimeIndex(dates)
    to_eid = _days_to_next(dates, EID)
    return pd.DataFrame({
        "dow": dates.dayofweek, "month": dates.month, "day_of_month": dates.day,
        "is_holiday": dates.isin(HOLIDAYS).astype(int),
        "days_to_holiday": _days_to_next(dates, HOLIDAYS),
        "pre_eid": ((to_eid >= 1) & (to_eid <= 7)).astype(int),  # Eid er age 7 din kenakata bare
    }, index=range(len(dates)))
