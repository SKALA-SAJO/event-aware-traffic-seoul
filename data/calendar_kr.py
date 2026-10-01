"""
캘린더 피처: 시각·요일(주기 인코딩), 주말, 공휴일.

공휴일은 출퇴근 정체가 사라져 속도가 "반대로" 오르는 예정된 이벤트입니다. 날짜가
미리 확정되므로 예측 대상 시각 기준의 미래 정보로 넣을 수 있습니다.
"""
import numpy as np
import pandas as pd

from data.config import holidays

CALENDAR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", "holiday"]


def holiday_mask(ts: pd.DatetimeIndex) -> np.ndarray:
    hol = holidays()
    return np.array([d in hol for d in ts.date], dtype=bool)


def calendar_features(ts: pd.DatetimeIndex) -> np.ndarray:
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    hol = holiday_mask(ts)
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(float),
            hol.astype(float),
        ],
        axis=1,
    ).astype("float32")
