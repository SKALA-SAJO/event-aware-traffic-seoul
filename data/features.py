"""
관측치·이벤트 → LSTM 입력 텐서 (학습·평가·서빙·드리프트 시뮬레이션 공용).

학습과 서빙이 같은 함수로 입력을 만들어야 "서빙 시점 입력"과 "학습 시점 입력"이
어긋나는 사고를 막을 수 있습니다 (HAIC 실습의 data/features.py 와 같은 원칙).

예측 문제 (발행 시각 t 기준)
    past   (LOOKBACK, F_past)  : t-LOOKBACK+1 … t 시각의 [속도, 교통량, 캘린더, 이벤트]
    future (HORIZON,  F_future): t+1 … t+HORIZON 시각의 [캘린더, 이벤트, 거점 one-hot]
                                 → 공휴일·이벤트 일정처럼 "미리 알려진 미래 정보"
    target (HORIZON,)          : t+1 … t+HORIZON 시각의 통행속도 (거점별 표준화)

교통량은 입력으로만 씁니다. 집회로 도로가 통제되면 통과 차량(교통량)은 줄지만 정체는
심해지므로, 사용자에게 의미 있는 예측 대상은 속도(→ 소요시간)입니다.

실험 ① 피처 단계 (STAGES)
    speed    : 속도만
    volume   : + 교통량 (+ 결측 마스크)
    calendar : + 시각·요일·공휴일
    full     : + 이벤트 피처 (인코딩 방식은 event_mode, data/events.py)
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.calendar_kr import CALENDAR_FEATURES, calendar_features, holiday_mask
from data.config import HORIZON, LOOKBACK, section
from data.events import EVENT_MODES, event_feature_names, event_features, event_mask

STAGES = ["speed", "volume", "calendar", "full"]
SEASONAL_LAG = 168  # 단순 예측법: 지난주 같은 요일·같은 시각


@dataclass
class FeatureSpec:
    units: list[str]          # 예측 단위 = corridor ID (거점 도로 × 방향)
    stage: str = "full"
    event_mode: str = "text"
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    # 이벤트 피처에 쓰는 출처 (None = 전부). 이벤트 시간대 판정(is_event: 평가·드리프트 제외)은 출처와 무관하게
    # 모든 이벤트를 씀. 모델 메타데이터에 함께 저장되므로 서빙도 학습과 같은 출처만 피처로 씀.
    event_sources: list[str] | None = field(default_factory=lambda: section("train").get("event_sources"))

    def __post_init__(self):
        assert self.stage in STAGES, self.stage
        assert self.event_mode in EVENT_MODES, self.event_mode
        if self.stage != "full":
            self.event_mode = "none"

    @property
    def key(self) -> str:
        return self.stage if self.stage != "full" else f"full-{self.event_mode}"

    @property
    def uses_volume(self) -> bool:
        return self.stage in ("volume", "calendar", "full")

    @property
    def uses_calendar(self) -> bool:
        return self.stage in ("calendar", "full")

    @property
    def event_names(self) -> list[str]:
        return event_feature_names(self.event_mode)

    def past_names(self) -> list[str]:
        names = ["speed_z"]
        if self.uses_volume:
            names += ["vol_z", "vol_missing"]
        if self.uses_calendar:
            names += CALENDAR_FEATURES
        return names + self.event_names

    def future_names(self) -> list[str]:
        names = list(CALENDAR_FEATURES) if self.uses_calendar else []
        return names + self.event_names + [f"unit_{u}" for u in self.units]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        d = dict(d)
        if "hubs" in d:  # 거점 단위로 학습된 이전 모델 메타데이터 호환
            d["units"] = d.pop("hubs")
        return cls(**d)


class TrafficScaler:
    """
    corridor 별 표준화 스케일러. corridor 마다 평소 속도 수준이 크게 달라(광화문 20km/h대 vs 상암 40km/h대)
    하나의 전역 스케일로는 작은 corridor 의 변화가 묻히므로 corridor 별 평균·표준편차를 씁니다.
    교통량은 log1p 후 표준화합니다.

    base 학습 때 train 구간으로 한 번 fit 하고 모델과 함께 저장합니다. fine-tuning 때 다시
    fit 하지 않는 이유는 HAIC 실습과 같습니다 - 기존 가중치가 학습한 스케일과 어긋나기 때문.
    """

    def __init__(self, stats: dict | None = None):
        self.stats = stats or {}

    def fit(self, obs: pd.DataFrame) -> "TrafficScaler":
        for hub, g in obs.groupby("corridor"):
            vol = np.log1p(g["volume"].where(g["volume"] > 0).dropna())
            self.stats[hub] = {
                "speed_mean": float(g["speed"].mean()),
                "speed_std": float(g["speed"].std() or 1.0),
                "vol_mean": float(vol.mean()) if len(vol) else 0.0,
                "vol_std": float(vol.std()) if len(vol) > 1 else 1.0,
            }
        return self

    def speed_z(self, hub: str, speed):
        s = self.stats[hub]
        return (np.asarray(speed, dtype="float64") - s["speed_mean"]) / s["speed_std"]

    def inverse_speed(self, hubs, z) -> np.ndarray:
        """hubs: 샘플별 corridor (n,), z: (n, H) → km/h"""
        z = np.asarray(z, dtype="float64")
        mean = np.array([self.stats[h]["speed_mean"] for h in hubs])[:, None]
        std = np.array([self.stats[h]["speed_std"] for h in hubs])[:, None]
        return z * std + mean

    def vol_z(self, hub: str, volume):
        s = self.stats[hub]
        return (np.log1p(np.asarray(volume, dtype="float64")) - s["vol_mean"]) / s["vol_std"]

    def to_dict(self) -> dict:
        return {"stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "TrafficScaler":
        return cls(d["stats"])


def build_frame(
    hub: str,
    obs_hub: pd.DataFrame,
    events_hub: pd.DataFrame,
    spec: FeatureSpec,
    scaler: TrafficScaler,
    start=None,
    end=None,
    known_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    한 corridor 의 1시간 간격 연속 프레임 (events_hub 는 그 corridor 가 속한 거점의 이벤트). end 를 마지막 관측 이후로 잡으면(서빙) 그 구간은 속도가
    NaN 이고 캘린더·이벤트 같은 미래 정보만 채워집니다.
    """
    start = pd.Timestamp(start if start is not None else obs_hub["ts"].min())
    end = pd.Timestamp(end if end is not None else obs_hub["ts"].max())
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    df = obs_hub.set_index("ts")[["speed", "volume"]].reindex(hours)

    frame = pd.DataFrame(index=hours)
    frame["speed"] = df["speed"].interpolate(limit=3, limit_area="inside")
    frame["speed_z"] = scaler.speed_z(hub, frame["speed"])

    # 교통량: API·엑셀 모두 결측을 0으로 기록하므로 0 은 결측으로 본다.
    # 같은 요일·시각 중앙값으로 채우고, 채웠다는 사실은 vol_missing 마스크로 모델에 알려준다.
    vol = df["volume"].where(df["volume"] > 0)
    frame["vol_missing"] = vol.isna().astype("float32")
    key = [hours.dayofweek, hours.hour]
    vol = vol.fillna(vol.groupby(key).transform("median"))
    vol = vol.fillna(vol.median() if vol.notna().any() else np.expm1(scaler.stats[hub]["vol_mean"]))
    frame["vol_z"] = scaler.vol_z(hub, vol)

    cal = calendar_features(hours)
    for i, name in enumerate(CALENDAR_FEATURES):
        frame[name] = cal[:, i]
    feat_events = events_hub
    if spec.event_sources is not None and "source" in events_hub:
        feat_events = events_hub[events_hub["source"].isin(spec.event_sources)]
    ev = event_features(hours, feat_events, spec.event_mode, known_cutoff=known_cutoff)
    for i, name in enumerate(spec.event_names):
        frame[name] = ev[:, i]

    frame["is_event"] = event_mask(hours, events_hub)
    frame["is_holiday"] = holiday_mask(hours)
    frame["naive"] = frame["speed"].shift(SEASONAL_LAG)
    return frame


@dataclass
class Windows:
    X_past: np.ndarray            # (n, L, F_past)
    X_fut: np.ndarray             # (n, H, F_future)
    y: np.ndarray                 # (n, H) 표준화된 목표 속도 (정답 없는 서빙 샘플은 NaN)
    units: np.ndarray             # (n,) corridor
    issued_at: np.ndarray         # (n,) datetime64
    actual: np.ndarray            # (n, H) km/h
    is_event: np.ndarray          # (n, H) bool
    is_holiday: np.ndarray        # (n, H) bool
    naive: np.ndarray             # (n, H) km/h, 지난주 같은 시각
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "Windows":
        return Windows(
            self.X_past[mask], self.X_fut[mask], self.y[mask], self.units[mask], self.issued_at[mask],
            self.actual[mask], self.is_event[mask], self.is_holiday[mask], self.naive[mask],
        )

    @staticmethod
    def concat(parts: list["Windows"]) -> "Windows":
        parts = [p for p in parts if len(p)]
        return Windows(*[np.concatenate([getattr(p, f) for p in parts]) for f in (
            "X_past", "X_fut", "y", "units", "issued_at", "actual", "is_event", "is_holiday", "naive")])


def make_windows(
    frames: dict[str, pd.DataFrame],
    spec: FeatureSpec,
    issue_start=None,
    issue_end=None,
    require_targets: bool = True,
) -> Windows:
    """[issue_start, issue_end] 범위의 모든 발행 시각에 대해 (past, future, target) 샘플을 만든다."""
    L, H = spec.lookback, spec.horizon
    past_cols, fut_cols = spec.past_names(), [c for c in spec.future_names() if not c.startswith("unit_")]
    parts = []
    for hub, fr in frames.items():
        T = len(fr)
        if T < L + H:
            continue
        P = fr[past_cols].to_numpy("float32")
        F = fr[fut_cols].to_numpy("float32") if fut_cols else np.zeros((T, 0), "float32")
        onehot = np.array([[1.0 if u == hub else 0.0 for u in spec.units]], dtype="float32")

        issue_pos = np.arange(L - 1, T - H)  # 발행 시각 위치
        times = fr.index[issue_pos]
        keep = np.ones(len(issue_pos), dtype=bool)
        if issue_start is not None:
            keep &= times >= pd.Timestamp(issue_start)
        if issue_end is not None:
            keep &= times <= pd.Timestamp(issue_end)
        issue_pos = issue_pos[keep]
        if not len(issue_pos):
            continue

        Xp = sliding_window_view(P, L, axis=0).transpose(0, 2, 1)[issue_pos - L + 1]
        Xf = sliding_window_view(F, H, axis=0).transpose(0, 2, 1)[issue_pos + 1]
        Xf = np.concatenate([Xf, np.broadcast_to(onehot, (len(issue_pos), H, onehot.shape[1]))], axis=2)

        def fut(col, fr=fr, issue_pos=issue_pos):
            return sliding_window_view(fr[col].to_numpy(), H)[issue_pos + 1]

        y = fut("speed_z")
        ok = ~np.isnan(Xp[:, :, 0]).any(axis=1)
        if require_targets:
            ok &= ~np.isnan(y).any(axis=1)
        parts.append(Windows(
            Xp[ok].astype("float32"), Xf[ok].astype("float32"), y[ok].astype("float32"),
            np.array([hub] * int(ok.sum())), fr.index[issue_pos][ok].to_numpy(),
            fut("speed")[ok], fut("is_event")[ok].astype(bool), fut("is_holiday")[ok].astype(bool), fut("naive")[ok],
        ))
    if not parts:
        empty = np.zeros((0,))
        return Windows(np.zeros((0, L, len(past_cols)), "float32"), np.zeros((0, H, len(spec.future_names())), "float32"),
                       np.zeros((0, H), "float32"), empty.astype(str), empty.astype("datetime64[ns]"),
                       np.zeros((0, H)), np.zeros((0, H), bool), np.zeros((0, H), bool), np.zeros((0, H)))
    return Windows.concat(parts)


def build_frames(
    obs: pd.DataFrame,
    events: pd.DataFrame,
    spec: FeatureSpec,
    scaler: TrafficScaler,
    end=None,
    known_cutoff=None,
) -> dict[str, pd.DataFrame]:
    frames = {}
    from data.config import hub_of, hubs

    hub_set = set(hubs(include_disabled=True))
    for unit in spec.units:
        o = obs[obs["corridor"] == unit]
        if o.empty:
            continue
        hub = unit if unit in hub_set else hub_of(unit)  # 단위가 거점 자체인 경우(실험 ④ 거점 평균)
        e = events[events["hub"] == hub] if events is not None and len(events) else None
        frames[unit] = build_frame(unit, o, e, spec, scaler, end=end, known_cutoff=known_cutoff)
    return frames


def usual_speed(frame: pd.DataFrame, until, weeks: int = 8) -> dict[tuple[int, int], float]:
    """
    "평소" 속도 프로필: 최근 N주 동안 같은 요일·시각의 중앙값 (이벤트·공휴일 시간 제외).
    대시보드의 "평소 대비 추가 소요시간" 계산 기준입니다.
    잠실 토요일 저녁처럼 거의 매주 이벤트가 있는 시간대는 이벤트를 빼면 표본이 없으므로
    같은 주중/주말·시각 → (그래도 없으면) 이벤트 포함 같은 요일·시각 순으로 대체합니다.
    """
    until = pd.Timestamp(until)
    hist = frame.loc[(frame.index > until - pd.Timedelta(weeks=weeks)) & (frame.index <= until)].dropna(subset=["speed"])
    clean = hist[~hist["is_event"] & ~hist["is_holiday"]]
    by_dow = clean.groupby([clean.index.dayofweek, clean.index.hour])["speed"].median()
    by_daytype = clean.groupby([clean.index.dayofweek >= 5, clean.index.hour])["speed"].median()
    by_dow_all = hist.groupby([hist.index.dayofweek, hist.index.hour])["speed"].median()
    out = {}
    for dow in range(7):
        for hour in range(24):
            for prof, key in ((by_dow, (dow, hour)), (by_daytype, (dow >= 5, hour)), (by_dow_all, (dow, hour))):
                if key in prof.index:
                    out[(dow, hour)] = float(prof[key])
                    break
    return out
