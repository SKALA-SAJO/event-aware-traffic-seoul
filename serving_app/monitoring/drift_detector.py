"""
드리프트 판정 - "알려진·일시적 변화"와 "예정에 없던 지속적 변화"를 구분하는 운영 정책.

    corridor 별로 최근 window_hours 시간의 예측 오차(정답이 도착한 것만)를 모으고, 다음 시간을 제외한 뒤
    남은 평상 시간대 RMSE 가 기준값(학습 직후 val 구간 RMSE)의 rmse_ratio_threshold 배를 넘으면 드리프트.

    제외: ① 등록된 이벤트 영향 구간 (취소된 일정은 실제로 열리지 않았으므로 제외 대상 아님)
          ② 공휴일
          ③ 돌발 정보 중 일시적인 것 - 사고·고장·낙하물·행사/집회 통제 (발생 ~ 해제 + buffer)
    제외하지 않음: 돌발 정보의 "공사" - 지속되는 구조 변화이므로 재학습 대상. 대신 드리프트 원인으로 표시.

오차는 현재 서빙 중인 모델 버전의 예측만 씁니다 (재학습 직후 이전 버전 오차로 재학습이 반복되지 않게).
"""
import numpy as np
import pandas as pd

from data import storage
from data.calendar_kr import holiday_mask
from data.config import hub_of, section
from data.events import event_mask

TRANSIENT = {"accident", "control", "event", "other"}


def incident_mask(hours: pd.DatetimeIndex, incidents: pd.DataFrame | None, buffer_hours: float) -> np.ndarray:
    """일시적 돌발(사고 등) 시간대."""
    mask = np.zeros(len(hours), dtype=bool)
    if incidents is None or incidents.empty:
        return mask
    for inc in incidents.itertuples(index=False):
        if inc.category not in TRANSIENT:
            continue
        end = inc.last_seen if not pd.isna(inc.last_seen) else inc.expected_end
        end = (end if not pd.isna(end) else inc.start + pd.Timedelta(hours=2)) + pd.Timedelta(hours=buffer_hours)
        mask |= (hours >= pd.Timestamp(inc.start).floor("h")) & (hours <= end)
    return mask


def ongoing_constructions(incidents: pd.DataFrame | None, start, end) -> list[str]:
    if incidents is None or incidents.empty:
        return []
    c = incidents[(incidents["category"] == "construction") & (incidents["start"] <= end)]
    c = c[c["expected_end"].isna() | (c["expected_end"] >= start)]
    return [f"{r.type_name or '공사'}: {r.info} (since {pd.Timestamp(r.start):%Y-%m-%d})" for r in c.itertuples()]


def evaluate_drift(errors: pd.DataFrame, events_hub: pd.DataFrame | None, reference_rmse: float | None,
                   cfg: dict | None = None, incidents: pd.DataFrame | None = None) -> dict:
    """errors: target_ts, horizon, predicted, actual (한 corridor). 온라인 판정과 오프라인 실험 ③이 공유."""
    cfg = cfg or section("drift")
    if errors.empty or not reference_rmse:
        return {"status": "insufficient_data", "n_hours": 0, "reference_rmse": reference_rmse}

    hours = pd.date_range(errors["target_ts"].min(), errors["target_ts"].max(), freq="h")
    ev = pd.Series(event_mask(hours, events_hub), index=hours)
    if cfg.get("exclude_holidays", True):
        ev |= pd.Series(holiday_mask(hours), index=hours)
    inc = pd.Series(incident_mask(hours, incidents, cfg.get("incident_buffer_hours", 1)), index=hours)
    is_ev = ev.reindex(errors["target_ts"]).to_numpy()
    is_inc = inc.reindex(errors["target_ts"]).to_numpy() & ~is_ev
    is_excl = is_ev | is_inc

    sq = (errors["predicted"].to_numpy() - errors["actual"].to_numpy()) ** 2
    kept_hours = errors.loc[~is_excl, "target_ts"].nunique()
    rmse = float(np.sqrt(sq[~is_excl].mean())) if (~is_excl).any() else None
    rmse_known = float(np.sqrt(sq[is_ev].mean())) if is_ev.any() else None
    ratio = rmse / reference_rmse if rmse is not None else None

    if kept_hours < cfg["min_samples"]:
        status = "insufficient_data"
    elif ratio > cfg["rmse_ratio_threshold"]:
        status = "drift"
    else:
        status = "ok"
    return {
        "status": status,
        "rmse": rmse,
        "reference_rmse": reference_rmse,
        "ratio": ratio,
        "threshold": cfg["rmse_ratio_threshold"],
        "n_hours": int(kept_hours),
        "excluded_hours": int(errors.loc[is_ev, "target_ts"].nunique()),
        "excluded_incident_hours": int(errors.loc[is_inc, "target_ts"].nunique()),
        "rmse_known_event_hours": rmse_known,
        "constructions": ongoing_constructions(incidents, hours[0], hours[-1]),
    }


def drift_status(corridor: str, model, now=None) -> dict:
    """서빙 중인 model 의 최근 예측 오차로 corridor 의 드리프트 여부를 판정."""
    cfg = section("drift")
    now = pd.Timestamp(now) if now is not None else storage.last_observation_ts(corridor)
    if now is None:
        return {"corridor": corridor, "status": "insufficient_data", "n_hours": 0}
    since = now - pd.Timedelta(hours=cfg["window_hours"] - 1)
    hub = hub_of(corridor)
    errors = storage.load_prediction_errors(corridor, since=since, until=now, model_version=model.version)
    events = storage.load_events(hubs=[hub], start=since - pd.Timedelta(hours=6), end=now + pd.Timedelta(hours=6))
    incidents = storage.load_incidents(hubs=[hub], start=since, end=now)
    if len(incidents):  # 링크가 매핑된 돌발은 해당 corridor 것만, 매핑이 없으면 거점 전체에 적용
        incidents = incidents[incidents["corridor"].isna() | (incidents["corridor"] == corridor)]
    result = evaluate_drift(errors, events, model.reference_rmse.get(corridor), cfg, incidents)
    return {"corridor": corridor, "hub": hub, "model_version": model.version, "window_end": str(now), **result}
