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
from data.features import make_windows
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
        return {"status": "insufficient_data", "reason": "no_predictions", "n_hours": 0,
                "min_samples": cfg["min_samples"], "reference_rmse": reference_rmse}

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

    reason = None
    if kept_hours < cfg["min_samples"]:
        status, reason = "insufficient_data", "too_few_hours"  # 이벤트·공휴일·돌발 제외 후 남은 시간이 모자람
    elif ratio > cfg["rmse_ratio_threshold"]:
        status = "drift"
    else:
        status = "ok"
    return {
        "status": status,
        "reason": reason,
        "min_samples": cfg["min_samples"],
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


_REPLAY_CACHE: dict = {}


def replay_errors(corridor: str, model, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """
    예측 기록이 없을 때(실시간 운영 전·발표 시연) 판정 구간을 현재 모델로 다시 예측해 실측과 비교한다.
    운영과 같은 방식 - 매시 발행, 향후 1~6시간, 발행 시각까지 공개된 일정만(학습과 같은 규칙) - 으로
    "그 기간에 서비스가 돌았다면 기록됐을 예측"을 재현한다. DB 에는 저장하지 않는다.
    """
    key = (corridor, model.version, str(since), str(now))
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    H = model.spec.horizon
    try:
        frame = model.unit_frame(corridor, since - pd.Timedelta(weeks=8), now)
    except LookupError:
        return pd.DataFrame(columns=["target_ts", "horizon", "predicted", "actual"])
    w = make_windows({corridor: frame}, model.spec, issue_start=since - pd.Timedelta(hours=H),
                     issue_end=now - pd.Timedelta(hours=1), require_targets=False)
    pred = model.predict_windows(w)
    rows = []
    for i, issued in enumerate(pd.to_datetime(w.issued_at)):
        for h in range(H):
            t = issued + pd.Timedelta(hours=h + 1)
            if since <= t <= now and not np.isnan(w.actual[i, h]):
                rows.append({"target_ts": t, "horizon": h + 1, "predicted": float(pred[i, h]),
                             "actual": float(w.actual[i, h])})
    out = pd.DataFrame(rows, columns=["target_ts", "horizon", "predicted", "actual"])
    _REPLAY_CACHE[key] = out
    return out


def drift_status(corridor: str, model, now=None, replay_if_empty: bool = True) -> dict:
    """
    서빙 중인 model 의 최근 예측 오차로 corridor 의 드리프트 여부를 판정.
    예측 기록이 하나도 없으면(replay_if_empty) 같은 구간을 다시 예측해 판정하고 mode="replay" 로 표시.
    """
    cfg = section("drift")
    now = pd.Timestamp(now) if now is not None else storage.last_observation_ts(corridor)
    if now is None:
        return {"corridor": corridor, "status": "insufficient_data", "n_hours": 0}
    since = now - pd.Timedelta(hours=cfg["window_hours"] - 1)
    hub = hub_of(corridor)
    errors = storage.load_prediction_errors(corridor, since=since, until=now, model_version=model.version)
    mode = "logged"
    if errors.empty and replay_if_empty:
        errors, mode = replay_errors(corridor, model, since, now), "replay"
    events = storage.load_events(hubs=[hub], start=since - pd.Timedelta(hours=6), end=now + pd.Timedelta(hours=6))
    incidents = storage.load_incidents(hubs=[hub], start=since, end=now)
    if len(incidents):  # 링크가 매핑된 돌발은 해당 corridor 것만, 매핑이 없으면 거점 전체에 적용
        incidents = incidents[incidents["corridor"].isna() | (incidents["corridor"] == corridor)]
    result = evaluate_drift(errors, events, model.reference_rmse.get(corridor), cfg, incidents)
    return {"corridor": corridor, "hub": hub, "model_version": model.version, "mode": mode,
            "window_start": str(since), "window_end": str(now), **result}
