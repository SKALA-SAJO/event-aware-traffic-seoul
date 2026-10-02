"""
감지 기반 조기 재학습의 "판단" 단계 - 일별 감지 기록과 재학습 신청 파일.

    관찰: check_alert_only 가 corridor 별 드리프트를 판정해 [WARN] 을 남기고 이 모듈에 그날의 감지를 기록
    판단: 같은 corridor 가 연속 consecutive_days 일 감지됐거나, 최근 recent_construction_days 일 안에 시작한
          공사가 있는 상태에서 감지되면 재학습을 "신청" (logs/retrain_request.json)
    행동: 서버가 아니라 별도 프로세스(scripts/periodic_retrain.sh → retrain_schedule.py)가 신청을 읽어 실행

날짜는 실제 날짜가 아니라 판정 구간의 끝(window_end) 날짜입니다 (시뮬레이션 데이터로도 검증 가능).
상태·신청 파일은 로컬 전용(logs/ 는 gitignore)이고, 없거나 깨져 있으면 "기록 없음"으로 봅니다.
모델 버전이 바뀌면 감지 기록을 비웁니다 (새 버전은 판정 보류 구간이 있어 이전 감지와 이어지지 않음).

이 파일은 TensorFlow 를 불러오지 않습니다 (매시 도는 retrain_schedule.py 가 가볍게 유지되도록).
"""
import datetime as dt
import json
import os

import pandas as pd

STATE_PATH = os.getenv("DRIFT_STATE_PATH", "logs/drift_state.json")
REQUEST_PATH = os.getenv("RETRAIN_REQUEST_PATH", "logs/retrain_request.json")
_KEEP_DAYS = 30  # 이보다 오래된 일별 기록은 지움


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _load_state(version: str) -> dict:
    state = _read(STATE_PATH)
    if state.get("model_version") != version:
        state = {"model_version": version}
    state.setdefault("drift_days", {})   # {corridor: {날짜: {"construction": bool}}}
    state.setdefault("seen", [])         # 하루 1회만 로그를 남기기 위한 "종류|corridor|날짜" 키
    return state


def first_today(kind: str, corridor: str, ts, version: str) -> bool:
    """같은 종류의 로그를 corridor·날짜당 1회만 남기기 위한 확인. 처음이면 True 를 돌려주고 기록."""
    state = _load_state(version)
    key = f"{kind}|{corridor}|{_day(ts)}"
    if key in state["seen"]:
        return False
    cutoff = _day(pd.Timestamp(ts) - pd.Timedelta(days=_KEEP_DAYS))
    state["seen"] = [k for k in state["seen"] if k.rsplit("|", 1)[1] >= cutoff] + [key]
    _write(STATE_PATH, state)
    return True


def record_drift(corridor: str, ts, version: str, recent_construction: bool) -> int:
    """그날 감지를 기록하고, ts 날짜로 끝나는 연속 감지 일수를 돌려줍니다."""
    state = _load_state(version)
    days = state["drift_days"].setdefault(corridor, {})
    entry = days.setdefault(_day(ts), {"construction": False})
    entry["construction"] = entry["construction"] or bool(recent_construction)
    cutoff = _day(pd.Timestamp(ts) - pd.Timedelta(days=_KEEP_DAYS))
    state["drift_days"][corridor] = {d: v for d, v in days.items() if d >= cutoff}
    _write(STATE_PATH, state)
    return consecutive_days(state["drift_days"][corridor], ts)


def consecutive_days(days: dict, ts) -> int:
    """ts 날짜부터 거꾸로, 달력상 연속으로 감지가 있던 일수. 하루라도 빠지면 거기서 끊김."""
    d, n = pd.Timestamp(ts).normalize(), 0
    while d.strftime("%Y-%m-%d") in days:
        n += 1
        d -= pd.Timedelta(days=1)
    return n


def recent_construction(incidents: pd.DataFrame | None, corridor: str, ts, days: int) -> list[str]:
    """ts 기준 최근 days 일 안에 시작한 공사 (링크가 매핑된 돌발은 해당 corridor 것만, 매핑 없으면 거점 전체)."""
    if incidents is None or incidents.empty:
        return []
    c = incidents[(incidents["category"] == "construction")
                  & (incidents["corridor"].isna() | (incidents["corridor"] == corridor))]
    ts = pd.Timestamp(ts)
    c = c[(c["start"] <= ts) & (c["start"] >= ts - pd.Timedelta(days=days))]
    return [f"{r.type_name or '공사'}: {r.info} (since {pd.Timestamp(r.start):%Y-%m-%d})" for r in c.itertuples()]


def early_reason(consecutive: int, construction: bool, cfg: dict) -> str | None:
    """신청 조건. 만족하면 사유 문자열, 아니면 None."""
    if construction:
        return f"recent construction + drift (consecutive_days={consecutive})"
    if consecutive >= cfg["consecutive_days"]:
        return f"drift {consecutive} consecutive days"
    return None


# ───────────────────────────── 재학습 신청 파일 ─────────────────────────────

def read_request() -> dict | None:
    req = _read(REQUEST_PATH)
    return req if req.get("created_at") else None


def write_request(corridor: str, reason: str, now: dt.datetime | None = None) -> bool:
    """신청이 이미 있으면 그대로 두고 False (처음 신청한 시각 유지), 새로 썼으면 True."""
    if read_request():
        return False
    now = now or dt.datetime.now()
    _write(REQUEST_PATH, {"created_at": now.strftime("%Y-%m-%d %H:%M:%S"), "corridor": corridor, "reason": reason})
    return True


def clear_request() -> None:
    try:
        os.remove(REQUEST_PATH)
    except OSError:
        pass
