"""헬스체크 · 거점/corridor 목록 · 드리프트 현황 · 돌발 정보."""
import datetime as dt
import os

import pandas as pd
from fastapi import APIRouter, HTTPException

from data import storage
from data.config import corridors, hubs
from serving_app.errors import ModelLoadError
from serving_app import model_loader
from serving_app.monitoring.drift_detector import drift_status

router = APIRouter()


@router.get("/health")
def health():
    cached = model_loader._model_cache
    return {
        "status": "ok",
        "model_loaded": cached is not None,
        "model_version": cached.version if cached else None,
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
        "model_source": os.getenv("MODEL_SOURCE", "mlflow"),
        "traffic_db": storage.DB_PATH,
        "traffic_db_env": os.getenv("TRAFFIC_DB"),
        "mlflow_tracking_uri_env": os.getenv("MLFLOW_TRACKING_URI"),
        "last_observation": str(storage.last_observation_ts() or ""),
    }


@router.get("/hubs")
def list_hubs():
    cached = model_loader._model_cache
    in_model = set(cached.units) if cached else set()
    cfg = corridors(include_disabled=True)
    return [
        {"id": hid, "name": h["name"], "enabled": h.get("enabled", True), "center": h.get("center"),
         "event_types": h.get("event_types", []),
         "corridors": [{"id": cid, "name": c["name"], "short": c.get("short", c["name"]), "role": c.get("role"), "length_km": c["length_km"],
                        "n_links": len(c.get("links") or []), "in_model": cid in in_model}
                       for cid, c in cfg.items() if c["hub"] == hid]}
        for hid, h in hubs(include_disabled=True).items()
    ]


@router.get("/monitoring/drift")
def drift():
    try:
        model = model_loader.get_model()
    except ModelLoadError as e:
        raise HTTPException(503, str(e))
    return [drift_status(c, model) for c in model.units]


@router.post("/monitoring/drift/check")
def drift_check():
    """
    주기 점검(scripts/sync_collected.sh 가 매시 예측 기록 직후 호출): 구간마다 실제 예측 기록으로 드리프트를 판정하고
    드리프트면 aiops.log 에 [WARN] 을 남긴다. retrain.on_drift 가 retrain 이면 재학습·게이트까지 이어진다.
    과거 재현 판정은 쓰지 않으므로 예측 기록이 쌓이기 전에는 모두 판정 대기(insufficient_data)다.
    """
    from serving_app.monitoring.retrain_trigger import check_and_trigger

    try:
        model = model_loader.get_model()
    except ModelLoadError as e:
        raise HTTPException(503, str(e))
    results = []
    for corridor in model.units:
        r = check_and_trigger(corridor)
        d = r.get("drift", {})
        results.append({"corridor": corridor, "status": r["status"], "ratio": d.get("ratio"),
                        "reason": d.get("reason"), "n_hours": d.get("n_hours"),
                        "retrain": r.get("retrain"), "promoted": r.get("promoted")})
    return {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "model_version": model.version,
            "drift": [x["corridor"] for x in results if x["status"] in ("drift_detected", "retrain_triggered")],
            "results": results}


@router.get("/incidents")
def incidents(hub: str | None = None, start: dt.datetime | None = None, end: dt.datetime | None = None,
              limit: int = 100):
    df = storage.load_incidents(hubs=[hub] if hub else None, start=start, end=end).tail(limit)
    df = df.astype(object).where(pd.notna(df), None)
    for c in ("start", "expected_end", "last_seen"):
        df[c] = df[c].map(lambda v: v.strftime(storage.TS_FMT) if v is not None else None)
    return df.to_dict(orient="records")
