"""헬스체크 · 거점/corridor 목록 · 드리프트 현황 · 돌발 정보."""
import datetime as dt
import os

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from data import storage
from data.config import corridors, hubs, section
from serving_app.errors import ModelLoadError
from serving_app import model_loader
from serving_app.monitoring.drift_detector import drift_status
from serving_app.monitoring.retrain_trigger import check_alert_only, check_and_trigger

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
    드리프트면 aiops.log 에 [WARN] 을 남긴다. 과거 재현 판정은 쓰지 않으므로 예측 기록이 쌓이기 전에는 모두 판정 대기
    (insufficient_data)다.

    retrain.on_drift 가 retrain(시연용)이면 기존대로 재학습·게이트까지 이어진다. 그 외(alert, 기본)에는 재학습을 실행하지
    않고 경보(같은 구간·날 1회)와 일별 감지 기록만 남긴다. 조기 재학습은 retrain.early.enabled 일 때만 신청하고,
    실행은 서버가 아니라 scripts/periodic_retrain.sh 가 별도 프로세스로 한다.
    """
    try:
        model = model_loader.get_model()
    except ModelLoadError as e:
        raise HTTPException(503, str(e))
    check = check_and_trigger if section("retrain").get("on_drift", "retrain") == "retrain" else check_alert_only
    results = []
    for corridor in model.units:
        r = check(corridor)
        d = r.get("drift", {})
        results.append({"corridor": corridor, "status": r["status"], "ratio": d.get("ratio"),
                        "reason": d.get("reason"), "n_hours": d.get("n_hours"),
                        "retrain": r.get("retrain"), "promoted": r.get("promoted"),
                        "consecutive_days": r.get("consecutive_days"), "early_retrain": r.get("early_retrain")})
    return {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "model_version": model.version,
            "drift": [x["corridor"] for x in results if x["status"] in ("drift_detected", "retrain_triggered")],
            "results": results}


@router.post("/admin/reload")
def admin_reload(request: Request):
    """재학습으로 Production 이 바뀐 뒤 모델 캐시를 비웁니다 (scripts/periodic_retrain.sh 가 호출). 같은 PC(localhost)에서만."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "localhost 에서만 호출할 수 있습니다")
    before = model_loader._model_cache.version if model_loader._model_cache else None
    model_loader.invalidate()
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()  # 첫 요청 지연을 없애려고 바로 다시 불러옴
    cached = model_loader._model_cache
    return {"previous_version": before, "model_loaded": cached is not None, "model_version": cached.version if cached else None}


@router.get("/incidents")
def incidents(hub: str | None = None, start: dt.datetime | None = None, end: dt.datetime | None = None,
              limit: int = 100):
    df = storage.load_incidents(hubs=[hub] if hub else None, start=start, end=end).tail(limit)
    df = df.astype(object).where(pd.notna(df), None)
    for c in ("start", "expected_end", "last_seen"):
        df[c] = df[c].map(lambda v: v.strftime(storage.TS_FMT) if v is not None else None)
    return df.to_dict(orient="records")
