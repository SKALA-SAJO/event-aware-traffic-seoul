"""
드리프트 → 알림 → fine-tuning → 게이트 재검증 → 승격(또는 기존 버전 유지).

    [WARN] drift detected corridor=... - triggering retrain  (진행 중인 공사가 있으면 원인으로 함께 기록)
    [INFO] retrain triggered (mode=fine-tune, window=last_14_days, parent=v1)
    [OK]   new_rmse=... - production promoted: TrafficSpeedForecaster v2
    [FAIL] gate failed - keep production v1 (...)

retrain.on_drift 가 "alert" 이면 [WARN] 알림만 남기고 재학습은 주기적 fine-tuning
(train_and_register.py --fine-tune, retrain.period_days 마다)에 맡깁니다 - 실데이터 실험 ③ 근거.

로그는 "aiops" 로거 → logs/aiops.log (serving_app/main.py 에서 연결). 대시보드가 이 문장을
그대로 보여주므로 형식을 바꾸면 대시보드 표시도 함께 확인하세요.
"""
import logging
import threading

from data.config import section
from serving_app import model_loader
from serving_app.monitoring.drift_detector import drift_status

logger = logging.getLogger("aiops")
_retrain_lock = threading.Lock()


def check_and_trigger(corridor: str) -> dict:
    model = model_loader.get_model()
    status = drift_status(corridor, model, replay_if_empty=False)  # 재학습은 실제 예측 기록으로만 판단

    if status["status"] != "drift":
        known = status.get("rmse_known_event_hours")
        ref = status.get("reference_rmse")
        if known and ref and known > section("drift")["rmse_ratio_threshold"] * ref:
            logger.info(
                f"[INFO] corridor={corridor} large error during known event/holiday hours "
                f"(rmse={known:.2f}, {status['excluded_hours']}h) - excluded from drift judgment"
            )
        return {"status": status["status"], "drift": status}

    cause = f" cause: {'; '.join(status['constructions'])}" if status.get("constructions") else ""
    logger.warning(
        f"[WARN] drift detected corridor={corridor} rmse={status['rmse']:.2f} ref={status['reference_rmse']:.2f} "
        f"ratio={status['ratio']:.2f} (excluded event/holiday {status['excluded_hours']}h, "
        f"incident {status['excluded_incident_hours']}h){cause}"
        + (" - triggering retrain" if section("retrain").get("on_drift", "retrain") == "retrain" else " - alert only")
    )
    if section("retrain").get("on_drift", "retrain") != "retrain":
        return {"status": "drift_detected", "drift": status, "retrain": "alert only (retrain.on_drift=alert)"}
    if model.version == "local":
        logger.warning("[WARN] MODEL_SOURCE=local - MLflow Production 이 없어 재학습을 건너뜁니다")
        return {"status": "drift_detected", "drift": status, "retrain": "skipped (MODEL_SOURCE=local)"}

    if not _retrain_lock.acquire(blocking=False):
        return {"status": "drift_detected", "drift": status, "retrain": "already running"}
    try:
        from serving_app.train_and_register import MODEL_NAME, fine_tune

        days = section("retrain")["fine_tune_days"]
        logger.info(f"[INFO] retrain triggered (mode=fine-tune, window=last_{days}_days, parent={model.version})")
        result = fine_tune(reason=f"drift corridor={corridor} ratio={status['ratio']:.2f}")
        g = result["gates"]
        if result["promoted"]:
            logger.info(
                f"[OK] new_rmse={result['rmse_normal']:.2f} (production was {result['production_rmse_normal']:.2f}) "
                f"- production promoted: {MODEL_NAME} v{result['version']}"
            )
            model_loader.invalidate()
        else:
            failed = [k for k in ("gate1_normal", "gate2_event", "gate3_not_worse") if g.get(k) and g[k]["passed"] is False]
            logger.warning(f"[FAIL] gate failed {failed} - keep production {model.version}")
        return {
            "status": "retrain_triggered",
            "drift": status,
            "promoted": result["promoted"],
            "new_version": result.get("version"),
            "rmse_normal": result["rmse_normal"],
            "production_rmse_normal": result["production_rmse_normal"],
            "gates": g,
        }
    finally:
        _retrain_lock.release()
