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

import pandas as pd

from data import storage
from data.config import hub_of, section
from serving_app import model_loader
from serving_app.monitoring import drift_state
from serving_app.monitoring.drift_detector import drift_status

logger = logging.getLogger("aiops")
_retrain_lock = threading.Lock()


def check_alert_only(corridor: str) -> dict:
    """
    운영용 관찰·판단 (POST /monitoring/drift/check, sync_collected.sh 가 호출). 재학습은 절대 실행하지 않습니다 -
    on_drift 값과 무관하게 [WARN] 경보와 조기 재학습 "신청"(drift_state)만 하고, 재학습은 별도 프로세스가 맡습니다.
    같은 corridor·같은 날의 같은 로그는 1회만 남깁니다.
    """
    model = model_loader.get_model()
    status = drift_status(corridor, model, replay_if_empty=False)  # 실제 예측 기록으로만 판단
    day = status.get("window_end")
    if status["status"] != "drift" or day is None:
        known, ref = status.get("rmse_known_event_hours"), status.get("reference_rmse")
        if (day is not None and known and ref and known > section("drift")["rmse_ratio_threshold"] * ref
                and drift_state.first_today("known_event", corridor, day, model.version)):
            logger.info(
                f"[INFO] corridor={corridor} large error during known event/holiday hours "
                f"(rmse={known:.2f}, {status['excluded_hours']}h) - excluded from drift judgment"
            )
        return {"status": status["status"], "drift": status, "early_retrain": None}

    rcfg = section("retrain")
    early = rcfg.get("early", {})
    early_on = bool(early.get("enabled")) and rcfg.get("on_drift", "retrain") == "alert"

    recent = []
    if early_on:
        now = pd.Timestamp(day)
        incidents = storage.load_incidents(hubs=[hub_of(corridor)], start=now - pd.Timedelta(days=early["recent_construction_days"]), end=now)
        recent = drift_state.recent_construction(incidents, corridor, now, early["recent_construction_days"])
    consecutive = drift_state.record_drift(corridor, day, model.version, bool(recent))

    if drift_state.first_today("drift", corridor, day, model.version):
        cause = f" cause: {'; '.join(status['constructions'])}" if status.get("constructions") else ""
        logger.warning(
            f"[WARN] drift detected corridor={corridor} rmse={status['rmse']:.2f} ref={status['reference_rmse']:.2f} "
            f"ratio={status['ratio']:.2f} (excluded event/holiday {status['excluded_hours']}h, "
            f"incident {status['excluded_incident_hours']}h){cause} - alert only"
        )

    requested = None
    if early_on:
        reason = drift_state.early_reason(consecutive, bool(recent), early)
        if reason:
            reason = f"{reason}, corridor={corridor}" + (f", {recent[0]}" if recent else "")
            if drift_state.write_request(corridor, reason):  # 이미 신청이 있으면 다시 쓰지 않음
                logger.info(f"[INFO] early retrain requested {reason}")
            requested = reason
    return {"status": "drift_detected", "drift": status, "consecutive_days": consecutive, "early_retrain": requested}


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
