"""
재학습을 지금 해야 하는지 판단 - scripts/periodic_retrain.sh 가 매시 부르는 가벼운 확인 (TensorFlow 를 불러오지 않음).

    python -m serving_app.monitoring.retrain_schedule

기준 시각 = MLflow 의 마지막 base-train / fine-tune run 시작 시각 (게이트 탈락·실패해도 포함 → 실패 후 반복 재시도 방지)

    periodic    : 기준 시각으로부터 retrain.period_days(14) 일이 지남
    drift-early : 감지 기반 재학습 신청(drift_state)이 있고, 기준 시각으로부터 early.cooldown_days(3) 일이 지남
                  (신청이 early.request_expire_days 일을 넘기면 만료되어 지움)

종료 코드  0: 실행해야 함 (stdout 첫 줄 "DUE mode=<mode> reason=<사유>")
           1: 아직 아님   (stdout "SKIP <사유>")
           2: 확인 불가   (Production 모델 없음 등)
RETRAIN_NOW=2026-10-20T09:00:00 로 "지금"을 바꿔 기간 판정을 시험할 수 있습니다.
"""
import datetime as dt
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # 매시 도는 확인이라 안내 로그를 끔

import mlflow
from mlflow.tracking import MlflowClient

from data.config import section
from serving_app.monitoring import drift_state

MODEL_NAME = "TrafficSpeedForecaster"  # train_and_register.MODEL_NAME 과 같아야 함 (TensorFlow import 를 피해 복제)
PRODUCTION_ALIAS = "production"
EXPERIMENT = "traffic-speed"


def _tracking_uri() -> str:
    # train_and_register._default_tracking_uri 와 같은 규칙
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    if os.path.exists("mlflow.db"):
        return "sqlite:///mlflow.db"
    if os.path.exists("mlflow_synthetic.db"):
        return "sqlite:///mlflow_synthetic.db"
    return "sqlite:///mlflow.db"


def last_training_time(client: MlflowClient) -> dt.datetime | None:
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return None
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=50)
    runs = [r for r in runs if r.info.run_name in ("base-train", "fine-tune")]  # 태그 IN 검색이 없어 직접 거름
    return dt.datetime.fromtimestamp(runs[0].info.start_time / 1000) if runs else None


def decide(now: dt.datetime | None = None) -> tuple[int, str]:
    now = now or (dt.datetime.fromisoformat(os.environ["RETRAIN_NOW"]) if os.getenv("RETRAIN_NOW") else dt.datetime.now())
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        return 2, f"no production model ({MODEL_NAME}@{PRODUCTION_ALIAS}) - base 학습을 먼저 하세요"
    last = last_training_time(client)
    if last is None:
        return 2, "no base-train/fine-tune run found"
    elapsed = (now - last).total_seconds() / 86400

    rcfg = section("retrain")
    early = rcfg.get("early", {})
    req = drift_state.read_request()
    if req:
        age = (now - dt.datetime.fromisoformat(req["created_at"])).total_seconds() / 86400
        if age > early.get("request_expire_days", 7):
            drift_state.clear_request()
            req = None

    if elapsed >= rcfg["period_days"]:
        return 0, f"DUE mode=periodic reason=period {elapsed:.1f}d >= {rcfg['period_days']}d since {last:%Y-%m-%d %H:%M}"
    if req and early.get("enabled") and rcfg.get("on_drift", "retrain") == "alert":
        if elapsed >= early["cooldown_days"]:
            return 0, f"DUE mode=drift-early reason={req['reason']}"
        return 1, f"SKIP early request pending, cooldown {elapsed:.1f}d < {early['cooldown_days']}d since {last:%Y-%m-%d %H:%M}"
    return 1, f"SKIP last training {last:%Y-%m-%d %H:%M} ({elapsed:.1f}d ago), period {rcfg['period_days']}d, no early request"


if __name__ == "__main__":
    try:
        code, msg = decide()
    except Exception as e:  # 확인 자체가 실패해도 "아직 아님(1)"과 구분되게 2 로 끝냄
        code, msg = 2, f"error: {e}"
    print(msg)
    sys.exit(code)
