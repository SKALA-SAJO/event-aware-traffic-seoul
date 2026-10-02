"""
MLflow: 학습 → 기록(Tracking) → 배포 게이트 → 등록(Registry) → Production 승격,
그리고 드리프트 감지 시 Production 가중치에서 이어서 학습하는 fine-tuning.

base 학습 (train_and_register)
    1) 후보 모델 (설정의 feature_stage / event_mode, 기본 full + text)
    2) 비교 모델 - 같은 데이터·시드의 "이벤트 정보 없는" 모델 (calendar 단계)  → 게이트 ②
    3) 단순 예측법 (지난주 같은 시각)                                     → 게이트 ① 기본 기준값
    게이트를 통과하면 후보를 등록하고 alias "production" 을 옮깁니다. 비교 모델과 전처리 정보
    (FeatureSpec, 스케일러, 드리프트 기준 RMSE, 게이트 기준값)는 같은 run 의 아티팩트로 남겨
    fine-tuning 때 그대로 이어받습니다.

fine-tuning (fine_tune)
    최근 retrain.fine_tune_days 일의 데이터로 Production 모델과 비교 모델을 각각 이어서 학습
    (최근일수록 큰 가중치), 마지막 eval_days 일로 게이트 ①②③ 재검증 → 통과 시 승격,
    실패 시 기존 Production 유지. 처음부터 다시 학습하지 않는 이유는 HAIC 실습과 같습니다 -
    최근 2주로는 스크래치 학습이 불안정하고, 이미 학습된 일반적인 패턴을 버릴 이유가 없습니다.

실행:
    python scripts/generate_synthetic_data.py   # 또는 실데이터 적재
    python serving_app/train_and_register.py
    python serving_app/train_and_register.py --fine-tune   # 재학습 (scripts/periodic_retrain.sh 가 period_days 경과·조기 재학습 신청 시 호출)
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from data.config import HORIZON, corridor_ids, section
from data.features import FeatureSpec, TrafficScaler, build_frames, make_windows
from serving_app import training as T

MODEL_NAME = "TrafficSpeedForecaster"
PRODUCTION_ALIAS = "production"


def _default_tracking_uri() -> str:
    # 환경변수가 없으면 "존재하는 파일"을 우선 사용합니다.
    # - 실데이터/운영: mlflow.db
    # - 빠른 시작(합성 데이터): mlflow_synthetic.db
    if os.path.exists("mlflow.db"):
        return "sqlite:///mlflow.db"
    if os.path.exists("mlflow_synthetic.db"):
        return "sqlite:///mlflow_synthetic.db"
    return "sqlite:///mlflow.db"


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or _default_tracking_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("traffic-speed")


def _flat(prefix: str, m: dict) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in m.items() if not np.isnan(v)}  # NaN 제외


def _log_and_register(model, ref_model, preprocess: dict, gates: dict, run_id: str) -> dict:
    # MLflow 3: 모델은 run 아티팩트가 아니라 LoggedModel 로 저장되므로 반환된 URI 로 참조한다
    model_uri = mlflow.tensorflow.log_model(model, name="model").model_uri
    ref_uri = mlflow.tensorflow.log_model(ref_model, name="reference_model").model_uri
    preprocess = {**preprocess, "reference_model_uri": ref_uri}
    mlflow.log_dict(preprocess, "preprocess.json")
    mlflow.log_dict(gates, "gates.json")

    result = {"run_id": run_id, "promoted": False, "gates": gates,
              "rmse_normal": gates["gate1_normal"]["rmse"], "rmse_event": gates["gate2_event"]["rmse"]}
    if gates["passed"]:
        v = mlflow.register_model(model_uri, MODEL_NAME)
        MlflowClient().set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, v.version)
        result.update(promoted=True, version=v.version)
        print(f"[GATE PASSED] normal_rmse={result['rmse_normal']:.2f} event_rmse={result['rmse_event']:.2f} "
              f"-> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(f"[GATE FAILED] {json.dumps(gates, ensure_ascii=False, default=str)} -> 배포 차단, 기존 Production 유지")
    return result


def train_and_register(seed: int | None = None, obs=None, events=None) -> dict:
    cfg = section("train")
    seed = seed if seed is not None else cfg["seeds"][0]
    if obs is None:
        obs, events = T.load_data()
    hubs = corridor_ids()
    spec = FeatureSpec(hubs, cfg["feature_stage"], cfg["event_mode"])
    ref_spec = FeatureSpec(hubs, "calendar")
    splits = T.time_splits(obs["ts"].max(), HORIZON)

    scaler, frames = T.prepare(spec, obs, events, splits)
    w = {k: T.windows_for(frames, spec, splits[k]) for k in ("train", "val", "test")}
    wr = {k: T.windows_for(frames, ref_spec, splits[k]) for k in ("train", "val", "test")}

    with mlflow.start_run(run_name="base-train") as run:
        model = T.fit(spec, w["train"], w["val"], seed, cfg["epochs"], cfg["lr"], cfg["batch_size"])
        ref_model = T.fit(ref_spec, wr["train"], wr["val"], seed, cfg["epochs"], cfg["lr"], cfg["batch_size"])

        m = T.metrics(T.predict_speed(model, w["test"], scaler), w["test"])
        m_ref = T.metrics(T.predict_speed(ref_model, wr["test"], scaler), wr["test"])
        m_naive = T.naive_metrics(w["test"])
        m_val = T.metrics(T.predict_speed(model, w["val"], scaler), w["val"])
        gate_normal = section("gates")["normal_rmse_max"] or m_naive["rmse_normal"]
        gates = T.check_gates(m, m_ref, gate_normal)

        mlflow.log_params({
            "mode": "scratch", "seed": seed, "feature_stage": spec.stage, "event_mode": spec.event_mode,
            "lookback": spec.lookback, "horizon": spec.horizon, "epochs": cfg["epochs"], "corridors": ",".join(hubs),
            "data_source": T.data_source(), "data_end": str(obs["ts"].max()), "test_start": str(splits["test"][0]),
        })
        mlflow.log_metrics({**_flat("cand", m), **_flat("noevent", m_ref), **_flat("naive", m_naive),
                            "gate_normal_rmse": gate_normal})
        preprocess = {
            "spec": spec.to_dict(), "ref_spec": ref_spec.to_dict(), "scaler": scaler.to_dict(),
            "reference_rmse": T.reference_rmse(m_val, hubs), "gate_normal_rmse": gate_normal,
            "data_end": str(obs["ts"].max()), "data_source": T.data_source(),
        }
        return _log_and_register(model, ref_model, preprocess, gates, run.info.run_id)


def load_production() -> dict:
    """현재 Production 버전의 모델·비교 모델·전처리 정보."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    preprocess = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/preprocess.json")
    return {
        "version": mv.version,
        "run_id": mv.run_id,
        "model": mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"),
        "ref_model": mlflow.tensorflow.load_model(preprocess["reference_model_uri"]),
        "preprocess": preprocess,
    }


def retrain_window(rcfg: dict, db_path: str | None = None):
    """
    재학습용 데이터의 출처·구간. 설정의 exclude_sources 를 뺀 확정 실측만 쓰고(시뮬레이션 주입분은 드리프트 신호로만 씀),
    데이터 끝 시각도 같은 기준으로 잡아야 학습·평가 구간이 시뮬레이션 때문에 비지 않는다.
    반환: (제외 출처, 끝 시각, 학습 시작, 평가 시작)
    """
    from data import storage

    exclude = tuple(rcfg.get("exclude_sources") or ())
    end = storage.last_observation_ts(exclude_sources=exclude, db_path=db_path)
    if end is None:
        raise ValueError(f"재학습에 쓸 관측치가 없습니다 (제외 출처: {exclude or '-'})")
    ft_start = end - pd.Timedelta(days=rcfg["fine_tune_days"])
    eval_start = end - pd.Timedelta(days=rcfg["eval_days"]) - pd.Timedelta(hours=HORIZON)
    return exclude, end, ft_start, eval_start


def fine_tune(reason: str = "", seed: int | None = None) -> dict:
    """Production 가중치에서 이어서 최근 데이터로 학습 → 게이트 재검증 → 통과 시 승격."""
    rcfg = section("retrain")
    seed = seed if seed is not None else section("train")["seeds"][0]
    prod = load_production()
    pre = prod["preprocess"]
    spec, ref_spec = FeatureSpec.from_dict(pre["spec"]), FeatureSpec.from_dict(pre["ref_spec"])
    scaler = TrafficScaler.from_dict(pre["scaler"])

    exclude, end, ft_start, eval_start = retrain_window(rcfg)
    hist_start = ft_start - pd.Timedelta(hours=spec.lookback + 24 * 7 * 8)  # 입력 윈도우 + 결측 보정용 이력
    obs, events = T.load_data(start=hist_start, exclude_sources=exclude)
    frames = build_frames(obs, events, spec, scaler)

    def win(s, e, sp):
        return make_windows(frames, sp, issue_start=s, issue_end=e)

    train_end = eval_start - pd.Timedelta(hours=HORIZON + 1)
    w_tr, w_ev = win(ft_start, train_end, spec), win(eval_start, None, spec)
    wr_tr, wr_ev = win(ft_start, train_end, ref_spec), win(eval_start, None, ref_spec)
    weights = T.recency_weights(w_tr.issued_at, rcfg["recency_half_life_days"])
    logging.getLogger("aiops").info(
        f"[INFO] retrain data: sources={T.data_source(exclude) or '-'} excluded={','.join(exclude) or '-'} "
        f"range={ft_start:%Y-%m-%d %H:%M}~{end:%Y-%m-%d %H:%M} n_train={len(w_tr)} n_eval={len(w_ev)}")

    m_prod = T.metrics(T.predict_speed(prod["model"], w_ev, scaler), w_ev)
    with mlflow.start_run(run_name="fine-tune") as run:
        model = T.fit(spec, w_tr, None, seed, rcfg["epochs"], rcfg["lr"], model=prod["model"], sample_weight=weights)
        ref_model = T.fit(ref_spec, wr_tr, None, seed, rcfg["epochs"], rcfg["lr"], model=prod["ref_model"], sample_weight=weights)
        m = T.metrics(T.predict_speed(model, w_ev, scaler), w_ev)
        m_ref = T.metrics(T.predict_speed(ref_model, wr_ev, scaler), wr_ev)
        gates = T.check_gates(m, m_ref, pre["gate_normal_rmse"], current=m_prod)

        mlflow.log_params({
            "mode": "fine-tune", "parent_version": prod["version"], "reason": reason[:250],
            "epochs": rcfg["epochs"], "lr": rcfg["lr"], "fine_tune_days": rcfg["fine_tune_days"],
            "n_train": len(w_tr), "n_eval": len(w_ev), "data_end": str(end),
            "exclude_sources": ",".join(exclude) or "-", "train_sources": T.data_source(exclude) or "-",
        })
        mlflow.log_metrics({**_flat("cand", m), **_flat("noevent", m_ref), **_flat("production", m_prod)})
        preprocess = {**pre, "data_end": str(end), "parent_version": prod["version"]}
        result = _log_and_register(model, ref_model, preprocess, gates, run.info.run_id)
        result["data_end"], result["n_train"], result["n_eval"] = str(end), len(w_tr), len(w_ev)
        result["exclude_sources"] = list(exclude)
        result["production_rmse_normal"] = m_prod["rmse_normal"]
        result["parent_version"] = prod["version"]
        return result


def _attach_aiops_log() -> logging.Logger:
    """CLI 프로세스에서도 서버(main.py)와 같은 logs/aiops.log 에 기록 - 대시보드 "재학습 로그"에 보이게."""
    logger = logging.getLogger("aiops")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        handler = logging.FileHandler(os.path.join("logs", "aiops.log"), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else default


if __name__ == "__main__":
    if "--fine-tune" in sys.argv:
        # --mode periodic | drift-early (scripts/periodic_retrain.sh 가 retrain_schedule 결과를 넘김), --reason 사유
        logger = _attach_aiops_log()
        mode, reason = _arg("--mode", "periodic"), _arg("--reason", "")
        parent = load_production()["version"]
        logger.info(f"[INFO] retrain triggered (mode={mode}, window=last_{section('retrain')['fine_tune_days']}_days, parent=v{parent})")
        try:
            r = fine_tune(reason=f"{mode}: {reason}".strip(": "))
        except Exception as e:
            logger.error(f"[FAIL] fine-tune error: {e} - keep production v{parent}")
            raise
        finally:
            from serving_app.monitoring.drift_state import clear_request
            clear_request()  # 실행했으면 신청은 처리된 것 (실패해도 쿨다운이 반복 재시도를 막음)
        if r["promoted"]:
            logger.info(f"[OK] new_rmse={r['rmse_normal']:.2f} (production was {r['production_rmse_normal']:.2f}) "
                        f"- production promoted: {MODEL_NAME} v{r['version']}")
        else:
            g = r["gates"]
            failed = [k for k in ("gate1_normal", "gate2_event", "gate3_not_worse") if g.get(k) and g[k]["passed"] is False]
            logger.warning(f"[FAIL] gate failed {failed} - keep production v{parent}")
        print(f"[{'OK' if r['promoted'] else 'FAIL'}] {mode} fine-tune rmse_normal={r['rmse_normal']:.2f} "
              f"(production {r['production_rmse_normal']:.2f}) promoted={r['promoted']} version={r.get('version')}")
    else:
        train_and_register()
