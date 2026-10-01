"""
MLflow 없이 로컬 모델 만들기 (MODEL_SOURCE=local 서빙 확인용).

    python scripts/train_local.py
    MODEL_SOURCE=local uvicorn serving_app.main:app --port 8077

serving_app/models/model.keras + preprocess.json 을 만듭니다. 게이트·승격·재학습은
MLflow 경로(serving_app/train_and_register.py)에서만 동작합니다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.config import HORIZON, corridor_ids, section
from data.features import FeatureSpec
from serving_app import training as T
from serving_app.model_loader import LOCAL_MODEL_PATH, LOCAL_PREPROCESS_PATH


def main():
    cfg = section("train")
    obs, events = T.load_data()
    spec = FeatureSpec(corridor_ids(), cfg["feature_stage"], cfg["event_mode"])
    splits = T.time_splits(obs["ts"].max(), HORIZON)
    scaler, frames = T.prepare(spec, obs, events, splits)
    w = {k: T.windows_for(frames, spec, splits[k]) for k in ("train", "val", "test")}

    model = T.fit(spec, w["train"], w["val"], cfg["seeds"][0], cfg["epochs"], cfg["lr"], cfg["batch_size"])
    m = T.metrics(T.predict_speed(model, w["test"], scaler), w["test"])
    naive = T.naive_metrics(w["test"])
    m_val = T.metrics(T.predict_speed(model, w["val"], scaler), w["val"])
    print(f"[{spec.key}] test RMSE normal={m['rmse_normal']:.2f} event={m['rmse_event']:.2f} "
          f"(단순 예측법 normal={naive['rmse_normal']:.2f} event={naive['rmse_event']:.2f})")

    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    model.save(LOCAL_MODEL_PATH)
    with open(LOCAL_PREPROCESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"spec": spec.to_dict(), "scaler": scaler.to_dict(),
                   "reference_rmse": T.reference_rmse(m_val, spec.units),
                   "gate_normal_rmse": naive["rmse_normal"], "data_end": str(obs["ts"].max())},
                  f, ensure_ascii=False, indent=2)
    print(f"saved -> {LOCAL_MODEL_PATH}, {LOCAL_PREPROCESS_PATH}")


if __name__ == "__main__":
    main()
