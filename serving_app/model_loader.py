"""
서빙용 모델 로딩 + 예측.

다른 모듈(routers, monitoring)은 get_model() 만 부르면 되고, 모델이 로컬 파일에서 왔는지
MLflow Production 에서 왔는지 알 필요가 없습니다.

■ 핵심 원칙 (HAIC 실습과 동일)
   1) 학습 때와 똑같이 전처리한다 - 입력은 data/features.py 의 같은 함수로 만들고,
      스케일러·FeatureSpec 은 모델과 함께 저장된 것을 그대로 쓴다 (새로 fit 하지 않음)
   2) Lazy(기본) / Eager 로딩
   3) 재학습으로 Production 이 바뀌면 invalidate() 로 캐시를 비워 다음 요청에서 새 버전을 읽는다

■ 환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local | mlflow(기본)
"""
import json
import os
import threading
import time

import numpy as np
import pandas as pd

from data import storage
from data.config import corridors, hub_of
from data.features import FeatureSpec, TrafficScaler, build_frame, make_windows, usual_speed
from serving_app.errors import ModelLoadError

LOCAL_MODEL_PATH = "serving_app/models/model.keras"
LOCAL_PREPROCESS_PATH = "serving_app/models/preprocess.json"

_model_cache = None
_lock = threading.Lock()


class LoadedModel:
    """keras 모델 + 전처리 정보(FeatureSpec, 스케일러, 드리프트 기준값) + 버전."""

    def __init__(self, keras_model, preprocess: dict, version: str):
        self._keras_model = keras_model
        self.preprocess = preprocess
        self.spec = FeatureSpec.from_dict(preprocess["spec"])
        self.scaler = TrafficScaler.from_dict(preprocess["scaler"])
        self.reference_rmse: dict = preprocess.get("reference_rmse", {})
        self.version = version

    @property
    def units(self) -> list[str]:
        """모델이 학습한 corridor 목록"""
        return self.spec.units

    def predict_windows(self, w) -> np.ndarray:
        """(n, H) km/h"""
        if not len(w):
            return np.zeros((0, self.spec.horizon))
        z = self._keras_model.predict([w.X_past, w.X_fut], batch_size=2048, verbose=0)
        return self.scaler.inverse_speed(w.units, z)

    def unit_frame(self, corridor: str, start, end, known_cutoff=None) -> pd.DataFrame:
        obs = storage.load_observations(corridors=[corridor], start=start, end=end)
        if obs.empty:
            raise LookupError(f"{corridor}: {start} ~ {end} 구간의 관측치가 없습니다")
        events = storage.load_events(hubs=[hub_of(corridor)], start=start, end=end)
        return build_frame(corridor, obs, events, self.spec, self.scaler, start=start, end=end,
                           known_cutoff=known_cutoff)

    def forecast(self, corridor: str, issued_at=None, history_weeks: int = 8) -> dict:
        """
        발행 시각(issued_at, 기본은 이 corridor 의 마지막 관측 시각)까지의 관측과, 그 시각까지 공개된
        (그리고 그 시각까지 취소가 공지되지 않은) 이벤트 일정으로 다음 HORIZON 시간의 속도를 예측합니다.
        """
        if corridor not in self.units:
            raise KeyError(f"모델이 학습하지 않은 corridor 입니다: {corridor} (학습 대상: {self.units})")
        last_obs = storage.last_observation_ts(corridor)
        issued_at = pd.Timestamp(issued_at).floor("h") if issued_at else last_obs
        if issued_at is None:
            raise LookupError(f"{corridor}: 관측치가 없습니다")
        if last_obs is None:
            raise LookupError(f"{corridor}: 관측치가 없습니다")
        if issued_at > last_obs:
            raise LookupError(
                f"{corridor}: issued_at({issued_at.strftime(storage.TS_FMT)})이 마지막 관측치(" \
                f"{last_obs.strftime(storage.TS_FMT)}) 이후라 예측할 수 없습니다"
            )
        hub = hub_of(corridor)
        H = self.spec.horizon
        start = issued_at - pd.Timedelta(weeks=history_weeks)

        frame = self.unit_frame(corridor, start, issued_at + pd.Timedelta(hours=H), known_cutoff=issued_at)
        w = make_windows({corridor: frame}, self.spec, issue_start=issued_at, issue_end=issued_at,
                         require_targets=False)
        if not len(w):
            raise LookupError(f"{corridor}: {issued_at} 직전 {self.spec.lookback}시간 관측치가 부족합니다")
        pred = self.predict_windows(w)[0]

        usual = usual_speed(frame, issued_at, weeks=history_weeks)
        cfg = corridors(include_disabled=True)[corridor]
        length = cfg["length_km"]
        targets = pd.date_range(issued_at + pd.Timedelta(hours=1), periods=H, freq="h")
        events = storage.load_events(hubs=[hub], start=targets[0] - pd.Timedelta(hours=3), end=targets[-1])
        known = events["announced_at"].isna() | (events["announced_at"] <= issued_at)
        cancelled = (events["status"] == "cancelled") & (events["status_changed_at"] <= issued_at)
        events = events[known & ~cancelled & (events["status"] != "review")]
        # 과거 시점 재현일 때는 실측도 함께 (운영 중 미래 시각은 None)
        obs = storage.load_observations(corridors=[corridor], start=targets[0], end=targets[-1])
        actual = dict(zip(obs["ts"], obs["speed"]))
        rows = []
        for i, ts in enumerate(targets):
            speed = float(max(pred[i], 1.0))
            base = usual.get((ts.dayofweek, ts.hour))
            tt = length / speed * 60
            base_tt = length / base * 60 if base else None
            active = events[(events["start"] - pd.Timedelta(hours=3) <= ts) & (events["end"] + pd.Timedelta(hours=3) >= ts)]
            rows.append({
                "target_ts": ts.strftime(storage.TS_FMT),
                "horizon": i + 1,
                "speed_kmh": round(speed, 1),
                "actual_speed_kmh": round(float(actual[ts]), 1) if pd.notna(actual.get(ts)) else None,
                "usual_speed_kmh": round(base, 1) if base else None,
                "travel_time_min": round(tt, 1),
                "usual_travel_time_min": round(base_tt, 1) if base_tt else None,
                "extra_min": round(tt - base_tt, 1) if base_tt else None,
                "events": active["title"].tolist(),
            })
        return {
            "corridor": corridor,
            "hub": hub,
            "corridor_name": cfg["name"],
            "issued_at": issued_at.strftime(storage.TS_FMT),
            "segment_length_km": length,
            "model_version": self.version,
            "forecast": rows,
        }


# ═══════════════════════════════ 어디서 불러올까? ═══════════════════════════════

def _load_from_local() -> LoadedModel:
    """scripts/train_local.py 가 저장한 모델 (MLflow 없이 서빙 확인용)."""
    from tensorflow import keras

    with open(LOCAL_PREPROCESS_PATH, encoding="utf-8") as f:
        preprocess = json.load(f)
    return LoadedModel(keras.models.load_model(LOCAL_MODEL_PATH), preprocess, version="local")


def _load_from_mlflow() -> LoadedModel:
    """MLflow Registry 에서 alias "production" 이 가리키는 버전 (재배포 시 서버 코드 수정 불필요)."""
    from serving_app.train_and_register import load_production

    try:
        prod = load_production()
    except Exception as e:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        hint = (
            "MLflow Production 모델을 찾을 수 없습니다. "
            "먼저 학습/등록을 수행하세요: `python serving_app/train_and_register.py`. "
            "합성 데이터라면 `export TRAFFIC_DB=data/traffic_synthetic.db MLFLOW_TRACKING_URI=sqlite:///mlflow_synthetic.db` "
            "설정도 확인하세요."
        )
        if uri:
            hint += f" (현재 MLFLOW_TRACKING_URI={uri})"
        raise ModelLoadError(hint) from e
    return LoadedModel(prod["model"], prod["preprocess"], version=f"v{prod['version']}")


def _load_model() -> LoadedModel:
    if os.getenv("MODEL_SOURCE", "mlflow") == "local":
        try:
            return _load_from_local()
        except Exception as e:
            raise ModelLoadError(
                "로컬 모델을 로드하지 못했습니다. `scripts/train_local.py`로 모델을 생성했는지, "
                f"파일이 존재하는지 확인하세요: {LOCAL_MODEL_PATH}, {LOCAL_PREPROCESS_PATH}"
            ) from e
    return _load_from_mlflow()


# ═══════════════════════════ 언제 불러올까? (Eager / Lazy) ═══════════════════════════

def load_eager() -> LoadedModel:
    start = time.time()
    model = get_model()
    print(f"[eager] model {model.version} loaded in {time.time() - start:.3f}s at startup")
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청 때 한 번만 불러오고 이후에는 캐시를 재사용."""
    global _model_cache
    if _model_cache is None:
        with _lock:
            if _model_cache is None:
                start = time.time()
                _model_cache = _load_model()
                print(f"[lazy] model {_model_cache.version} loaded in {time.time() - start:.3f}s")
    return _model_cache


def invalidate():
    """재학습으로 Production 이 바뀐 뒤 호출 - 다음 요청에서 새 버전을 불러옵니다."""
    global _model_cache
    with _lock:
        _model_cache = None
