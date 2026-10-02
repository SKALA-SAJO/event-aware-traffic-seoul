"""
학습·평가·배포 게이트 공용 로직 (MLflow 와 무관한 순수 함수들).

serving_app/train_and_register.py (MLflow 기록·승격), scripts/train_local.py (로컬 모델),
scripts/run_experiments.py (ablation) 가 모두 이 모듈을 씁니다. 실험에서 쓴 평가 방식과
배포 게이트의 평가 방식이 같아야 "실험에서 효과가 있었던 방법만 채택"이 의미가 있습니다.

평가 구분
    event  : 이벤트 영향 구간 (시작 전 ~ 종료 후, data/events.py event_mask)
    normal : 이벤트도 공휴일도 아닌 평상 시간대
    공휴일 시간은 normal 에서 빼고 따로 집계하지 않습니다 (예정된 이벤트이므로).
"""
import numpy as np
import pandas as pd
from tensorflow import keras

from data import storage
from data.config import section
from data.features import (
    FeatureSpec,
    TrafficScaler,
    Windows,
    build_frames,
    make_windows,
)
from serving_app.lstm_model import build_model

# ───────────────────────────────────── 데이터 ─────────────────────────────────────

# base 학습·실험에서 제외하는 관측 출처: 드리프트 시뮬레이션 주입분, 2023 이전 이력(코로나 구간 분석용)
DEFAULT_EXCLUDE = ("simulation", "topis_history")


def load_data(start=None, end=None, exclude_sources: tuple = DEFAULT_EXCLUDE):
    """기본적으로 시뮬레이션 주입 데이터·2023 이전 이력은 base 학습·실험에서 제외 (이벤트는 관측 출처와 무관)."""
    events_exclude = tuple(s for s in exclude_sources if s != "topis_history")
    return (storage.load_observations(start=start, end=end, exclude_sources=exclude_sources),
            storage.load_events(start=start, end=end, exclude_sources=events_exclude))


def data_source(exclude: tuple = ("simulation", "topis_history")) -> str:
    """관측치 출처 요약 - 합성 데이터로 낸 결과임을 리포트·MLflow 에 남기기 위함."""
    q = "SELECT DISTINCT source FROM observations" + (
        f" WHERE source NOT IN ({','.join('?' * len(exclude))})" if exclude else "")
    with storage.connect() as conn:
        rows = conn.execute(q, list(exclude)).fetchall()
    return ",".join(sorted(r[0] for r in rows if r[0]))


def time_splits(obs_end, horizon: int, cfg: dict | None = None) -> dict:
    """
    시간 순서를 지키는 train / val / test 분할 (발행 시각 기준).
    train 샘플의 정답 구간이 val 로 넘어가지 않도록 horizon 만큼 간격을 둡니다.
    """
    cfg = cfg or section("train")
    end = pd.Timestamp(obs_end)
    test_start = (end - pd.Timedelta(days=cfg["test_days"])).floor("D")
    val_start = test_start - pd.Timedelta(days=cfg["val_days"])
    gap = pd.Timedelta(hours=horizon)
    return {
        "train": (None, val_start - gap - pd.Timedelta(hours=1)),
        "val": (val_start, test_start - gap - pd.Timedelta(hours=1)),
        "test": (test_start, end),
        "scaler_end": val_start,
    }


def windows_for(frames, spec: FeatureSpec, rng: tuple) -> Windows:
    return make_windows(frames, spec, issue_start=rng[0], issue_end=rng[1])


# ───────────────────────────────────── 학습 ─────────────────────────────────────

def fit(
    spec: FeatureSpec,
    w_train: Windows,
    w_val: Windows | None,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int = 256,
    model: keras.Model | None = None,
    sample_weight: np.ndarray | None = None,
    verbose: int = 0,
) -> keras.Model:
    """model 을 넘기면 그 가중치에서 이어서 학습(warm start), 없으면 처음부터."""
    keras.utils.set_random_seed(seed)
    if model is None:
        model = build_model(spec, lr)
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    callbacks = []
    validation = None
    if w_val is not None and len(w_val):
        validation = ([w_val.X_past, w_val.X_fut], w_val.y)
        callbacks.append(keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True))
    model.fit(
        [w_train.X_past, w_train.X_fut], w_train.y,
        validation_data=validation, epochs=epochs, batch_size=batch_size,
        sample_weight=sample_weight, callbacks=callbacks, shuffle=True, verbose=verbose,
    )
    return model


def predict_speed(model: keras.Model, w: Windows, scaler: TrafficScaler) -> np.ndarray:
    """(n, H) km/h"""
    if not len(w):
        return np.zeros((0, w.y.shape[1]))
    z = model.predict([w.X_past, w.X_fut], batch_size=4096, verbose=0)
    return scaler.inverse_speed(w.units, z)


# ───────────────────────────────────── 평가 ─────────────────────────────────────

def _rmse(err):
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def _mae(err):
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def metrics(pred: np.ndarray, w: Windows) -> dict:
    """
    MAE·RMSE 를 전체 / 평상 / 이벤트 시간대, 거점별로 계산합니다 (모든 예보 시차 h=1..H 포함).
    단순 예측법과 같은 조건으로 비교하도록, 지난주 값이 없는 지점은 모든 모델에서 제외합니다.
    """
    valid = ~np.isnan(w.actual) & ~np.isnan(w.naive) & ~np.isnan(pred)
    err = pred - w.actual
    normal = valid & ~w.is_event & ~w.is_holiday
    event = valid & w.is_event
    H = w.y.shape[1]
    out = {
        "rmse_all": _rmse(err[valid]), "mae_all": _mae(err[valid]),
        "rmse_normal": _rmse(err[normal]), "mae_normal": _mae(err[normal]),
        "rmse_event": _rmse(err[event]), "mae_event": _mae(err[event]),
        "n_event_hours": int(round(event.sum() / H)),
        "n_normal_hours": int(round(normal.sum() / H)),
    }
    for h in range(H):
        out[f"rmse_normal_h{h + 1}"] = _rmse(err[:, h][normal[:, h]])
    for hub in np.unique(w.units):
        m = (w.units == hub)[:, None]
        out[f"rmse_normal_{hub}"] = _rmse(err[normal & m])
        out[f"rmse_event_{hub}"] = _rmse(err[event & m])
        out[f"mae_event_{hub}"] = _mae(err[event & m])
    return out


def naive_metrics(w: Windows) -> dict:
    """비교 기준: 단순 예측법 (지난주 같은 요일·같은 시각)."""
    return metrics(w.naive.copy(), w)


def reference_rmse(m: dict, hubs: list[str]) -> dict:
    """
    드리프트 판정 기준값: corridor 별 평상 시간대 RMSE.
    test 구간이 아니라 val 구간(학습 직후, test 직전)으로 계산합니다 - test 구간에 이미 변화(공사 등)가
    섞여 있으면 기준이 부풀어 그 corridor 의 드리프트를 둔감하게 잡기 때문입니다.
    """
    return {h: m.get(f"rmse_normal_{h}") for h in hubs if not np.isnan(m.get(f"rmse_normal_{h}", np.nan))}


# ───────────────────────────────────── 게이트 ─────────────────────────────────────

def check_gates(cand: dict, no_event_ref: dict, gate_normal_rmse: float, current: dict | None = None) -> dict:
    """
    배포 게이트
      ① 평상 시간대 RMSE ≤ 기준값
      ② 이벤트 시간대 RMSE < 이벤트 정보 없는 모델의 이벤트 시간대 RMSE
         (평가 구간의 이벤트 시간이 min_event_hours 미만이면 건너뜀)
      ③ (재학습 시) 평상 시간대 RMSE 가 현재 Production 보다 나빠지지 않을 것
    """
    min_event_hours = section("gates")["min_event_hours"]
    g1 = cand["rmse_normal"] <= gate_normal_rmse
    if cand["n_event_hours"] >= min_event_hours:
        g2 = cand["rmse_event"] < no_event_ref["rmse_event"]
    else:
        g2 = None
    g3 = None if current is None else cand["rmse_normal"] <= current["rmse_normal"]
    passed = bool(g1) and g2 is not False and g3 is not False
    return {
        "passed": passed,
        "gate1_normal": {"passed": bool(g1), "rmse": cand["rmse_normal"], "max": gate_normal_rmse},
        "gate2_event": {
            "passed": g2, "rmse": cand["rmse_event"], "no_event_rmse": no_event_ref["rmse_event"],
            "event_hours": cand["n_event_hours"], "skipped": g2 is None,
        },
        "gate3_not_worse": None if current is None else {
            "passed": bool(g3), "rmse": cand["rmse_normal"], "production_rmse": current["rmse_normal"],
        },
    }


def recency_weights(issued_at: np.ndarray, half_life_days: float) -> np.ndarray:
    t = pd.to_datetime(issued_at)
    age_days = (t.max() - t) / pd.Timedelta(days=1)
    return np.power(0.5, np.asarray(age_days, dtype="float64") / half_life_days).astype("float32")


def prepare(spec: FeatureSpec, obs, events, splits, scaler: TrafficScaler | None = None):
    """스케일러(train 구간으로 fit) + 프레임 + 분할별 윈도우."""
    if scaler is None:
        scaler = TrafficScaler().fit(obs[obs["ts"] < splits["scaler_end"]])
    frames = build_frames(obs, events, spec, scaler)
    return scaler, frames
