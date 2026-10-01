"""
예측 API.

    POST /predict             : corridor(거점 도로 × 방향)별 향후 HORIZON 시간 예상 속도 + 평소 대비 추가 소요시간
    POST /predict/batch-test  : 드리프트 시뮬레이션 - 관측치 주입 → 슬라이딩 예측 → 드리프트 판정 → (재학습)

계산은 model_loader(예측)와 monitoring(드리프트·재학습)에 맡기고, 여기서는 요청을 연결만 합니다.
"""
import pandas as pd
from fastapi import APIRouter, HTTPException

from data import storage
from data.config import corridors, hubs
from data.features import make_windows
from serving_app import model_loader
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import (
    BatchPoint,
    BatchTestRequest,
    BatchTestResponse,
    CorridorForecast,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


def _log_forecast(fc: dict):
    storage.log_predictions([
        {"corridor": fc["corridor"], "issued_at": fc["issued_at"], "target_ts": p["target_ts"],
         "horizon": p["horizon"], "predicted": p["speed_kmh"], "model_version": fc["model_version"]}
        for p in fc["forecast"]
    ])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = model_loader.get_model()
    if req.corridor:
        targets = [req.corridor]
    elif req.hub:
        targets = [c for c in model.units if corridors(include_disabled=True).get(c, {}).get("hub") == req.hub]
        if not targets:
            raise HTTPException(404, f"모델이 학습한 corridor 가 없는 거점입니다: {req.hub}")
    else:
        targets = model.units
    names = hubs(include_disabled=True)
    out = []
    for corridor in targets:
        try:
            fc = model.forecast(corridor, req.issued_at)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except LookupError as e:
            raise HTTPException(409, str(e))
        if req.log:
            _log_forecast(fc)
        out.append(CorridorForecast(name=names.get(fc["hub"], {}).get("name", fc["hub"]), **fc))
    return PredictResponse(forecasts=out)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    받은 관측치(연속된 시간)를 저장하고, 각 시각 ts 에 대해 "ts-1시까지의 데이터"로 예측한
    값을 정답과 비교해 예측 기록에 쌓은 뒤 드리프트를 판정합니다.
    (발행 시각별로 공개된 일정만 쓰는 대신 학습과 같은 보수적 규칙을 적용합니다 - data/events.py)
    """
    model = model_loader.get_model()
    if req.corridor not in model.units:
        raise HTTPException(404, f"모델이 학습하지 않은 corridor 입니다: {req.corridor}")

    obs = pd.DataFrame([{"corridor": req.corridor, **o.model_dump()} for o in req.observations]).sort_values("ts")
    obs["ts"] = pd.to_datetime(obs["ts"]).dt.floor("h")
    obs.loc[obs["volume"] == 0, "volume"] = None
    storage.upsert_observations(obs, source="simulation")

    first, last = obs["ts"].min(), obs["ts"].max()
    H = model.spec.horizon
    frame = model.unit_frame(req.corridor, first - pd.Timedelta(weeks=8), last + pd.Timedelta(hours=H))
    w = make_windows({req.corridor: frame}, model.spec, issue_start=first - pd.Timedelta(hours=1),
                     issue_end=last - pd.Timedelta(hours=1), require_targets=False)
    if not len(w):
        raise HTTPException(409, "예측에 필요한 직전 관측치가 부족합니다")
    pred = model.predict_windows(w)

    rows, points = [], []
    for i, issued in enumerate(pd.to_datetime(w.issued_at)):
        for h in range(H):
            rows.append({"corridor": req.corridor, "issued_at": issued, "target_ts": issued + pd.Timedelta(hours=h + 1),
                         "horizon": h + 1, "predicted": pred[i, h], "model_version": model.version})
        actual = w.actual[i, 0]
        points.append(BatchPoint(ts=str(issued + pd.Timedelta(hours=1)), predicted=round(float(pred[i, 0]), 1),
                                 actual=None if pd.isna(actual) else round(float(actual), 1),
                                 known_event=bool(w.is_event[i, 0])))
    storage.log_predictions(rows)

    drift_check = check_and_trigger(req.corridor)
    return BatchTestResponse(corridor=req.corridor, model_version=model.version, points=points, drift_check=drift_check)
