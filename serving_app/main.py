"""
FastAPI 앱 진입점 - 이벤트 인지형 서울 거점 교통 예측 서비스.

    /predict, /predict/batch-test   예측 · 드리프트 시뮬레이션
    /events                         이벤트 일정 등록·조회
    /data/upload, /data/status      관측치 적재·현황
    /hubs, /health, /monitoring/drift
    /logs                           logs/aiops.log 조회 (드리프트 → 재학습 → 승격 이력)

"aiops" 로거를 logs/aiops.log 에 연결하고, API 라우터를 모두 등록한 뒤 마지막에 대시보드
(static/index.html)를 "/" 에 mount 합니다 (Starlette 는 등록 순서대로 라우트를 검사).
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.routers import data, events, health, logs, predict

_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(os.path.join(_LOG_DIR, "aiops.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
    yield


app = FastAPI(title="Seoul Event-aware Traffic Forecast", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(events.router)
app.include_router(data.router)
app.include_router(health.router)
app.include_router(logs.router)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
