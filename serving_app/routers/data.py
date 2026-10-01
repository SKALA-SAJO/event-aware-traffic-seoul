"""
관측치 업로드·현황.

    POST /data/upload : CSV (corridor, datetime, speed, volume) → observations 테이블에 upsert
    GET  /data/status : corridor 별 관측 기간·건수·교통량 결측 수

TOPIS 자료실 엑셀은 scripts/import_topis_excel.py 로(링크·지점 원천 → corridor 재집계),
실시간 API 는 scripts/collect_hourly.py 로 적재합니다. 이 엔드포인트는 이미 corridor 단위로 정리된
CSV 나 시뮬레이션 데이터를 넣을 때 씁니다. volume 0 은 결측으로 저장합니다.
"""
import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from data import storage
from data.config import corridors

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"corridor", "speed"}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(400, f"CSV 를 읽을 수 없습니다: {e}")

    ts_col = "datetime" if "datetime" in df.columns else "ts" if "ts" in df.columns else None
    if ts_col is None or not REQUIRED_COLUMNS.issubset(df.columns):
        raise HTTPException(400, "CSV에 corridor, datetime(또는 ts), speed 컬럼이 있어야 합니다 (volume 선택).")
    unknown = set(df["corridor"]) - set(corridors(include_disabled=True))
    if unknown:
        raise HTTPException(400, f"config/hubs.yaml 에 없는 corridor: {sorted(unknown)}")

    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce").dt.floor("h")
    if df["ts"].isna().any():
        raise HTTPException(400, f"시각을 해석할 수 없는 행이 {int(df['ts'].isna().sum())}개 있습니다.")
    if "volume" not in df.columns:
        df["volume"] = None
    df.loc[df["volume"] == 0, "volume"] = None
    df = df[(df["speed"] > 0) | df["speed"].isna()]
    n = storage.upsert_observations(df[["corridor", "ts", "speed", "volume"]], source="upload")
    return {"filename": file.filename, "rows": n, "corridors": sorted(df["corridor"].unique().tolist()),
            "start": str(df["ts"].min()), "end": str(df["ts"].max())}


@router.get("/status")
def status():
    summary = storage.observation_summary()
    lag = storage.volume_arrival_lag()
    return {
        "exists": bool(summary),
        "corridors": summary,
        "volume_arrival_lag_hours": None if lag.empty else {
            "median": float(lag["lag_hours"].median()), "p90": float(lag["lag_hours"].quantile(0.9)), "n": len(lag)},
    }
