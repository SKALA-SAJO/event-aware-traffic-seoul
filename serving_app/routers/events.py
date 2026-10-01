"""
이벤트 일정 등록·조회·삭제.

    POST   /events          : 일정 1건 등록(또는 같은 hub·type·title·start 갱신)
    POST   /events/upload   : CSV 일괄 등록 (hub,type,title,start,end,expected_size,description,announced_at)
    GET    /events          : ?hub=&start=&end=&status=
    PATCH  /events/{id}/status : 검토 대기 일정 확정(scheduled) / 취소(cancelled, 공지 시각 기록)
    DELETE /events/{id}

자동 수집(KOPIS·서울시 문화행사)은 시각을 확정하지 못한 회차를 status=review 로 넣습니다.
review 일정은 예측에 쓰이지 않으므로, 대시보드에서 확인 후 확정해야 반영됩니다.
취소는 삭제하지 말고 cancelled 로 바꾸세요 - 취소 공지 이전의 예측·학습에서는 "예정"이었다는
사실이 남아 있어야 학습과 운영이 같은 정보로 동작합니다.

등록 시 description 에서 행진·차로 통제 여부를 추출해 함께 저장합니다(data/event_nlp.py).
announced_at 을 생략하면 "지금 공개됨"으로 기록합니다 - 예측은 발행 시각까지 공개된 일정만
쓰므로, 과거 일정을 소급 등록할 때는 실제 공개 시각을 넣어야 정보 누수가 없습니다.
"""
import csv
import datetime as dt
import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import ValidationError

from data import storage
from data.config import hubs as hub_config
from data.event_nlp import extract
from serving_app.schemas import EventIn, EventOut, EventStatusIn

router = APIRouter(prefix="/events")


def _register(events: list[EventIn]) -> list[dict]:
    known = hub_config(include_disabled=True)
    records = []
    for ev in events:
        if ev.hub not in known:
            raise HTTPException(400, f"알 수 없는 거점: {ev.hub} (config/hubs.yaml 에 먼저 추가하세요)")
        flags = extract(ev.description)
        records.append({**ev.model_dump(), "announced_at": ev.announced_at or dt.datetime.now(),
                        "march": flags["march"], "lane_control": flags["lane_control"], "status": "scheduled"})
    ids = storage.upsert_events(records)
    return [{**r, "id": i} for r, i in zip(records, ids)]


@router.post("", response_model=EventOut)
def create_event(ev: EventIn):
    return _register([ev])[0]


@router.post("/upload")
async def upload_events(file: UploadFile = File(...)):
    try:
        text = (await file.read()).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8 CSV 파일만 업로드할 수 있습니다.")
    events, errors = [], []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        row = {k: (v if v not in ("", None) else None) for k, v in row.items() if k}
        try:
            events.append(EventIn(**{k: v for k, v in row.items() if k in EventIn.model_fields}))
        except ValidationError as e:
            errors.append({"line": i, "error": e.errors()[0]["msg"]})
    if errors:
        raise HTTPException(400, {"message": "잘못된 행이 있습니다", "errors": errors[:20]})
    saved = _register(events)
    return {"registered": len(saved)}


@router.get("")
def list_events(hub: str | None = None, start: dt.datetime | None = None, end: dt.datetime | None = None,
                status: str | None = None, limit: int = 200):
    df = storage.load_events(hubs=[hub] if hub else None, start=start, end=end,
                             statuses=(status,) if status else None)
    df = df.tail(limit)
    df = df.astype(object).where(pd.notna(df), None)
    for c in ("start", "end", "announced_at", "status_changed_at"):
        df[c] = df[c].map(lambda v: v.strftime(storage.TS_FMT) if v is not None else None)
    # pandas 3 는 문자열 열의 None 을 NaN 으로 바꾸므로 JSON 직전에 다시 None 으로
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in df.to_dict(orient="records")]


@router.patch("/{event_id}/status")
def update_status(event_id: int, body: EventStatusIn):
    if not storage.set_event_status(event_id, body.status, body.changed_at):
        raise HTTPException(404, "이벤트를 찾을 수 없습니다")
    return {"id": event_id, "status": body.status}


@router.delete("/{event_id}")
def delete_event(event_id: int):
    if not storage.delete_event(event_id):
        raise HTTPException(404, "이벤트를 찾을 수 없습니다")
    return {"deleted": event_id}
