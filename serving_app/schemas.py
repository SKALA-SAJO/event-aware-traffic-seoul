"""
FastAPI 요청/응답 스키마.

서빙 입력 검증이 학습 시점 데이터 정의(data/storage.py, data/events.py)와 어긋나지 않도록
값 범위·이벤트 유형을 스키마 단에서 강제합니다.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EventType = Literal["rally", "concert", "sports", "festival", "marathon", "other"]


class PredictRequest(BaseModel):
    hub: str | None = Field(None, description="거점 ID - 그 거점의 모든 방향(corridor)을 예측")
    corridor: str | None = Field(None, description="corridor ID (거점 도로 × 방향). hub·corridor 모두 생략 시 전체")
    issued_at: datetime | None = Field(None, description="예측 발행 시각 (생략 시 거점별 마지막 관측 시각)")
    log: bool = Field(True, description="예측 기록을 저장해 드리프트 판정에 사용")


class ForecastPoint(BaseModel):
    target_ts: str
    horizon: int
    speed_kmh: float
    actual_speed_kmh: float | None = Field(None, description="실측 속도 (과거 시점 재현일 때만)")
    usual_speed_kmh: float | None
    travel_time_min: float
    usual_travel_time_min: float | None
    extra_min: float | None = Field(description="평소 대비 추가 소요시간(분)")
    events: list[str]


class CorridorForecast(BaseModel):
    corridor: str
    corridor_name: str
    hub: str
    name: str
    issued_at: str
    segment_length_km: float
    model_version: str
    forecast: list[ForecastPoint]


class PredictResponse(BaseModel):
    forecasts: list[CorridorForecast]


class ObservationIn(BaseModel):
    ts: datetime
    speed: float = Field(..., gt=0, le=150, description="km/h")
    volume: float | None = Field(None, ge=0, description="대/시 (0 또는 생략 = 결측)")


class BatchTestRequest(BaseModel):
    # 드리프트 시뮬레이션 (scripts/simulate_drift.py). 연속된 시간 단위 관측치를 보내면 서버가
    # 저장 → 각 시각에 대해 직전까지의 데이터로 예측(슬라이딩) → 오차 누적 → 드리프트 판정.
    corridor: str
    observations: list[ObservationIn] = Field(..., min_length=1)


class BatchPoint(BaseModel):
    ts: str
    predicted: float
    actual: float | None
    known_event: bool


class BatchTestResponse(BaseModel):
    corridor: str
    model_version: str
    points: list[BatchPoint]
    drift_check: dict


class EventIn(BaseModel):
    hub: str
    type: EventType
    title: str = Field(..., min_length=1)
    start: datetime
    end: datetime
    expected_size: float | None = Field(None, ge=0, description="집회 신고 인원 / 공연장·경기장 수용 인원 (사후 관중 수 금지)")
    description: str | None = Field(None, description="공지 원문 - 행진·차로 통제 여부를 추출합니다")
    announced_at: datetime | None = Field(None, description="공개 시각 (생략 시 등록 시각)")
    source: str = "manual"

    @model_validator(mode="after")
    def _check_range(self):
        if self.end <= self.start:
            raise ValueError("end 는 start 이후여야 합니다")
        return self


class EventOut(EventIn):
    id: int
    march: bool
    lane_control: bool
    status: str


class EventStatusIn(BaseModel):
    status: Literal["scheduled", "review", "cancelled"] = Field(
        description="scheduled = 검토 후 확정, cancelled = 취소(지금 시각으로 공지 기록)")
    changed_at: datetime | None = Field(None, description="상태가 공지된 시각 (생략 시 지금)")
