"""
이벤트 일정 → 시간 단위 피처 (실험 ②: 이벤트를 모델에 넣는 방식 비교).

    flag  : 영향 구간(시작 전 ~ 종료 후) 0/1 플래그
    scale : 플래그 × 규모 가중치 (집회 신고 인원 / 공연장·경기장 수용 인원, 로그 스케일)
    decay : 규모 × 시간 감쇠 - 시작 전(도착), 진행 중, 종료 후(귀가)를 별도 채널로 분리하고
            시작·종료 시각과 멀어질수록 지수적으로 줄어듦 → 공연 종료 후 귀가 정체를 표현
    text  : decay + 텍스트에서 추출한 행진·차로 통제 여부 (data/event_nlp.py)

정보 누수 방지
    이벤트 피처는 "예측을 발행하는 시점에 이미 공개된 일정"만 써야 합니다 (announced_at).
    - 서빙: known_cutoff=발행 시각 → 그 시각까지 공개된 일정만 반영
    - 학습: 샘플마다 발행 시각이 달라 per-sample 필터는 비싸므로, 시각 t의 피처에는
            announced_at <= t - horizon 인 일정만 넣습니다. 발행 시각은 항상 t - horizon 이후이므로
            어떤 샘플에서도 미래에 공개될 일정이 새어 들어가지 않는 보수적인 규칙입니다.
    - 관중 수처럼 사후에 확정되는 값은 expected_size 에 넣지 않습니다(수용 인원으로 대체).

상태(status)
    review    : 자동 수집했지만 시각을 확정하지 못한 일정 (예: KOPIS 회차 시각 미확인) → 피처에서 제외
    cancelled : 취소. 취소가 공개된 시각(status_changed_at) 이후에 발행된 예측에서만 제외합니다.
                그 전에는 "예정"으로 보였으므로 학습에서도 예정으로 넣어야 운영과 같아집니다(우천취소 등).

거점마다 이벤트 유형이 겹치지 않도록 선정했기 때문에(광화문=집회, 잠실·상암=경기·공연 …)
유형별 채널을 따로 두지 않고, 유형별 반응 차이는 거점 one-hot과의 조합으로 학습합니다.
"""
import numpy as np
import pandas as pd

from data.config import FORECAST, HORIZON

EVENT_TYPES = ["rally", "concert", "sports", "festival", "marathon", "other"]
EVENT_MODES = ["none", "flag", "scale", "decay", "text"]

# 규모 정보가 없을 때 쓰는 유형별 기본값
DEFAULT_SIZE = {"rally": 2000, "concert": 30000, "sports": 20000, "festival": 50000, "marathon": 20000, "other": 5000}
SIZE_REF = 100_000  # log1p(size)/log1p(SIZE_REF) 로 정규화 (10만 명 ≈ 1.0)

PRE_H = float(FORECAST["event_pre_hours"])
POST_H = float(FORECAST["event_post_hours"])
TAU = float(FORECAST["decay_tau_hours"])

_NAMES = {
    "none": [],
    "flag": ["ev_flag"],
    "scale": ["ev_scale"],
    "decay": ["ev_pre", "ev_active", "ev_post"],
    "text": ["ev_pre", "ev_active", "ev_post", "ev_march", "ev_lane"],
}


def event_feature_names(mode: str) -> list[str]:
    return _NAMES[mode]


def size_scale(size, ev_type: str) -> float:
    if size is None or pd.isna(size) or size <= 0:
        size = DEFAULT_SIZE.get(ev_type, DEFAULT_SIZE["other"])
    return float(min(np.log1p(size) / np.log1p(SIZE_REF), 1.5))


def _positions(hours: pd.DatetimeIndex, ts) -> float:
    return (pd.Timestamp(ts) - hours[0]) / pd.Timedelta(hours=1)


def event_features(
    hours: pd.DatetimeIndex,
    events: pd.DataFrame,
    mode: str,
    known_cutoff: pd.Timestamp | None = None,
    leak_guard_hours: int = HORIZON,
) -> np.ndarray:
    """
    hours: 1시간 간격의 연속된 시각(시간대 시작 시각). events: 한 거점의 이벤트.
    반환: (len(hours), len(event_feature_names(mode))) - 겹치는 이벤트는 채널별 최댓값.
    """
    names = event_feature_names(mode)
    out = np.zeros((len(hours), len(names)), dtype="float32")
    if not names or events is None or events.empty:
        return out

    idx = np.arange(len(hours), dtype="float64")
    centers = idx + 0.5
    for ev in events.itertuples(index=False):
        status = getattr(ev, "status", "scheduled") or "scheduled"
        if status == "review":
            continue
        s_pos, e_pos = _positions(hours, ev.start), _positions(hours, ev.end)
        if e_pos + POST_H < 0 or s_pos - PRE_H > len(hours):
            continue

        # 이 이벤트가 "예정으로 알려진" 시간대: 공개 이후 & (취소됐다면) 취소 공개 이전
        announced = ev.announced_at if not pd.isna(ev.announced_at) else None
        cancelled_at = getattr(ev, "status_changed_at", None) if status == "cancelled" else None
        cancelled_at = None if cancelled_at is None or pd.isna(cancelled_at) else pd.Timestamp(cancelled_at)
        if status == "cancelled" and cancelled_at is None:
            continue  # 취소 시각을 모르면 처음부터 없던 일정으로 취급
        if known_cutoff is not None:
            if announced is not None and pd.Timestamp(announced) > known_cutoff:
                continue
            if cancelled_at is not None and cancelled_at <= known_cutoff:
                continue
            known = np.ones(len(hours), dtype=bool)
        else:
            known = np.ones(len(hours), dtype=bool)
            if announced is not None:
                known &= idx >= _positions(hours, announced) + leak_guard_hours
            if cancelled_at is not None:  # 취소가 알려진 뒤에 발행된 예측에서는 빠짐
                known &= idx < _positions(hours, cancelled_at) + leak_guard_hours

        active = (idx + 1 > s_pos) & (idx <= e_pos)
        pre = (idx + 1 <= s_pos) & (idx >= s_pos - PRE_H)
        post = (idx > e_pos) & (idx <= e_pos + POST_H)
        window = (active | pre | post) & known
        if not window.any():
            continue

        scale = size_scale(ev.expected_size, ev.type)
        if mode == "flag":
            chans = [window.astype("float32")]
        elif mode == "scale":
            chans = [window * scale]
        else:
            pre_v = np.where(pre & known, scale * np.exp(np.minimum(centers - s_pos, 0) / TAU), 0.0)
            act_v = np.where(active & known, scale, 0.0)
            post_v = np.where(post & known, scale * np.exp(np.minimum(e_pos - centers, 0) / TAU), 0.0)
            chans = [pre_v, act_v, post_v]
            if mode == "text":
                chans += [act_v * bool(ev.march), act_v * bool(ev.lane_control)]
        out = np.maximum(out, np.stack(chans, axis=1).astype("float32"))
    return out


def event_mask(hours: pd.DatetimeIndex, events: pd.DataFrame) -> np.ndarray:
    """
    평가·드리프트 판정용 "이벤트 시간대" (시작 PRE_H 전 ~ 종료 POST_H 후). 공개 시점과 무관.
    취소된 일정은 실제로 일어나지 않았으므로 제외하고, 검토 대기 일정은 실제로 열렸을 수 있으므로
    포함합니다 (드리프트 오탐 방지 쪽으로 보수적).
    """
    mask = np.zeros(len(hours), dtype=bool)
    if events is None or events.empty:
        return mask
    idx = np.arange(len(hours), dtype="float64")
    for ev in events.itertuples(index=False):
        if (getattr(ev, "status", "scheduled") or "scheduled") == "cancelled":
            continue
        s_pos, e_pos = _positions(hours, ev.start), _positions(hours, ev.end)
        mask |= (idx + 1 > s_pos - PRE_H) & (idx <= e_pos + POST_H)
    return mask
