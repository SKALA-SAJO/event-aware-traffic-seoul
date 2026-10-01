"""
서울시 문화행사 정보(culturalEventInfo) → 거점 이벤트.

KOPIS 는 공연장 공연 중심이라 광장·공원의 무료 공공행사(서울광장 행사, 여의도 봄꽃축제 등)가 빠집니다.
이 데이터셋으로 광화문·여의도 거점의 축제·행사를 채웁니다.

    거점 배정 : LAT/LOT 좌표가 거점 반경 안인 행사만 (좌표가 없으면 PLACE 에 거점 장소명이 있을 때)
    교통 관련 : 반경 안이라도 박물관 전시·실내 공연이 대부분이므로(2026-10 실데이터: 반경 안 5,559건 중 대부분),
               PLACE 에 그 거점의 venues(광화문광장·서울광장·여의도공원 등 도로에 붙은 장소)가 있는 행사만 쓰고,
               31일 이상 이어지는 상설 프로그램은 제외한다.
    유형      : CODENAME 에 축제 → festival, 콘서트·클래식·국악·뮤지컬·연극·무용 → concert, 그 외 other
    시각      : PRO_TIME 의 "HH:MM~HH:MM" / "HH:MM" 을 해석. 해석 못하면 검토 대기(status=review) 1건
    공개 시각 : RGSTDATE(등록일) - 그 이전 예측에는 반영되지 않음
"""
import re

import pandas as pd

from data.config import hubs, nearest_hub

_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def _type(codename: str) -> str:
    if "축제" in codename:
        return "festival"
    if re.search(r"콘서트|클래식|국악|뮤지컬|오페라|연극|무용", codename):
        return "concert"
    return "other"


def _hub(row: dict) -> str | None:
    if row.get("lat") and row.get("lon"):
        lat, lon = float(row["lat"]), float(row["lon"])
        if lat > 90:  # 위도·경도가 뒤바뀐 행
            lat, lon = lon, lat
        return nearest_hub(lat, lon)
    place = row.get("place", "")
    return next((hid for hid, h in hubs().items() for v in h.get("venues", []) if v in place), None)


def to_events(rows: list[dict], start, end, max_days: int = 31) -> list[dict]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    venues = {hid: h.get("venues", []) for hid, h in hubs(include_disabled=True).items()}
    out = []
    for r in rows:
        hub = _hub(r)
        if hub is None or not r.get("start_date"):
            continue
        if not any(v in (r.get("place") or "") for v in venues.get(hub, [])):
            continue
        d0 = pd.Timestamp(str(r["start_date"])[:10])
        d1 = pd.Timestamp(str(r.get("end_date") or r["start_date"])[:10])
        if d1 < start or d0 > end:
            continue
        times = _TIME.findall(r.get("time_text") or "")
        typ = _type(r.get("codename", ""))
        reg = pd.to_datetime(r.get("registered"), errors="coerce")
        base = {"hub": hub, "type": typ, "title": r.get("title") or "", "expected_size": None,
                "description": f"{r.get('place', '')} / {r.get('codename', '')} / {r.get('time_text') or ''}",
                "source": "seoul_culture", "external_id": r.get("external_id"), "venue": r.get("place"),
                "lat": r.get("lat"), "lon": r.get("lon"),
                "announced_at": None if pd.isna(reg) else reg}
        if (d1 - d0).days >= max_days:  # 상설 프로그램
            continue
        if not times:  # 시각 미상 → 검토 대기
            s = d0 + pd.Timedelta(hours=10)
            out.append({**base, "start": s, "end": s + pd.Timedelta(hours=8), "status": "review", "end_estimated": True})
            continue
        h0, m0 = map(int, times[0])
        h1, m1 = map(int, times[1]) if len(times) > 1 else (None, None)
        for day in pd.date_range(max(d0, start.normalize()), min(d1, end), freq="D"):
            s = day + pd.Timedelta(hours=h0, minutes=m0)
            e = day + pd.Timedelta(hours=h1, minutes=m1) if h1 is not None else s + pd.Timedelta(hours=3)
            if e <= s:
                e += pd.Timedelta(days=1)
            out.append({**base, "start": s, "end": e, "status": "scheduled", "end_estimated": h1 is None})
    return out
