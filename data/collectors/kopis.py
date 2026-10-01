"""
KOPIS(공연예술통합전산망) Open API → 거점 대형 공연 이벤트.

    공연시설목록 GET /prfplc?signgucode=11&signgucodesub={구 코드}      거점이 있는 구의 공연시설
    공연시설상세 GET /prfplc/{mt10id}   la/lo(위도·경도), mt13s(공연장별 객석 수)
    공연목록     GET /pblprfr?stdate&eddate&prfplccd={mt10id}          기간 최대 31일, 페이지당 최대 100건
    공연상세     GET /pblprfr/{mt20id}  fcltynm("시설 (공연장)"), dtguidance(요일별 시각), prfruntime, updatedate
    인증키: 환경변수 KOPIS_API_KEY.  초당 10회 이상 호출 시 서비스가 중지되므로 호출 간격을 둡니다.

설계
    - 교통에 영향을 줄 만한 공연만: 거점 중심 반경(event_match_radius_km) 안 시설 중, 실제 공연한 공연장(홀)의
      객석이 kopis_min_seats 이상인 공연만 이벤트로 만든다. 규모 = 그 공연장 객석 수 (시설 전체 합계가 아님 -
      잠실종합운동장 81,813석 중 주경기장 65,599 / 실내체육관 11,032 / 빅탑시어터 500).
    - 시설 목록은 구 단위로 받아 좌표·객석을 확인하고 data/raw/kopis_venues.json 에 캐시한다 (호출 수 절약).
    - "공연 기간 ≠ 매일 공연": dtguidance 를 요일별 시각으로 해석해 실제 회차만 이벤트로 만든다.
      해석하지 못하면 공연 1건당 검토 대기(status=review) 1건만 남긴다.
    - 종료 시각 = 시작 + prfruntime. 애매한 표기는 기본 소요시간으로 추정(end_estimated=1). 공연 시간이 비어 있고
      축제(festival=Y)이거나 낮(DAYTIME_START_HOUR 이전)에 시작하면 종일 행사로 보고 그날 FESTIVAL_END_HOUR 시
      종료로 추정한다 (예: MADLY MEDLEY 2026-09-05 11:00 시작은 festival=N 으로 등록돼 있음).
    - 공개 시각(announced_at): KOPIS 는 등록 시각을 주지 않지만 updatedate(마지막 갱신 시각)가 첫 공연 전이면
      늦어도 그때는 공개돼 있었으므로 그 값을 쓴다(보수적). 첫 공연 이후에 갱신된 공연만
      fallback_days 일 전 공개로 가정하고 description 에 표시한다.
※ 2026-10-01 인증키로 호출해 필드(mt13s/seatscale, fcltynm 괄호 공연장명, updatedate)를 확인했습니다.
"""
import functools
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from data.config import haversine_km, holidays, hubs, section

BASE = "http://www.kopis.or.kr/openApi/restful"
DEFAULT_RUNTIME_MIN = 180
FESTIVAL_END_HOUR = 22
DAYTIME_START_HOUR = 15
VENUE_CACHE = "data/raw/kopis_venues.json"
_DOW = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
_last_call = [0.0]


class KopisError(RuntimeError):
    """KOPIS 호출 실패. 메시지에 요청 URL(인증키 포함)을 절대 넣지 않는다."""


def _get(path: str, **params) -> ET.Element:
    key = os.getenv("KOPIS_API_KEY")
    if not key:
        raise KopisError("KOPIS_API_KEY 환경변수를 설정하세요")
    status = None
    for attempt in range(3):
        wait = section("collection")["kopis_request_interval_sec"] - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(f"{BASE}/{path}", params={"service": key, **params}, timeout=30)
        except requests.RequestException as e:  # 예외 문자열에 URL 이 들어가므로 종류만 남김
            status = type(e).__name__
        else:
            status = resp.status_code
            if resp.ok:
                try:
                    return ET.fromstring(resp.content)
                except ET.ParseError:  # 간헐적으로 XML 이 아닌 응답
                    status = "non-xml"
        finally:
            _last_call[0] = time.time()
        time.sleep(2 * (attempt + 1))
    raise KopisError(f"KOPIS {path} 실패 ({status})")


def _items(root: ET.Element) -> list[dict]:
    return [{c.tag: (c.text or "").strip() for c in db} for db in root.findall("db")]


def _paged(path: str, **params) -> list[dict]:
    out, page = [], 1
    while True:
        items = _items(_get(path, cpage=page, rows=100, **params))
        out += items
        if len(items) < 100:
            return out
        page += 1


def _seats(v) -> int:
    return int(re.sub(r"\D", "", v or "") or 0)


# ─────────────────────────────── 공연시설 ───────────────────────────────

def venue_detail(mt10id: str) -> dict:
    d = _get(f"prfplc/{mt10id}").find("db")
    if d is None:
        return {}
    halls = {(m.findtext("prfplcnm") or "").strip(): _seats(m.findtext("seatscale")) for m in d.findall("mt13s/mt13")}
    try:
        lat, lon = float(d.findtext("la")), float(d.findtext("lo"))
    except (TypeError, ValueError):
        lat = lon = None
    return {"mt10id": mt10id, "name": (d.findtext("fcltynm") or "").strip(), "lat": lat, "lon": lon,
            "seats": _seats(d.findtext("seatscale")), "halls": halls}


def hub_venues(min_seats: int | None = None, refresh: bool = False) -> list[dict]:
    """거점 반경 안에 있고 min_seats 이상 공연장이 있는 시설 [{mt10id, name, hub, lat, lon, halls}]."""
    cfg = section("collection")
    min_seats = min_seats if min_seats is not None else cfg["kopis_min_seats"]
    radius = cfg["event_match_radius_km"]
    if os.path.exists(VENUE_CACHE) and not refresh:
        with open(VENUE_CACHE, encoding="utf-8") as f:
            details = json.load(f)
    else:
        districts = sorted({str(d) for h in hubs(include_disabled=True).values() for d in h.get("kopis_districts", [])})
        ids = {v["mt10id"] for sub in districts for v in _paged("prfplc", signgucode="11", signgucodesub=sub)}
        details = []
        for i in sorted(ids):
            try:
                details.append(venue_detail(i))
            except KopisError as e:  # 상세가 없는 시설(400 등)은 건너뜀
                print(f"  건너뜀: {e}")
        os.makedirs(os.path.dirname(VENUE_CACHE), exist_ok=True)
        with open(VENUE_CACHE, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False)
    out = []
    for v in details:
        if v.get("lat") is None or max(v["halls"].values(), default=v["seats"]) < min_seats:
            continue
        dist = {h: haversine_km((v["lat"], v["lon"]), tuple(c["center"]))
                for h, c in hubs(include_disabled=True).items() if c.get("center")}
        hub = min(dist, key=dist.get)
        if dist[hub] <= radius:
            out.append({**v, "hub": hub, "dist_km": round(dist[hub], 2)})
    return out


# ─────────────────────────────── 공연 ───────────────────────────────

def list_performances(start, end, venue_id: str) -> list[dict]:
    """31일 단위로 나눠 한 시설의 공연 목록 (mt20id 중복 제거)."""
    max_days = section("collection")["kopis_max_days"]
    out: dict[str, dict] = {}
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        chunk_end = min(s + pd.Timedelta(days=max_days - 1), e)
        for it in _paged("pblprfr", stdate=f"{s:%Y%m%d}", eddate=f"{chunk_end:%Y%m%d}", prfplccd=venue_id):
            out.setdefault(it.get("mt20id"), it)
        s = chunk_end + pd.Timedelta(days=1)
    return list(out.values())


@functools.lru_cache(maxsize=4096)
def detail(mt20id: str) -> dict:
    items = _items(_get(f"pblprfr/{mt20id}"))
    return items[0] if items else {}


def parse_dtguidance(text: str) -> dict:
    """
    '화요일 ~ 금요일(20:00), 토요일(14:00,18:00), HOL(15:00)' →
    {1: ['20:00'], ..., 4: ['20:00'], 5: ['14:00', '18:00'], 'HOL': ['15:00']}
    해석할 수 없으면 {}.
    """
    out: dict = {}
    for m in re.finditer(r"([월화수목금토일]요일|HOL)\s*(?:~\s*([월화수목금토일])요일)?\s*\(([^)]*)\)", text or ""):
        times = re.findall(r"\d{1,2}:\d{2}", m.group(3))
        if not times:
            continue
        if m.group(1) == "HOL":
            out.setdefault("HOL", []).extend(times)
            continue
        d0 = _DOW[m.group(1)[0]]
        d1 = _DOW[m.group(2)] if m.group(2) else d0
        for d in range(d0, (d1 if d1 >= d0 else d1 + 7) + 1):
            out.setdefault(d % 7, []).extend(times)
    return out


def parse_runtime(text: str) -> int | None:
    """'2시간 30분' → 150, '100분' → 100. 인터미션·범위·복수 표기 등 애매하면 None."""
    t = (text or "").replace(" ", "")
    m = re.fullmatch(r"(?:(\d+)시간)?(?:(\d+)분)?", t)
    if not t or not m or not (m.group(1) or m.group(2)):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def hall_seats(fcltynm: str, venue: dict) -> int | None:
    """'잠실종합운동장 (실내체육관)' → 그 공연장 객석. 공연장명을 못 찾으면 None (시설 합계로 대신하지 않음)."""
    m = re.search(r"\(([^()]*)\)\s*$", fcltynm or "")
    halls = venue["halls"]
    if m:
        name = m.group(1).strip()
        for hall, seats in halls.items():
            if hall == name or name in hall or hall in name:
                return seats or None
        return None
    return next(iter(halls.values())) if len(halls) == 1 else None


def events_for_hubs(start, end, collected_at: pd.Timestamp, fallback_days: int = 30,
                    min_seats: int | None = None) -> tuple[list[dict], dict]:
    """거점 대형 공연장의 공연을 회차 단위 이벤트로 변환. 반환 (이벤트, 집계)."""
    min_seats = min_seats if min_seats is not None else section("collection")["kopis_min_seats"]
    hol = holidays()
    events, stats = [], {"venues": [], "performances": 0, "small_hall": 0, "unknown_hall": 0, "assumed_announce": 0}
    for v in hub_venues(min_seats):
        stats["venues"].append(f"{v['name']}({v['hub']})")
        for p in list_performances(start, end, v["mt10id"]):
            try:
                d = detail(p["mt20id"])
            except KopisError as e:
                print(f"  건너뜀: {e}")
                continue
            seats = hall_seats(d.get("fcltynm") or p.get("fcltynm", ""), v)
            if seats is None:
                stats["unknown_hall"] += 1
                continue
            if seats < min_seats:
                stats["small_hall"] += 1
                continue
            stats["performances"] += 1
            times = parse_dtguidance(d.get("dtguidance", ""))
            runtime = parse_runtime(d.get("prfruntime", ""))
            first = pd.Timestamp(p["prfpdfrom"].replace(".", "-"))
            last = pd.Timestamp(p["prfpdto"].replace(".", "-"))
            updated = pd.to_datetime((d.get("updatedate") or "")[:19], errors="coerce")
            if pd.notna(updated) and updated < first:
                announced, note = updated, f"공개 시각 = KOPIS 갱신 시각 {updated:%Y-%m-%d}"
            elif pd.Timestamp(collected_at) < first:
                announced, note = pd.Timestamp(collected_at), "공개 시각 = 수집 시각"
            else:
                announced, note = first - pd.Timedelta(days=fallback_days), f"공개 시각 가정: 첫 공연 {fallback_days}일 전"
                stats["assumed_announce"] += 1
            base = {
                "hub": v["hub"], "type": "concert", "title": p.get("prfnm", ""), "expected_size": float(seats),
                "description": f"{d.get('fcltynm', '')} {seats:,}석 / {d.get('dtguidance', '')} / "
                               f"공연시간 {d.get('prfruntime', '')} / {note}",
                "source": "kopis", "external_id": p["mt20id"], "venue": d.get("fcltynm"), "lat": v["lat"],
                "lon": v["lon"], "runtime_text": d.get("prfruntime"), "end_estimated": runtime is None,
                "announced_at": announced,
            }
            festival = d.get("festival") == "Y"

            def end_of(start_ts):
                if runtime:
                    return start_ts + pd.Timedelta(minutes=runtime)
                if (festival or start_ts.hour < DAYTIME_START_HOUR) and start_ts.hour < FESTIVAL_END_HOUR:
                    return start_ts.normalize() + pd.Timedelta(hours=FESTIVAL_END_HOUR)
                return start_ts + pd.Timedelta(minutes=DEFAULT_RUNTIME_MIN)

            if not times:  # 회차 시각 미확인 → 매일 만들지 않고 검토 대기 1건
                s = pd.Timestamp(f"{first:%Y-%m-%d} 18:00")
                events.append({**base, "start": s, "end": end_of(s), "status": "review", "end_estimated": True})
                continue
            for day in pd.date_range(first, last, freq="D"):
                slots = times.get("HOL") if day.date() in hol and "HOL" in times else times.get(day.dayofweek)
                for hhmm in slots or []:
                    s = pd.Timestamp(f"{day:%Y-%m-%d} {hhmm}")
                    events.append({**base, "title": f"{base['title']} {hhmm}" if len(slots) > 1 else base["title"],
                                   "start": s, "end": end_of(s), "status": "scheduled"})
    return events, stats
