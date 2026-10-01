"""
서울 열린데이터광장 Open API 클라이언트.

    서비스                 데이터셋                         용도
    TrafficInfo            실시간 도로 소통 정보             링크별 현재 속도 (현재값만 → 매시간 수집해 이력 축적)
    VolInfo                교통량 이력 정보                  지점·시간·유입/유출·차로별 교통량 (최신 시간 도착 지연은 측정 필요)
    LinkInfo               소통 돌발 도로별 링크 정보        링크 도로명·시종점·지도거리 → corridor 길이·소요시간
    SpotInfo               교통량 지점 정보                  교통량 지점 이름·좌표
    AccInfo / AccMainCode  실시간 돌발 정보 / 돌발 유형 코드  사고·공사·통제 → 드리프트 판정 보조
    culturalEventInfo      서울시 문화행사 정보              무료 공공행사·축제 (KOPIS 에 없는 광장·공원 행사)

인증키: 환경변수 SEOUL_API_KEY. URL 형식: http://openapi.seoul.go.kr:8088/{KEY}/xml/{SERVICE}/{START}/{END}/{ARGS...}
(TrafficInfo 등 TOPIS 계열 서비스는 json 을 지원하지 않아(ERROR-301) 모든 서비스를 xml 로 받습니다.)

※ 2026-10-01 샘플 키(SEOUL_API_KEY=sample, 요청당 최대 5건)로 TrafficInfo·VolInfo·LinkInfo·SpotInfo·AccInfo·
   AccMainCode 를 호출해 서비스명·필드명을 확인했습니다. 샘플 키로는 앞 5건만 받으므로 운영에는 정식 키가 필요합니다.
"""
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BASE = "http://openapi.seoul.go.kr:8088"
TIMEOUT = 20
PAGE = 1000


def _key() -> str:
    key = os.getenv("SEOUL_API_KEY")
    if not key:
        raise RuntimeError("SEOUL_API_KEY 환경변수를 설정하세요 (서울 열린데이터광장 인증키)")
    return key


def _rows(service: str, *path, start: int = 1, end: int = PAGE) -> list[dict]:
    key = _key()
    if key == "sample":  # 샘플 키는 요청당 최대 5건
        if start > 5:
            return []
        end = min(end, 5)
    url = "/".join([BASE, key, "xml", service, str(start), str(end), *map(str, path)])
    # 인증키가 URL 경로에 들어가므로 requests 예외·raise_for_status 메시지(URL 포함)를 그대로 내보내지 않음
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(f"{service} 요청 실패 ({type(e).__name__})") from None
    if not resp.ok:
        raise RuntimeError(f"{service} HTTP {resp.status_code}")
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        raise RuntimeError(f"{service} XML 이 아닌 응답") from None
    code = root.findtext("RESULT/CODE") or root.findtext("CODE") or ""
    if code == "INFO-200":  # 해당 데이터 없음
        return []
    if not code.startswith("INFO-000"):
        raise RuntimeError(f"{service} 오류: {code} {root.findtext('RESULT/MESSAGE') or root.findtext('MESSAGE')}")
    return [{c.tag: (c.text or "").strip() for c in row} for row in root.findall("row")]


def _all_rows(service: str, *path, limit: int = 20000) -> list[dict]:
    out, start = [], 1
    while start <= limit:
        rows = _rows(service, *path, start=start, end=start + PAGE - 1)
        out += rows
        if len(rows) < PAGE:
            break
        start += PAGE
    return out


def _f(row: dict, *names, default=None):
    """대소문자 무시, 후보 이름 중 처음 있는 필드."""
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        v = low.get(n.lower())
        if v not in (None, ""):
            return v
    return default


def _require(row: dict, *names):
    v = _f(row, *names)
    if v is None:
        raise KeyError(f"응답에 {names} 필드가 없습니다 - 실제 필드: {list(row)}")
    return v


# ─────────────────────────────── 속도 · 교통량 · 링크 ───────────────────────────────

def link_speed(link_id: str) -> float | None:
    """TrafficInfo: 링크의 현재 속도(km/h)."""
    rows = _rows("TrafficInfo", link_id, end=5)
    if not rows:
        return None
    v = float(_require(rows[0], "prcs_spd"))
    return v if v > 0 else None


def spot_volume(spot_num: str, ts: pd.Timestamp) -> list[dict]:
    """VolInfo(교통량 이력 정보): 지점·시간대의 유입/유출별 교통량 (차로 합). 반환 [{io_type, volume}]"""
    rows = _rows("VolInfo", spot_num, ts.strftime("%Y%m%d"), ts.strftime("%H"))
    agg: dict[int, float] = {}
    for r in rows:
        io = int(_f(r, "io_type", default=0) or 0)
        agg[io] = agg.get(io, 0.0) + float(_f(r, "vol", default=0) or 0)
    return [{"io_type": io, "volume": v if v > 0 else None} for io, v in agg.items()]


def link_info(link_id: str) -> dict | None:
    """LinkInfo: 도로명·시작/종료 노드명·지도거리(m)."""
    rows = _rows("LinkInfo", link_id, end=5)
    if not rows:
        return None
    r = rows[0]
    return {"link_id": link_id, "road_name": _f(r, "road_name"), "st_node": _f(r, "st_node_nm"),
            "ed_node": _f(r, "ed_node_nm"), "length_m": float(_f(r, "map_dist", default=0) or 0)}


def spot_info() -> list[dict]:
    """SpotInfo: 교통량 지점 번호·이름·좌표(GRS80TM)."""
    return [{"spot": _f(r, "spot_num"), "name": _f(r, "spot_nm"),
             "x": _f(r, "grs80tm_x"), "y": _f(r, "grs80tm_y")} for r in _all_rows("SpotInfo")]


# ─────────────────────────────────── 돌발 정보 ───────────────────────────────────

_CATEGORY_RULES = [("construction", r"공사|보수|정비"), ("event", r"집회|행사|시위|축제|마라톤"),
                   ("accident", r"사고|고장|낙하물|화재"), ("control", r"통제|차단|폐쇄")]


def classify_incident(type_name: str | None, info: str | None) -> str:
    text = f"{type_name or ''} {info or ''}"
    for cat, pat in _CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    return "other"


def incident_type_names() -> dict[str, str]:
    """AccMainCode: 돌발 유형 코드 → 이름. 실패하면 빈 dict (분류는 acc_info 텍스트로 대체)."""
    try:
        return {str(_f(r, "acc_type")): _f(r, "acc_type_nm") for r in _all_rows("AccMainCode")}
    except Exception:
        return {}


def _dt(date, time) -> pd.Timestamp | None:
    if not date:
        return None
    t = str(time or "0").strip()
    t = t.zfill(4) + "00" if len(t) <= 4 else t.zfill(6)  # occr_time 은 HHMM(예: 1000), exp_clr_time 은 HHMMSS
    try:
        return pd.Timestamp(f"{date} {t[:2]}:{t[2:4]}:{t[4:6]}")
    except ValueError:
        return None


def incidents() -> list[dict]:
    """AccInfo: 현재 발생 중인 돌발 목록."""
    names = incident_type_names()
    out = []
    for r in _all_rows("AccInfo"):
        code = str(_f(r, "acc_type", default=""))
        info = _f(r, "acc_info")
        out.append({
            "acc_id": str(_require(r, "acc_id")), "link_id": _f(r, "link_id"),
            "type_code": code, "type_name": names.get(code), "info": info,
            "category": classify_incident(names.get(code), info),
            "start": _dt(_f(r, "occr_date"), _f(r, "occr_time")),
            "expected_end": _dt(_f(r, "exp_clr_date"), _f(r, "exp_clr_time")),
        })
    return out


# ─────────────────────────────────── 문화행사 ───────────────────────────────────

def cultural_events() -> list[dict]:
    """culturalEventInfo: 서울시 문화행사 전체 (STRTDATE/END_DATE/PRO_TIME/PLACE/LAT/LOT/RGSTDATE ...)."""
    out = []
    for r in _all_rows("culturalEventInfo"):
        lat, lot = _f(r, "LAT"), _f(r, "LOT")
        try:
            lat, lot = float(lat), float(lot)
            if lat > 90:  # 위도·경도가 뒤바뀐 행 방어
                lat, lot = lot, lat
        except (TypeError, ValueError):
            lat = lot = None
        out.append({
            "title": _f(r, "TITLE"), "codename": _f(r, "CODENAME", default=""), "place": _f(r, "PLACE", default=""),
            "start_date": _f(r, "STRTDATE"), "end_date": _f(r, "END_DATE"), "time_text": _f(r, "PRO_TIME", "DATE"),
            "registered": _f(r, "RGSTDATE"), "lat": lat, "lon": lot, "org": _f(r, "ORG_NAME"),
            "external_id": _f(r, "HMPG_ADDR", "ORG_LINK"),
        })
    return out
