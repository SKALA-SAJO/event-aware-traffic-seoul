"""
서울경찰청 "오늘의 주요집회" 정리본 → 광화문·여의도 집회 이벤트.

    원천: data/raw/events/smpa_rallies.csv  (date, start, end, count, station, place, written_at, posted_date, announced_at)
          data/raw/events/smpa_rallies_team.csv  (+ dong, march, lane - 팀원이 원문에서 추출)
    거점 배정: hubs.yaml 의 police_stations(관할서)로, event_types 에 rally 가 있는 거점만 (광화문·여의도).
              동(dong) 정보가 있으면 hubs.yaml 의 rally_dong_keywords 중 하나가 들어간 동만 남김
              (예: 영등포서 중 '여의'·'의사당' - 표기가 여의도동/여의도/여의대로 등으로 제각각이라 부분 일치).
    규모: 신고 인원(사전 공개값). 소규모 집회가 하루 수십 건이라 min_count 이상만 이벤트로 만든다.
    공개 시각: announced_at - PDF 작성 시각(보통 전날 18시)과 게시일 중 늦은 쪽 (원천 정리 규칙).
    행진·차로 점거: 팀원 자료와 (date, start, count, station) 으로 맞춰 march / lane_control 에 넣는다.
"""
import pandas as pd

from data.config import hubs

DEFAULT_HOURS = 3  # 종료 시각이 없을 때


def _at(date, hhmm: str) -> pd.Timestamp:
    """'24:00'·'25:30' 처럼 자정 넘는 표기도 허용."""
    h, m = map(int, hhmm.strip().split(":")[:2])
    return pd.Timestamp(date) + pd.Timedelta(hours=h, minutes=m)


def _hub_for(station: str, dong, station_hubs: dict, dongs: dict) -> str | None:
    for st in str(station or "").split("|"):
        hub = station_hubs.get(st.strip())
        if hub is None:
            continue
        allowed = dongs.get(hub)
        if allowed and isinstance(dong, str) and dong and not any(k in dong for k in allowed):
            return None
        return hub
    return None


def to_events(rallies_csv: str, team_csv: str | None = None, min_count: int = 1000,
              start=None, end=None) -> list[dict]:
    r = pd.read_csv(rallies_csv, dtype={"start": str, "end": str})
    if team_csv:
        t = pd.read_csv(team_csv, dtype={"start": str, "end": str})[["date", "start", "count", "station", "dong", "march", "lane"]]
        t = t.drop_duplicates(["date", "start", "count", "station"])
        r = r.merge(t, on=["date", "start", "count", "station"], how="left")
    r = r[pd.to_numeric(r["count"], errors="coerce").fillna(0) >= min_count]

    cfg = hubs(include_disabled=True)
    rally_hubs = {h: c for h, c in cfg.items() if "rally" in c.get("event_types", [])}
    station_hubs = {st: h for h, c in rally_hubs.items() for st in c.get("police_stations", [])}
    dongs = {h: c["rally_dong_keywords"] for h, c in rally_hubs.items() if c.get("rally_dong_keywords")}

    out = []
    for x in r.itertuples(index=False):
        hub = _hub_for(x.station, getattr(x, "dong", None), station_hubs, dongs)
        if hub is None or not isinstance(x.start, str):
            continue
        try:
            s = _at(x.date, x.start)
            e = _at(x.date, x.end) if isinstance(x.end, str) and ":" in x.end else s + pd.Timedelta(hours=DEFAULT_HOURS)
        except ValueError:
            continue
        if e <= s:
            e += pd.Timedelta(days=1)
        if (start is not None and e < pd.Timestamp(start)) or (end is not None and s > pd.Timestamp(end)):
            continue
        march = int(getattr(x, "march", 0) == 1)
        lane = int(getattr(x, "lane", 0) == 1)
        place = x.place if isinstance(x.place, str) else ""
        out.append({
            "hub": hub, "type": "rally", "title": f"집회 {int(x.count):,}명 ({x.station})", "start": s, "end": e,
            "expected_size": float(x.count), "source": "smpa", "venue": place or None,
            "description": f"{place} 신고 {int(x.count):,}명 {x.station}서" + (" 행진" if march else "") + (" 차로 점거" if lane else ""),
            "announced_at": pd.Timestamp(x.announced_at), "march": march, "lane_control": lane, "status": "scheduled",
        })
    return out
