"""
프로스포츠 경기 일정 → 거점 이벤트 (인증키 불필요, 각 리그 홈페이지의 일정 데이터).

    KBO   : koreabaseball.com 일정 (구장·시각·비고 '우천취소' 등)  → 잠실(jamsil)·고척(gocheok)
    K리그 : kleague.com 일정 (FC서울 K09 홈경기, 서울월드컵경기장)  → 상암(sangam)

누수 방지
    - expected_size 는 경기장 수용 인원. 관중 수(audienceQty)는 경기 후 확정 → 쓰지 않음.
    - announced_at: 과거 일정의 실제 공개 시각은 남아 있지 않아 '경기 announce_days 일 전' 으로 가정
      (정규시즌 일정은 시즌 전에 발표되므로 보수적인 가정). 리포트에 이 가정을 명시합니다.
    - 우천취소 등: status=cancelled, 취소 공지 시각 = 경기 시작 cancel_notice_hours 시간 전으로 가정.
"""
import re

import pandas as pd
import requests

KBO_URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList"
KLEAGUE_URL = "https://www.kleague.com/getScheduleList.do"
HEADERS = {"Referer": "https://www.koreabaseball.com/Schedule/Schedule.aspx",
           "X-Requested-With": "XMLHttpRequest", "User-Agent": "Mozilla/5.0"}
KBO_STADIUMS = {"잠실": ("jamsil", "잠실야구장", 23750), "고척": ("gocheok", "고척스카이돔", 16000)}
KBO_SERIES = "0,1,3,4,5,7,9,6"   # 정규시즌 + 포스트시즌
KBO_MINUTES = 190
FC_SEOUL, SANGAM_CAPACITY, KLEAGUE_MINUTES = "K09", 66704, 120


def _strip(html: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def _event(hub, title, start, minutes, size, desc, source, announce_days, cancelled, cancel_notice_hours):
    ev = {"hub": hub, "type": "sports", "title": title, "start": start, "end": start + pd.Timedelta(minutes=minutes),
          "expected_size": size, "description": desc, "source": source, "status": "scheduled",
          "announced_at": start.normalize() - pd.Timedelta(days=announce_days)}
    if cancelled:
        ev.update(status="cancelled", status_changed_at=start - pd.Timedelta(hours=cancel_notice_hours))
    return ev


def kbo_events(start: str, end: str, announce_days: int = 7, cancel_notice_hours: float = 2) -> list[dict]:
    out = []
    for month in pd.period_range(start, end, freq="M"):
        rows = requests.post(KBO_URL, headers=HEADERS, timeout=30, data={
            "leId": 1, "srIdList": KBO_SERIES, "seasonId": month.year, "gameMonth": f"{month.month:02d}", "teamId": "",
        }).json().get("rows", [])
        day = None
        for r in rows:
            cells = r["row"]
            if cells[0].get("Class") == "day":
                day = re.match(r"(\d{2})\.(\d{2})", _strip(cells[0]["Text"]))
                cells = cells[1:]
            if day is None or len(cells) < 3 or not re.fullmatch(r"\d{1,2}:\d{2}", _strip(cells[0]["Text"])):
                continue
            stadium, note = _strip(cells[-2]["Text"]), _strip(cells[-1]["Text"])
            if stadium not in KBO_STADIUMS:
                continue
            hub, venue, cap = KBO_STADIUMS[stadium]
            t0 = pd.Timestamp(f"{month.year}-{day.group(1)}-{day.group(2)} {_strip(cells[0]['Text'])}")
            if not (pd.Timestamp(start) <= t0 <= pd.Timestamp(end) + pd.Timedelta(days=1)):
                continue
            teams = re.sub(r"\s*\d+\s*vs\s*\d+\s*|\s+vs\s+", " vs ", _strip(cells[1]["Text"]))
            out.append(_event(hub, f"KBO {teams}", t0, KBO_MINUTES, cap,
                              f"{venue} KBO {teams} {note if note != '-' else ''}".strip(), "kbo_schedule",
                              announce_days, "취소" in note, cancel_notice_hours))
    return out


def kleague_events(start: str, end: str, announce_days: int = 7) -> list[dict]:
    out = []
    for month in pd.period_range(start, end, freq="M"):
        data = requests.post(KLEAGUE_URL, timeout=30, json={
            "year": str(month.year), "month": f"{month.month:02d}", "leagueId": "1", "teamId": FC_SEOUL,
        }).json().get("data", {}) or {}
        for g in data.get("scheduleList") or []:
            if g.get("homeTeam") != FC_SEOUL or "월드컵" not in (g.get("fieldName") or ""):
                continue
            t0 = pd.Timestamp(f"{g['gameDate'].replace('.', '-')} {g.get('gameTime') or '19:00'}")
            if not (pd.Timestamp(start) <= t0 <= pd.Timestamp(end) + pd.Timedelta(days=1)):
                continue
            title = f"K리그 서울 vs {g.get('awayTeamName', '')}"
            out.append(_event("sangam", title, t0, KLEAGUE_MINUTES, SANGAM_CAPACITY,
                              f"서울월드컵경기장 {g.get('meetName', '')} {title}", "kleague_schedule",
                              announce_days, False, 0))
    return out
