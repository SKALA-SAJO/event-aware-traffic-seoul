"""
서울경찰청 "오늘의 주요집회" 게시판을 매일 받아 구조화 (GitHub Actions 일일 수집용).

    게시판  : https://www.smpa.go.kr/user/nd54882.do  (행사일별 게시글 1개, 첨부 PDF)
    PDF     : pdftotext -layout 으로 텍스트로 바꿔 한 건(시간~시간 + 장소 + 신고 인원 + 관할서)씩 나눔
    공개 시각: PDF 상단 "(10. 1. 18:00 기준 작성)"과 게시일 중 늦은 쪽 (게시일이 더 늦으면 게시일 23:59). 행사 당일 게시도 있어
              "전날 18시"로 가정하지 않음. 이 규칙은 학습에 쓴 과거 집회(scripts/import_team_raw.py)와 같다.
    날짜 점검: 본문 날짜가 파일(행사)일과 다르면 - 요일이 맞으면 본문 오타로 보고 사용, 다른 날짜 자료와 내용이 같거나(복사) 요일도
              다르면 어느 날 자료인지 알 수 없어 제외.

과거 12,000여 건을 정리한 원본 스크립트는 팀 저장소의 fetch_smpa_rallies.py · parse_smpa_rallies.py 이고, 여기서는
같은 규칙을 "새로 올라온 며칠치만" 받도록 줄여 옮겼다 (표준 라이브러리만 사용, pdftotext 필요).
"""
import datetime as dt
import hashlib
import html
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

BASE = "https://www.smpa.go.kr"
BOARD = "/user/nd54882.do"
TIMEOUT = 20
# 해외(GitHub Actions) 러너에서 PDF 받기가 간헐적으로 끊김(10-02: RemoteDisconnected, 같은 실행의 다른 PDF는 성공).
# 브라우저처럼 User-Agent 를 보내고, 실패하면 3·6·12초 쉬었다가 다시 시도한다 (하루 1~2개 PDF 라 시간 부담 없음).
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36"}
RETRY_WAITS = (3, 6, 12)

WRITTEN_RE = re.compile(r"\(\s*(?:(20\d{2})\s*\.\s*)?(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*(\d{1,2}):(\d{2})\s*기준\s*작[성서]\s*\)")
BODY_DATE_RE = re.compile(r"(20\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\(\s*([월화수목금토일])")
WEEKDAY = "월화수목금토일"
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*[~∼～-]\s*(\d{1,2}):(\d{2})")
COUNT_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)\s*명")
STATIONS = ["종로", "남대문", "중부", "용산", "영등포", "서초", "강남", "송파", "마포", "서대문",
            "혜화", "동대문", "성북", "구로", "관악", "동작", "강서", "양천", "노원", "광진",
            "성동", "은평", "수서", "방배", "중랑", "강동", "도봉", "강북", "금천", "종암"]
# "명"이 빠진 PDF: 표의 인원 칸 숫자는 뒤에 넓은 공백·줄끝·관할서명이 온다 ("4出", "2개 차로", "1가" 같은 장소 속 숫자와 구분)
COUNT_FALLBACK_RE = re.compile(
    r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d{2,})(?=\s{2,}|\s*$|\s+(?:" + "|".join(STATIONS) + r"))", re.M)

ROW_COLS = ["date", "start", "end", "count", "station", "place", "text", "written_at", "posted_date", "announced_at"]


def _get(url: str, data: dict | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data else None
    last = None
    for wait in (0, *RETRY_WAITS):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=HEADERS), timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:
            last = e
    raise last


def list_posts(pages: int = 2) -> list[dict]:
    """게시판 최신 글 목록 [{board_no, event_ymd, posted(date)}] (최신순)."""
    posts = []
    for page in range(1, pages + 1):
        t = _get(BASE + BOARD, {"page": page, "pageSC": "SORT_ORDER", "pageSO": "DESC", "dmlType": "SELECT"}).decode("utf-8", "ignore")
        for row in re.findall(r"<tr.*?</tr>", t, flags=re.S)[1:]:
            board_no = re.findall(r"View','(\d+)'", row)
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", row)))
            ymd = re.search(r"(\d{6})", text)
            posted = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            if board_no and ymd:
                posts.append({"board_no": board_no[0], "event_ymd": ymd[1],
                              "posted": dt.date.fromisoformat(posted[1]) if posted else None})
    return posts


def fetch_text(post: dict) -> tuple[str, str] | None:
    """게시글 → (행사일 YYMMDD, PDF 텍스트). PDF 첨부가 없으면 None."""
    t = _get(f"{BASE}{BOARD}?View&boardNo={post['board_no']}").decode("utf-8", "ignore")
    attaches = [(no, name.strip()) for no, name in re.findall(r"attachfileDownload\('[^']+','(\d+)'\)[^>]*>\s*([^<]+)<", t)]
    ymd = post["event_ymd"]
    for _, name in attaches:  # 게시글 제목의 날짜는 가끔 오타가 있어 첨부 파일명 "240807(수) 인터넷집회.pdf" 의 날짜를 우선
        m = re.match(r"(\d{6})\(", name)
        if m:
            ymd = m[1]
            break
    pdf = next((a for a in attaches if a[1].lower().endswith(".pdf")), None)
    if not pdf:
        return None
    raw = _get(f"{BASE}/common/attachfile/attachfileDownload.do?attachNo={pdf[0]}")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(raw)
        f.flush()
        text = subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True, check=True).stdout
    return ymd, text


def _squash(s: str) -> str:
    """옛 PDF 표의 "종   로"처럼 글자 사이 공백이 벌어진 관할서명을 붙인다."""
    for st in STATIONS:
        s = re.sub(rf"(?<![가-힣]){r'\s+'.join(st)}(?![가-힣])", st, s)
    return s.rstrip()


def digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


def header_info(ymd: str, text: str, posted: dt.date | None, other_days_same_text: list[str]) -> dict:
    """작성 시각·공개 시각·날짜 점검 (ok / typo_ok / duplicate / mismatch)."""
    fdate = dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    head = text[:800]
    written = None
    m = WRITTEN_RE.search(head)
    if m:
        mo, d = int(m[2]), int(m[3])
        y = int(m[1]) if m[1] else fdate.year - (1 if mo > fdate.month else 0)  # 1월 행사의 12월 작성
        try:
            written = dt.datetime(y, mo, d, int(m[4]), int(m[5]))
        except ValueError:
            written = None
    if posted and (written is None or posted > written.date()):
        announced = dt.datetime.combine(posted, dt.time(23, 59))
    else:
        announced = written

    check = "ok"
    b = BODY_DATE_RE.search(head)
    if b and f"{b[1]}-{int(b[2]):02d}-{int(b[3]):02d}" != fdate.isoformat():
        if other_days_same_text:
            check = "duplicate"
        elif WEEKDAY[fdate.weekday()] == b[4]:
            check = "typo_ok"
        else:
            check = "mismatch"
    fmt = lambda x: x.isoformat(sep=" ", timespec="minutes") if x else ""
    return {"date": fdate.isoformat(), "date_check": check, "written_at": fmt(written),
            "posted_date": posted.isoformat() if posted else "", "announced_at": fmt(announced)}


def parse_blocks(text: str, date: str) -> list[dict]:
    """텍스트 → 집회 한 건씩 [{date, start, end, count, station, place, text}]."""
    entries, cur = [], None
    for line in (l.rstrip() for l in text.splitlines()):
        m = TIME_RE.search(line)
        if m and m.start() < 25:  # 시간은 표의 첫 칸에 온다
            if cur:
                entries.append(cur)
            cur = {"lines": [line[m.end():]], "start": f"{int(m[1]):02d}:{m[2]}", "end": f"{int(m[3]):02d}:{m[4]}"}
        elif cur is not None and line.strip():
            cur["lines"].append(line)
    if cur:
        entries.append(cur)

    rows = []
    for e in entries:
        body = [_squash(l) for l in e["lines"]]
        joined = "\n".join(body)
        m = COUNT_RE.search(joined) or COUNT_FALLBACK_RE.search(joined)
        count = int(m[1].replace(",", "")) if m else 0
        # 관할서: 인원 뒤쪽에서 <...>(장소 보충 설명) 밖에 있는 경찰서명. 행진은 "종로,\n남대문"처럼 여러 곳이 줄을 넘겨 나온다.
        tail = re.sub(r"<[^>]*>", " ", joined[m.end():] if m else joined)
        stations = [s for s in STATIONS if re.search(rf"(?<![가-힣]){s}(?![가-힣])", tail)]
        place = re.sub(r"\s+", " ", joined[: m.start()] if m else body[0]).strip()
        full = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*", " / ", joined)).strip()
        rows.append({"date": date, "start": e["start"], "end": e["end"], "count": count,
                     "station": "|".join(stations), "place": place[:200], "text": full[:600]})
    return rows
