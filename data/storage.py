"""
교통 관측치 · 원천 데이터 · 이벤트 · 돌발 · 예측 기록 저장소 (SQLite, data/traffic.db).

계층
    원천(raw)   link_speed   : TOPIS 링크별 시간 속도 (엑셀 / TrafficInfo 수집)
                spot_volume  : 지점·방향별 시간 교통량 (엑셀 / 교통량 이력 정보 VolInfo)
                links        : 링크 메타 (도로명·시종점·길이, 소통 링크 정보 API)
    집계        observations : corridor(거점 도로 × 방향) 시간 속도·교통량 ← 모델 입력·정답
    일정·상황   events       : 집회·공연·경기·축제 (공개 시각·상태 포함)
                incidents    : 돌발 정보 (사고·공사·통제) - 드리프트 판정 보조
    운영        predictions  : 발행한 예측 (정답 도착 후 오차 계산)

원천을 따로 두는 이유: corridor 구성(링크 목록)이나 집계 방식이 바뀌어도 다시 수집하지 않고
aggregate_observations() 로 재집계할 수 있고, 교통량 도착 지연 같은 원천 특성을 측정할 수 있습니다.

시각은 모두 KST naive datetime 을 'YYYY-MM-DD HH:MM:SS' 문자열로 저장합니다.
ts 는 "그 시간대의 시작 시각"입니다 (ts=08:00 → 08:00~08:59).
"""
import contextlib
import datetime as dt
import os
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = os.getenv("TRAFFIC_DB", "data/traffic.db")
TS_FMT = "%Y-%m-%d %H:%M:%S"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    corridor TEXT NOT NULL,
    ts       TEXT NOT NULL,
    speed    REAL,              -- km/h = 구간 길이 / Σ(링크 길이 / 링크 속도)
    volume   REAL,              -- 대/시, 해당 방향 교통량 지점 합 (결측은 NULL, 0 으로 저장하지 않음)
    source   TEXT,              -- topis | upload | synthetic | simulation
    PRIMARY KEY (corridor, ts)
);
CREATE TABLE IF NOT EXISTS link_speed (
    link_id    TEXT NOT NULL,
    ts         TEXT NOT NULL,
    speed      REAL,
    source     TEXT,            -- topis_excel | api
    fetched_at TEXT,
    PRIMARY KEY (link_id, ts)
);
CREATE TABLE IF NOT EXISTS spot_volume (
    spot          TEXT NOT NULL,
    ts            TEXT NOT NULL,
    io_type       INTEGER NOT NULL,  -- 유입/유출 구분
    volume        REAL,              -- 전 차로 합, 0 → NULL
    source        TEXT,
    first_seen_at TEXT,              -- 처음 0 이 아닌 값을 받은 시각 (API 도착 지연 측정용)
    PRIMARY KEY (spot, ts, io_type)
);
CREATE TABLE IF NOT EXISTS links (
    link_id    TEXT PRIMARY KEY,
    road_name  TEXT,
    st_node    TEXT,
    ed_node    TEXT,
    length_m   REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    hub               TEXT NOT NULL,
    type              TEXT NOT NULL,     -- rally | concert | sports | festival | marathon | other
    title             TEXT NOT NULL,
    start             TEXT NOT NULL,
    end               TEXT NOT NULL,
    expected_size     REAL,              -- 신고 인원 / 수용 인원 (사후 확정 관중 수는 넣지 않음)
    description       TEXT,
    source            TEXT,
    announced_at      TEXT,              -- 실제 공개(게시) 시각 - 정보 누수 방지 기준
    march             INTEGER DEFAULT 0,
    lane_control      INTEGER DEFAULT 0,
    created_at        TEXT,
    status            TEXT DEFAULT 'scheduled',  -- scheduled | review(검토 대기) | cancelled
    status_changed_at TEXT,              -- 취소 등 상태가 바뀐 시각 (그 이전 예측에는 '예정'으로 보였음)
    external_id       TEXT,              -- KOPIS mt20id / 문화행사 코드 등
    venue             TEXT,
    lat               REAL,
    lon               REAL,
    end_estimated     INTEGER DEFAULT 0, -- 종료 시각을 기본 소요시간으로 추정했으면 1
    runtime_text      TEXT,              -- 원문 공연 시간 (예: "2시간 30분(인터미션 20분 포함)")
    UNIQUE (hub, type, title, start)
);
CREATE TABLE IF NOT EXISTS incidents (
    acc_id       TEXT PRIMARY KEY,
    hub          TEXT,
    corridor     TEXT,
    link_id      TEXT,
    category     TEXT,           -- accident | construction | control | event | other
    type_code    TEXT,
    type_name    TEXT,
    start        TEXT,
    expected_end TEXT,
    last_seen    TEXT,           -- 마지막으로 API 에 보인 시각 (실제 해제 시각 추정)
    info         TEXT,
    source       TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    corridor      TEXT NOT NULL,
    issued_at     TEXT NOT NULL,
    target_ts     TEXT NOT NULL,
    horizon       INTEGER NOT NULL,
    predicted     REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT,
    PRIMARY KEY (corridor, issued_at, target_ts, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions (corridor, target_ts);
CREATE INDEX IF NOT EXISTS idx_events_hub ON events (hub, start);
"""

_EVENT_NEW_COLS = {
    "status": "TEXT DEFAULT 'scheduled'", "status_changed_at": "TEXT", "external_id": "TEXT", "venue": "TEXT",
    "lat": "REAL", "lon": "REAL", "end_estimated": "INTEGER DEFAULT 0", "runtime_text": "TEXT",
}


def fmt_ts(ts) -> str:
    return pd.Timestamp(ts).strftime(TS_FMT)


def _now() -> str:
    return dt.datetime.now().strftime(TS_FMT)


def _migrate(conn: sqlite3.Connection):
    """이전 스키마(거점 단위 observations/predictions)는 지우지 않고 *_legacy_hub 로 보존."""
    for table in ("observations", "predictions"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if "hub" in cols and "corridor" not in cols:
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy_hub")
    conn.execute("DROP INDEX IF EXISTS idx_pred_target")
    ev_cols = [r[1] for r in conn.execute("PRAGMA table_info(events)")]
    if ev_cols:
        for col, decl in _EVENT_NEW_COLS.items():
            if col not in ev_cols:
                conn.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")


@contextlib.contextmanager
def connect(db_path: str | None = None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _migrate(conn)
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT else v


def _where(query: str, params: list, col: str, values) -> str:
    if values:
        params += list(values)
        return query + f" AND {col} IN ({','.join('?' * len(values))})"
    return query


# ─────────────────────────────── observations (corridor) ───────────────────────────────

def upsert_observations(df: pd.DataFrame, source: str, db_path: str | None = None) -> int:
    """df: corridor, ts, speed, volume. 같은 (corridor, ts)는 덮어씁니다."""
    rows = [
        (r.corridor, fmt_ts(r.ts), _none(None if pd.isna(r.speed) else float(r.speed)),
         None if pd.isna(r.volume) else float(r.volume), source)
        for r in df.itertuples(index=False)
    ]
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO observations (corridor, ts, speed, volume, source) VALUES (?, ?, ?, ?, ?)", rows
        )
    return len(rows)


def load_observations(
    corridors: list[str] | None = None, start=None, end=None, exclude_sources: tuple = (), db_path: str | None = None
) -> pd.DataFrame:
    """start <= ts <= end 범위의 관측치 (corridor, ts, speed, volume)."""
    query, params = "SELECT corridor, ts, speed, volume FROM observations WHERE 1=1", []
    query = _where(query, params, "corridor", corridors)
    if exclude_sources:
        query += f" AND source NOT IN ({','.join('?' * len(exclude_sources))})"
        params += list(exclude_sources)
    if start is not None:
        query += " AND ts >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        query += " AND ts <= ?"
        params.append(fmt_ts(end))
    with connect(db_path) as conn:
        df = pd.read_sql_query(query + " ORDER BY corridor, ts", conn, params=params)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def observation_summary(db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT corridor, COUNT(*), MIN(ts), MAX(ts), SUM(volume IS NULL), "
            "GROUP_CONCAT(DISTINCT source) FROM observations GROUP BY corridor ORDER BY corridor"
        )
        return [
            {"corridor": c, "rows": n, "start": s, "end": e, "volume_missing": int(m or 0), "sources": src}
            for c, n, s, e, m, src in cur.fetchall()
        ]


def last_observation_ts(corridor: str | None = None, db_path: str | None = None) -> pd.Timestamp | None:
    with connect(db_path) as conn:
        if corridor:
            row = conn.execute("SELECT MAX(ts) FROM observations WHERE corridor = ?", (corridor,)).fetchone()
        else:
            row = conn.execute("SELECT MAX(ts) FROM observations").fetchone()
    return pd.Timestamp(row[0]) if row and row[0] else None


def delete_observations(source: str | None = None, db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        if source:
            return conn.execute("DELETE FROM observations WHERE source = ?", (source,)).rowcount
        return conn.execute("DELETE FROM observations").rowcount


# ─────────────────────────────── 원천: 링크 속도 · 지점 교통량 · 링크 정보 ───────────────────────────────

def upsert_link_speed(df: pd.DataFrame, source: str, db_path: str | None = None) -> int:
    """df: link_id, ts, speed"""
    now = _now()
    rows = [(str(r.link_id), fmt_ts(r.ts), None if pd.isna(r.speed) or r.speed <= 0 else float(r.speed), source, now)
            for r in df.itertuples(index=False)]
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO link_speed VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def upsert_spot_volume(df: pd.DataFrame, source: str, db_path: str | None = None) -> int:
    """df: spot, ts, io_type, volume. 0 은 결측(NULL). first_seen_at 은 처음 값이 들어온 시각을 유지."""
    now = _now()
    with connect(db_path) as conn:
        for r in df.itertuples(index=False):
            vol = None if pd.isna(r.volume) or r.volume <= 0 else float(r.volume)
            conn.execute(
                "INSERT INTO spot_volume (spot, ts, io_type, volume, source, first_seen_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (spot, ts, io_type) DO UPDATE SET "
                "volume = COALESCE(excluded.volume, spot_volume.volume), source = excluded.source, "
                "first_seen_at = COALESCE(spot_volume.first_seen_at, excluded.first_seen_at)",
                (str(r.spot), fmt_ts(r.ts), int(r.io_type), vol, source, now if vol is not None else None),
            )
    return len(df)


def upsert_links(rows: list[dict], db_path: str | None = None) -> int:
    now = _now()
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO links VALUES (?, ?, ?, ?, ?, ?)",
            [(str(r["link_id"]), r.get("road_name"), r.get("st_node"), r.get("ed_node"),
              r.get("length_m"), now) for r in rows],
        )
    return len(rows)


def link_lengths(db_path: str | None = None) -> dict[str, float]:
    with connect(db_path) as conn:
        return {k: v for k, v in conn.execute("SELECT link_id, length_m FROM links WHERE length_m > 0")}


def volume_arrival_lag(days: int = 7, db_path: str | None = None) -> pd.DataFrame:
    """교통량 이력 API: 시간대(ts) 종료 후 처음 값이 들어오기까지 걸린 시간 분포 (시간 단위)."""
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT spot, ts, first_seen_at FROM spot_volume WHERE source = 'api' AND first_seen_at IS NOT NULL "
            "AND ts >= ?", conn, params=[fmt_ts(pd.Timestamp.now() - pd.Timedelta(days=days))])
    if df.empty:
        return df
    lag = (pd.to_datetime(df["first_seen_at"]) - (pd.to_datetime(df["ts"]) + pd.Timedelta(hours=1)))
    df["lag_hours"] = lag / pd.Timedelta(hours=1)
    return df


def aggregate_observations(corridor_cfg: dict[str, dict], start, end, source: str = "topis",
                           min_coverage: float = 0.5, db_path: str | None = None) -> int:
    """
    원천 → corridor 관측치.
      속도  : 구간 길이 / Σ(링크 길이 / 링크 속도)  (거리 가중 조화평균 = 실제 통과 시간 기준)
              데이터가 있는 링크 길이가 전체의 min_coverage 미만인 시간은 결측
      교통량: 설정의 (지점, 유입/유출) 합. 모두 결측이면 결측
    """
    lengths = link_lengths(db_path)
    out = []
    with connect(db_path) as conn:
        for cid, c in corridor_cfg.items():
            links = {str(link["link_id"]): float(link.get("length_m") or lengths.get(str(link["link_id"]), 0))
                     for link in c.get("links") or []}
            links = {k: v for k, v in links.items() if v > 0}
            speed = pd.Series(dtype=float)
            if links:
                df = pd.read_sql_query(
                    f"SELECT link_id, ts, speed FROM link_speed WHERE link_id IN ({','.join('?' * len(links))}) "
                    "AND ts >= ? AND ts <= ? AND speed > 0", conn, params=[*links, fmt_ts(start), fmt_ts(end)])
                if len(df):
                    df["len"] = df["link_id"].map(links)
                    df["hours"] = df["len"] / 1000 / df["speed"]
                    g = df.groupby("ts").agg(cov=("len", "sum"), hours=("hours", "sum"))
                    g = g[g["cov"] >= min_coverage * sum(links.values())]
                    speed = g["cov"] / 1000 / g["hours"]
            vol = pd.Series(dtype=float)
            for s in c.get("volume_spots") or []:
                v = pd.read_sql_query(
                    "SELECT ts, volume FROM spot_volume WHERE spot = ? AND io_type = ? AND ts >= ? AND ts <= ?",
                    conn, params=[s["spot"], int(s.get("io_type", 1)), fmt_ts(start), fmt_ts(end)],
                ).set_index("ts")["volume"]
                vol = v if vol.empty else vol.add(v, fill_value=0)
            idx = speed.index.union(vol.index)
            if len(idx):
                out.append(pd.DataFrame({"corridor": cid, "ts": pd.to_datetime(idx),
                                         "speed": speed.reindex(idx).to_numpy(),
                                         "volume": vol.reindex(idx).to_numpy()}))
    if not out:
        return 0
    return upsert_observations(pd.concat(out, ignore_index=True), source=source, db_path=db_path)


# ─────────────────────────────────── events ───────────────────────────────────

_EVENT_COLS = [
    "hub", "type", "title", "start", "end", "expected_size", "description", "source", "announced_at",
    "march", "lane_control", "status", "status_changed_at", "external_id", "venue", "lat", "lon",
    "end_estimated", "runtime_text",
]
_TS_COLS = ("start", "end", "announced_at", "status_changed_at")
_BOOL_COLS = ("march", "lane_control", "end_estimated")


def upsert_events(events: list[dict], db_path: str | None = None) -> list[int]:
    """같은 (hub, type, title, start) 일정은 갱신합니다 (공개 정보 변경·취소 반영)."""
    now = _now()
    ids = []
    with connect(db_path) as conn:
        for ev in events:
            ev = {"status": "scheduled", **{k: v for k, v in ev.items() if v is not None}}
            values = []
            for c in _EVENT_COLS:
                v = _none(ev.get(c))
                if c in _TS_COLS and v is not None:
                    v = fmt_ts(v)
                if c in _BOOL_COLS:
                    v = int(bool(v))
                values.append(v)
            conn.execute(
                f"INSERT INTO events ({','.join(_EVENT_COLS)}, created_at) "
                f"VALUES ({','.join('?' * len(_EVENT_COLS))}, ?) "
                "ON CONFLICT (hub, type, title, start) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in _EVENT_COLS if c not in ("hub", "type", "title", "start")),
                values + [now],
            )
            row = conn.execute(
                "SELECT id FROM events WHERE hub=? AND type=? AND title=? AND start=?",
                (values[0], values[1], values[2], values[3]),
            ).fetchone()
            ids.append(row[0])
    return ids


def set_event_status(event_id: int, status: str, changed_at=None, db_path: str | None = None) -> bool:
    with connect(db_path) as conn:
        return conn.execute(
            "UPDATE events SET status = ?, status_changed_at = ? WHERE id = ?",
            (status, fmt_ts(changed_at or pd.Timestamp.now()), event_id),
        ).rowcount > 0


def load_events(
    hubs: list[str] | None = None, start=None, end=None, exclude_sources: tuple = (),
    statuses: tuple | None = None, db_path: str | None = None,
) -> pd.DataFrame:
    """[start, end] 와 겹치는 이벤트 (statuses=None 이면 모든 상태)."""
    query, params = "SELECT * FROM events WHERE 1=1", []
    query = _where(query, params, "hub", hubs)
    query = _where(query, params, "status", statuses)
    if exclude_sources:
        query += f" AND source NOT IN ({','.join('?' * len(exclude_sources))})"
        params += list(exclude_sources)
    if start is not None:
        query += " AND end >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        query += " AND start <= ?"
        params.append(fmt_ts(end))
    with connect(db_path) as conn:
        df = pd.read_sql_query(query + " ORDER BY start", conn, params=params)
    for c in _TS_COLS:
        df[c] = pd.to_datetime(df[c])
    df["status"] = df["status"].fillna("scheduled")
    return df


def delete_event(event_id: int, db_path: str | None = None) -> bool:
    with connect(db_path) as conn:
        return conn.execute("DELETE FROM events WHERE id = ?", (event_id,)).rowcount > 0


def delete_events(source: str | None = None, db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        if source:
            return conn.execute("DELETE FROM events WHERE source = ?", (source,)).rowcount
        return conn.execute("DELETE FROM events").rowcount


# ─────────────────────────────────── incidents ───────────────────────────────────

_INC_COLS = ["acc_id", "hub", "corridor", "link_id", "category", "type_code", "type_name",
             "start", "expected_end", "last_seen", "info", "source"]


def upsert_incidents(rows: list[dict], db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        for r in rows:
            vals = [fmt_ts(r[c]) if c in ("start", "expected_end", "last_seen") and r.get(c) is not None
                    else _none(r.get(c)) for c in _INC_COLS]
            conn.execute(
                f"INSERT INTO incidents ({','.join(_INC_COLS)}) VALUES ({','.join('?' * len(_INC_COLS))}) "
                "ON CONFLICT (acc_id) DO UPDATE SET "
                + ", ".join(f"{c} = COALESCE(excluded.{c}, incidents.{c})" for c in _INC_COLS[1:]),
                vals,
            )
    return len(rows)


def load_incidents(hubs: list[str] | None = None, start=None, end=None, db_path: str | None = None) -> pd.DataFrame:
    query, params = "SELECT * FROM incidents WHERE 1=1", []
    query = _where(query, params, "hub", hubs)
    if start is not None:
        query += " AND COALESCE(last_seen, expected_end, start) >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        query += " AND start <= ?"
        params.append(fmt_ts(end))
    with connect(db_path) as conn:
        df = pd.read_sql_query(query + " ORDER BY start", conn, params=params)
    for c in ("start", "expected_end", "last_seen"):
        df[c] = pd.to_datetime(df[c])
    return df


def delete_incidents(source: str | None = None, db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        if source:
            return conn.execute("DELETE FROM incidents WHERE source = ?", (source,)).rowcount
        return conn.execute("DELETE FROM incidents").rowcount


# ───────────────────────────────── predictions ─────────────────────────────────

def log_predictions(rows: list[dict], db_path: str | None = None) -> int:
    """rows: corridor, issued_at, target_ts, horizon, predicted, model_version."""
    now = _now()
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO predictions "
            "(corridor, issued_at, target_ts, horizon, predicted, model_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r["corridor"], fmt_ts(r["issued_at"]), fmt_ts(r["target_ts"]), int(r["horizon"]),
              float(r["predicted"]), r["model_version"], now) for r in rows],
        )
    return len(rows)


def load_prediction_errors(
    corridor: str, since, until=None, model_version: str | None = None, db_path: str | None = None
) -> pd.DataFrame:
    """정답(관측치)이 도착한 예측만 (target_ts, horizon, predicted, actual, model_version)."""
    query = (
        "SELECT p.target_ts, p.horizon, p.predicted, o.speed AS actual, p.model_version "
        "FROM predictions p JOIN observations o ON o.corridor = p.corridor AND o.ts = p.target_ts "
        "WHERE p.corridor = ? AND p.target_ts >= ? AND o.speed IS NOT NULL"
    )
    params: list = [corridor, fmt_ts(since)]
    if until is not None:
        query += " AND p.target_ts <= ?"
        params.append(fmt_ts(until))
    if model_version:
        query += " AND p.model_version = ?"
        params.append(model_version)
    with connect(db_path) as conn:
        df = pd.read_sql_query(query + " ORDER BY p.target_ts, p.horizon", conn, params=params)
    df["target_ts"] = pd.to_datetime(df["target_ts"])
    return df
