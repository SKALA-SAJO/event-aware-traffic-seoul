"""
파이프라인 검증용 합성 데이터 생성기 (source='synthetic').

실데이터(TOPIS 엑셀 / 열린데이터광장 API / 이벤트 일정)를 모으기 전에도 학습 → 게이트 →
서빙 → 드리프트 → 재학습 전 과정과 실험 스크립트를 돌려볼 수 있도록, 기획서의 가정을
그대로 반영한 시간 단위 데이터를 만듭니다. 여기서 나온 수치는 "모델·파이프라인이 의도대로
동작하는가"를 확인하는 용도이며, 실제 서울 교통에 대한 성능 주장에 쓰면 안 됩니다
(scripts/run_experiments.py 는 합성 데이터로 실행되면 리포트 상단에 이를 명시합니다).

반영한 가정
    - 평일 출퇴근 정체, 주말·공휴일 완화 (공휴일 = 정체가 풀리는 예정된 이벤트)
    - 집회: 진행 중 정체, 행진·차로 통제가 있으면 더 심하고 교통량은 오히려 감소
    - 공연·경기: 시작 전 도착 정체 < 종료 후 귀가 정체(감쇠), 교통량 증가
    - 신고 인원·수용 인원은 실제 참가 인원과 다름 (규모 피처는 노이즈가 있는 정보)
    - 방향성: 출근 시간엔 도심 방향, 퇴근 시간엔 외곽 방향이 더 막힘. 경기·공연은 시작 전에는
      행사장으로 들어가는 방향(inbound), 종료 후에는 나가는 방향(outbound)이 주로 막힘.
      집회·마라톤은 양방향(through) 모두 영향
    - 집회 공지 게시 시각은 대부분 전날 저녁이지만 일부는 당일 아침에 늦게 올라옴 (실제 게시일 저장의 근거)
    - 잠실 야구 일부 우천취소: 경기 3시간 전에 취소가 공지됨 (status=cancelled, status_changed_at)
    - 돌발 정보: corridor 별 무작위 사고(1~3시간 속도 저하, incidents 에만 기록)
    - 교통량 결측이 0으로 기록되는 구간 (무작위 + 장비 장애)
    - 예정에 없던 구조 변화: 상암 경기장→도심 방향 도로 공사(차로 축소)로 용량 감소 → 드리프트.
      공사는 돌발 정보에 "공사"로 등록됨 (드리프트 원인 표시용, 판정에서 제외하지는 않음)
"""
import numpy as np
import pandas as pd

from data.calendar_kr import holiday_mask
from data.event_nlp import extract_rule
from data.events import size_scale

PROFILES = {
    "gwanghwamun": {"free": 34.0, "cong": 0.55, "cap": 3200},
    "yeouido": {"free": 46.0, "cong": 0.50, "cap": 4200},
    "jamsil": {"free": 44.0, "cong": 0.55, "cap": 4600},
    "sangam": {"free": 52.0, "cong": 0.45, "cap": 3800},
    "gocheok": {"free": 40.0, "cong": 0.50, "cap": 3500},
}

DRIFT = {"corridor": "sangam_down", "start": "2026-08-17", "cong_mult": 1.6, "free_mult": 0.9, "vol_mult": 0.85}

# role 별 이벤트 반응 가중치: (시작 전 도착, 진행 중, 종료 후 귀가)
ROLE_WEIGHT = {"inbound": (1.0, 1.0, 0.3), "outbound": (0.3, 1.0, 1.0), "through": (0.65, 1.0, 0.65)}


def _g(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def demand(hours: pd.DatetimeIndex) -> np.ndarray:
    h = hours.hour.to_numpy() + 0.5
    dow = hours.dayofweek.to_numpy()
    hol = holiday_mask(hours)
    weekday = 0.12 + 0.75 * _g(h, 8.3, 1.3) + 0.45 * _g(h, 13.5, 3.5) + 0.85 * _g(h, 18.6, 1.6)
    weekday += np.where(dow == 4, 0.15 * _g(h, 20, 2), 0)
    saturday = 0.10 + 0.55 * _g(h, 14.5, 3.8) + 0.25 * _g(h, 19, 2)
    sunday = 0.08 + 0.45 * _g(h, 15, 3.8)
    d = np.where(dow < 5, weekday, np.where(dow == 5, saturday, sunday))
    d = np.where(hol, sunday * 0.9, d)
    season = 1 + 0.05 * np.sin(2 * np.pi * hours.dayofyear.to_numpy() / 365.0)
    return np.clip(d * season, 0.03, 1.1)


# ───────────────────────────────── 이벤트 일정 생성 ─────────────────────────────────

def _ev(hub, typ, title, start, hours, size, desc, announce_days=None, rng=None, effect_size=None,
        march=False, lane=False, cancelled=False):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    if typ == "rally":  # 경찰청 "오늘의 집회/시위": 대부분 전날 저녁 게시, 일부는 당일 아침 늦게 게시
        if rng.random() < 0.85:
            announced = start.normalize() - pd.Timedelta(hours=float(rng.uniform(3, 6)))
        else:
            announced = start.normalize() + pd.Timedelta(hours=float(rng.uniform(7, 9)))
    else:
        announced = start - pd.Timedelta(days=announce_days or 30)
    return {
        "hub": hub, "type": typ, "title": title, "start": start, "end": end, "expected_size": size,
        "description": desc, "source": "synthetic", "announced_at": announced,
        "status": "cancelled" if cancelled else "scheduled",
        "status_changed_at": start - pd.Timedelta(hours=3) if cancelled else None,
        # 아래 _ 필드는 "실제로 일어난 일" - 영향 생성에만 쓰고 DB에는 저장하지 않음
        "_effect_size": effect_size if effect_size is not None else size * float(rng.lognormal(-0.3, 0.4)),
        "_march": march, "_lane": lane, "_cancelled": cancelled,
    }


def _rally_desc(rng, hub, size, march, lane):
    org = rng.choice(["시민단체 연대", "노동조합 총연맹", "OO 범국민운동본부", "OO 대책위원회", "청년단체 연합"])
    topic = rng.choice(["노동 정책 규탄", "정부 정책 반대", "민생 촉구", "환경 정책 촉구", "제도 개선 요구"])
    place = {"gwanghwamun": "광화문광장·세종대로", "yeouido": "국회의사당 앞"}.get(hub, "일대")
    text = f"{org} 주최 {topic} 집회 ({place}), 신고 인원 {int(size):,}명."
    if march:
        text += " " + rng.choice(["세종대로 → 숭례문 방면 행진 예정.", "국회 → 여의도공원 방면 행진.", "종각 → 광화문 행진 예정."])
    if lane:
        text += " " + rng.choice(["세종대로 4개 차로 이용.", "일부 차로 통제.", "집회 구간 교통 통제 예정."])
    return text


def generate_events(hubs: list[str], start, end, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    events = []

    def rally(hub, day, p_weekend, p_weekday, median_size, weekend_hour=13):
        dow = day.dayofweek
        p = p_weekend if dow == 5 else p_weekday if dow < 5 else p_weekend * 0.3
        if rng.random() > p:
            return
        size = float(np.clip(rng.lognormal(np.log(median_size), 1.0), 300, 300_000))
        march, lane = rng.random() < 0.45, rng.random() < 0.4
        hour = weekend_hour if dow >= 5 else int(rng.integers(10, 18))
        events.append(_ev(hub, "rally", f"집회 {day:%m/%d}", day + pd.Timedelta(hours=hour), int(rng.integers(2, 6)),
                          size, _rally_desc(rng, hub, size, march, lane), rng=rng, march=march, lane=lane))

    for day in days:
        m, dow = day.month, day.dayofweek
        if "gwanghwamun" in hubs:
            rally("gwanghwamun", day, 0.7, 0.12, 4000)
            if dow == 6 and (m, (day.day - 1) // 7) in {(3, 2), (10, 3), (11, 0)}:
                events.append(_ev("gwanghwamun", "marathon", f"서울 도심 마라톤 {day:%Y-%m}", day + pd.Timedelta(hours=7), 5,
                                  30000, "마라톤 대회 진행에 따라 세종대로 전 차로 통제", announce_days=60, rng=rng,
                                  effect_size=30000, lane=True))
            if dow == 5 and m in (5, 6, 9, 10) and rng.random() < 0.25:
                events.append(_ev("gwanghwamun", "festival", f"서울광장 행사 {day:%m/%d}", day + pd.Timedelta(hours=11), 8,
                                  20000, "서울광장 문화 행사", announce_days=21, rng=rng))
        if "yeouido" in hubs:
            rally("yeouido", day, 0.15, 0.10, 2000, weekend_hour=14)
            if m == 4 and 3 <= day.day <= 12:
                events.append(_ev("yeouido", "festival", f"여의도 봄꽃축제 {day:%m/%d}", day + pd.Timedelta(hours=11), 10,
                                  50000, "여의서로 차량 통제, 봄꽃축제", announce_days=30, rng=rng, lane=True))
            if m == 10 and dow == 5 and day.day <= 7:
                events.append(_ev("yeouido", "festival", f"서울세계불꽃축제 {day:%Y}", day + pd.Timedelta(hours=19), 2,
                                  1_000_000, "불꽃축제 - 여의동로 전 차로 교통 통제", announce_days=60, rng=rng,
                                  effect_size=1_000_000, lane=True))
        if "jamsil" in hubs and (3, 22) <= (m, day.day) <= (10, 1) and dow != 0 and rng.random() < 0.88:
            hour = 18.5 if dow < 5 else 17 if dow == 5 else 14
            if (m in (7, 8)) and dow == 6:
                hour = 17
            events.append(_ev("jamsil", "sports", f"KBO 잠실 {day:%m/%d}", day + pd.Timedelta(hours=hour), 3.25,
                              25000, "KBO 정규시즌 잠실야구장 경기, 경기 종료 후 귀가 차량 혼잡",
                              announce_days=90, rng=rng, effect_size=float(rng.uniform(9000, 25000)),
                              cancelled=(rng.random() < 0.06)))
        if "jamsil" in hubs and dow >= 5 and 4 <= m <= 11 and rng.random() < 0.12:
            events.append(_ev("jamsil", "concert", f"종합운동장 공연 {day:%m/%d}", day + pd.Timedelta(hours=18), 3.5,
                              50000, "잠실종합운동장 주경기장 대형 콘서트", announce_days=60, rng=rng,
                              effect_size=float(rng.uniform(30000, 55000))))
        if "sangam" in hubs:
            if 3 <= m <= 11 and dow >= 5 and rng.random() < 0.28:
                events.append(_ev("sangam", "sports", f"K리그 FC서울 홈 {day:%m/%d}", day + pd.Timedelta(hours=19 if dow == 5 else 16.5), 2,
                                  66000, "K리그1 FC서울 홈경기 (서울월드컵경기장)", announce_days=60, rng=rng,
                                  effect_size=float(rng.uniform(12000, 50000))))
            if dow in (1, 3) and m in (3, 6, 9, 10, 11) and rng.random() < 0.08:
                events.append(_ev("sangam", "sports", f"축구 국가대표 A매치 {day:%m/%d}", day + pd.Timedelta(hours=20), 2,
                                  66000, "축구 국가대표 A매치 (서울월드컵경기장)", announce_days=45, rng=rng,
                                  effect_size=float(rng.uniform(55000, 66000))))
            if dow == 5 and 4 <= m <= 10 and rng.random() < 0.1:
                for k in range(2):  # 대형 공연은 이틀 연속
                    events.append(_ev("sangam", "concert", f"월드컵경기장 콘서트 {day:%m/%d}-{k + 1}",
                                      day + pd.Timedelta(days=k, hours=18), 3.5, 60000,
                                      "서울월드컵경기장 대형 콘서트, 공연 종료 후 월드컵로 혼잡 예상",
                                      announce_days=90, rng=rng, effect_size=float(rng.uniform(45000, 60000))))
        if "gocheok" in hubs and (3, 22) <= (m, day.day) <= (10, 1) and dow != 0 and rng.random() < 0.42:
            events.append(_ev("gocheok", "sports", f"KBO 고척 {day:%m/%d}", day + pd.Timedelta(hours=18.5 if dow < 5 else 17),
                              3.25, 16000, "KBO 정규시즌 고척스카이돔 경기", announce_days=90, rng=rng,
                              effect_size=float(rng.uniform(6000, 16000))))
    for ev in events:  # 등록 시점과 같은 방식으로 텍스트 속성을 추출해 둔다
        flags = extract_rule(ev["description"])
        ev["march"], ev["lane_control"] = flags["march"], flags["lane_control"]
    return events


# ───────────────────────────────── 관측치 생성 ─────────────────────────────────

def _event_effect(hours: pd.DatetimeIndex, events: list[dict], role: str):
    """(속도 감소율 r, 교통량 배수) - 여러 이벤트가 겹치면 곱으로 누적. role 로 방향별 반응을 다르게."""
    w_pre, w_act, w_post = ROLE_WEIGHT.get(role, ROLE_WEIGHT["through"])
    keep = np.ones(len(hours))
    vol_mult = np.ones(len(hours))
    centers = np.arange(len(hours)) + 0.5
    for ev in events:
        if ev["_cancelled"]:
            continue
        s = (pd.Timestamp(ev["start"]) - hours[0]) / pd.Timedelta(hours=1)
        e = (pd.Timestamp(ev["end"]) - hours[0]) / pd.Timedelta(hours=1)
        if e + 4 < 0 or s - 4 > len(hours):
            continue
        sc = size_scale(ev["_effect_size"], ev["type"])
        active = (centers > s) & (centers < e)
        pre = (centers <= s) & (centers > s - 3)
        post = (centers >= e) & (centers < e + 3)
        r = np.zeros(len(hours))
        v = np.ones(len(hours))
        if ev["type"] == "rally":  # 집회·행진은 양방향 모두
            k = sc * (0.22 + 0.18 * ev["_march"] + 0.2 * ev["_lane"])
            r += np.where(active, k, 0) + np.where(pre | post, 0.3 * k * np.exp(-np.minimum(np.abs(centers - np.where(pre, s, e)), 50)), 0)
            v *= np.where(active, 1 - 0.35 * sc * (ev["_lane"] or ev["_march"]), 1)
        elif ev["type"] == "marathon":
            r += np.where(active, 0.55, 0)
            v *= np.where(active, 0.4, 1)
        elif ev["type"] == "festival":
            r += np.where(active, 0.2 * sc * w_act, 0)
            r += np.where(pre, w_pre * 0.3 * sc * np.exp(np.minimum(centers - s, 0) / 1.5), 0)
            r += np.where(post, w_post * 0.45 * sc * np.exp(np.minimum(e - centers, 0) / 1.2), 0)
            v *= np.where(active & ev["_lane"], 0.7, 1)
        else:  # concert / sports: 도착은 inbound, 귀가는 outbound 에 몰림
            r += np.where(pre, w_pre * 0.25 * sc * np.exp(np.minimum(centers - s, 0) / 1.5), 0)
            r += np.where(active, w_act * 0.05 * sc, 0)
            r += np.where(post, w_post * 0.55 * sc * np.exp(np.minimum(e - centers, 0) / 1.2), 0)
            v *= 1 + np.where(pre | post, 0.6 * r, 0)
        keep *= 1 - np.clip(r, 0, 0.8)
        vol_mult *= v
    return 1 - keep, vol_mult


def _directional_demand(d: np.ndarray, hours: pd.DatetimeIndex, role: str) -> np.ndarray:
    """출근(오전)은 도심 방향, 퇴근(저녁)은 외곽 방향이 더 붐빔. inbound=행사장/도심 방향으로 가정."""
    h = hours.hour.to_numpy() + 0.5
    weekday = hours.dayofweek.to_numpy() < 5
    am, pm = _g(h, 8.3, 1.5), _g(h, 18.6, 1.8)
    tilt = {"inbound": 0.15 * am - 0.1 * pm, "outbound": -0.1 * am + 0.15 * pm}.get(role, 0.0)
    return np.clip(d * (1 + np.where(weekday, tilt, 0)), 0.03, 1.15)


def generate_incidents(corridor_cfg: dict, start, end, seed: int = 0, drift: dict | None = DRIFT):
    """돌발 정보: corridor 별 무작위 사고(월 2건 내외) + 드리프트용 도로 공사 1건."""
    rng = np.random.default_rng(seed + 2)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    n_days = (end - start).days
    out = []
    for cid, c in corridor_cfg.items():
        for k in range(int(rng.poisson(n_days / 15))):
            t0 = start + pd.Timedelta(hours=int(rng.integers(0, n_days * 24)))
            dur = float(rng.uniform(1, 3))
            out.append({
                "acc_id": f"syn-{cid}-{k}", "hub": c["hub"], "corridor": cid, "link_id": None,
                "category": "accident", "type_code": "A01", "type_name": "사고",
                "start": t0, "expected_end": t0 + pd.Timedelta(hours=dur), "last_seen": t0 + pd.Timedelta(hours=dur),
                "info": f"{c['road']} 차량 사고, 1개 차로 통제", "source": "synthetic",
                "_drop": float(rng.uniform(0.3, 0.5)),
            })
    if drift and drift["corridor"] in corridor_cfg:
        c = corridor_cfg[drift["corridor"]]
        t0 = pd.Timestamp(drift["start"])
        out.append({
            "acc_id": "syn-construction-1", "hub": c["hub"], "corridor": drift["corridor"], "link_id": None,
            "category": "construction", "type_code": "A04", "type_name": "공사",
            "start": t0, "expected_end": t0 + pd.Timedelta(days=180), "last_seen": end,
            "info": f"{c['road']} 지하 공사로 2개 차로 축소 (장기)", "source": "synthetic", "_drop": 0.0,
        })
    return out


def generate_observations(corridor_cfg: dict, start, end, events: list[dict], incidents: list[dict] | None = None,
                          seed: int = 0, drift: dict | None = DRIFT) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    hours = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")
    d = demand(hours)
    out = []
    for cid, c in corridor_cfg.items():
        hub, role = c["hub"], c.get("sim_role", c.get("role", "through"))
        p = PROFILES.get(hub, {"free": 45.0, "cong": 0.5, "cap": 4000})
        dh = _directional_demand(d * float(rng.uniform(0.95, 1.05)), hours, role)
        free = np.full(len(hours), p["free"] * float(rng.uniform(0.95, 1.05)))
        cong = np.full(len(hours), p["cong"])
        vol_m = np.ones(len(hours))
        if drift and drift["corridor"] == cid:
            after = hours >= pd.Timestamp(drift["start"])
            cong = np.where(after, cong * drift["cong_mult"], cong)
            free = np.where(after, free * drift["free_mult"], free)
            vol_m = np.where(after, drift["vol_mult"], 1.0)

        speed = free * (1 - np.clip(cong * dh ** 1.6, 0, 0.85))
        # AR(1) 잡음 + 일 단위 변동 (날씨 등)
        eps = rng.normal(0, 0.035, len(hours))
        ar = np.zeros(len(hours))
        for i in range(1, len(hours)):
            ar[i] = 0.85 * ar[i - 1] + eps[i]
        daily = np.repeat(rng.normal(0, 0.03, len(hours) // 24 + 1), 24)[: len(hours)]
        r, v_ev = _event_effect(hours, [e for e in events if e["hub"] == hub], role)
        acc = np.ones(len(hours))
        for inc in incidents or []:
            if inc["corridor"] == cid and inc["_drop"] > 0:
                acc[(hours >= inc["start"].floor("h")) & (hours < inc["expected_end"])] *= 1 - inc["_drop"]
        speed = np.clip(speed * (1 + ar + daily) * (1 - r) * acc, 4, 90)
        volume = p["cap"] * dh * vol_m * v_ev * (1 + rng.normal(0, 0.06, len(hours)))

        # 교통량 결측(원천 데이터에서는 0으로 기록됨): 무작위 2% + 장비 장애 2회
        vol_missing = rng.random(len(hours)) < 0.02
        for _ in range(2):
            s = int(rng.integers(0, len(hours) - 72))
            vol_missing[s : s + int(rng.integers(24, 72))] = True
        volume = np.where(vol_missing, np.nan, np.round(volume))
        out.append(pd.DataFrame({"corridor": cid, "ts": hours, "speed": np.round(speed, 1), "volume": volume}))
    return pd.concat(out, ignore_index=True)


def public_records(rows: list[dict]) -> list[dict]:
    """DB에 저장할 공개 정보만 남긴다 (실제 참가 인원 등 사후 정보 제거)."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
