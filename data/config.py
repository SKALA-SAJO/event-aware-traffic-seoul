"""
config/hubs.yaml · config/holidays_kr.yaml 로더.

거점(hub)·예측 단위(corridor = 거점 도로 × 방향)·예보 범위·게이트·드리프트 기준 같은 운영
파라미터는 설정 파일에서만 관리합니다. 학습·서빙·모니터링·수집·실험이 모두 이 모듈을 씁니다.
"""
import datetime as dt
import functools
import math
import os

import yaml


def _load_dotenv(path: str = ".env") -> None:
    """프로젝트 루트 .env 의 KEY=VALUE 를 환경변수로 (이미 설정된 값은 유지). 값은 어디에도 출력하지 않음."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

CONFIG_PATH = os.getenv("HUBS_CONFIG", "config/hubs.yaml")
HOLIDAYS_PATH = os.getenv("HOLIDAYS_CONFIG", "config/holidays_kr.yaml")
# scripts/map_corridors.py 가 만드는 corridor ↔ 링크·교통량 지점 매핑 (없으면 hubs.yaml 값만 사용)
LINKS_PATH = os.getenv("CORRIDOR_LINKS", "config/corridor_links.yaml")


@functools.lru_cache
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache
def corridor_links(path: str = LINKS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def section(name: str) -> dict:
    return load_config()[name]


def hubs(include_disabled: bool = False) -> dict[str, dict]:
    """{hub_id: {...}} - 기본은 enabled 거점만, 설정 파일 순서 그대로."""
    all_hubs = load_config()["hubs"]
    return {k: v for k, v in all_hubs.items() if include_disabled or v.get("enabled", True)}


def hub_ids() -> list[str]:
    return list(hubs())


def corridors(include_disabled: bool = False) -> dict[str, dict]:
    """{corridor_id: {..., "hub": hub_id, "length_km": 실제 길이}} - 설정 파일 순서 그대로."""
    out = {}
    for hub_id, h in hubs(include_disabled).items():
        for cid, c in (h.get("corridors") or {}).items():
            c = {**c, **corridor_links().get(cid, {})}
            links = c.get("links") or []
            length = sum(float(link.get("length_m") or 0) for link in links) / 1000 if links else 0.0
            out[cid] = {**c, "hub": hub_id, "hub_name": h["name"],
                        "length_km": length if length > 0 else float(c.get("length_km", 1.0))}
    return out


def corridor_ids() -> list[str]:
    return list(corridors())


def hub_of(corridor_id: str) -> str:
    return corridors(include_disabled=True)[corridor_id]["hub"]


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def nearest_hub(lat: float, lon: float, radius_km: float | None = None) -> str | None:
    """좌표가 어느 거점 반경 안에 있으면 그 거점 (KOPIS 공연시설·서울시 문화행사 좌표 → 거점 배정)."""
    radius_km = radius_km if radius_km is not None else section("collection")["event_match_radius_km"]
    best, best_d = None, float("inf")
    for hub_id, h in hubs(include_disabled=True).items():
        if not h.get("center"):
            continue
        d = haversine_km((lat, lon), tuple(h["center"]))
        if d < best_d:
            best, best_d = hub_id, d
    return best if best_d <= radius_km else None


@functools.lru_cache
def holidays(path: str = HOLIDAYS_PATH) -> frozenset[dt.date]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["holidays"]
    return frozenset(d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)) for d in raw)


FORECAST = section("forecast")
LOOKBACK = int(FORECAST["lookback_hours"])
HORIZON = int(FORECAST["horizon_hours"])
