"""
드리프트 대응 시뮬레이션 - "알려진 이벤트"와 "예정에 없던 변화"를 구분하는지 확인합니다.

    1) normal        : 평소 패턴 48시간          → 드리프트 아님
    2) known_event   : 대규모 집회(행진·차로 통제)를 /events 로 미리 등록한 뒤, 그 시간대에
                       큰 정체가 발생한 48시간  → 이벤트 시간대 오차는 크지만 판정에서 제외 → 드리프트 아님
    3) construction  : 일정에 없는 도로 공사(차로 축소)로 출퇴근 정체가 심해진 5일
                       → 평상 시간대 오차 증가 → [WARN] drift → fine-tuning → 게이트 → 승격/유지

각 시나리오는 corridor 의 마지막 관측 시각 이후로 이어 붙여 /predict/batch-test 로 보냅니다.
"평소 패턴"은 DB의 최근 8주에서 이벤트·공휴일 시간을 뺀 요일·시각별 중앙값 + 잡음입니다.

사전 준비: 서버 실행 + Production 모델 (python serving_app/train_and_register.py)
실행: python scripts/simulate_drift.py [--corridor yeouido_up] [--scenario all|normal|known_event|construction]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import storage
from data.calendar_kr import holiday_mask
from data.config import hub_of
from data.events import event_mask

API_URL = os.getenv("API_URL", "http://localhost:8077")
rng = np.random.default_rng(int(os.getenv("SIM_SEED", "0")))


def usual_profile(hub: str) -> pd.DataFrame:
    end = storage.last_observation_ts(hub)
    obs = storage.load_observations(corridors=[hub], start=end - pd.Timedelta(weeks=8), end=end)
    events = storage.load_events(hubs=[hub_of(hub)], start=end - pd.Timedelta(weeks=8), end=end)
    hours = pd.DatetimeIndex(obs["ts"])
    key = lambda o: [o["ts"].dt.dayofweek, o["ts"].dt.hour]
    everything = obs.groupby(key(obs))[["speed", "volume"]].median()
    calm = obs[~(event_mask(hours, events) | holiday_mask(hours))]
    # 저녁 경기가 잦은 거점은 이벤트를 빼면 비는 요일·시각이 있음 → 전체 중앙값으로 채움
    return calm.groupby(key(calm))[["speed", "volume"]].median().reindex(everything.index).fillna(everything)


def normal_batch(hub: str, hours: int) -> pd.DataFrame:
    start = storage.last_observation_ts(hub) + pd.Timedelta(hours=1)
    ts = pd.date_range(start, periods=hours, freq="h")
    prof = usual_profile(hub).reindex(list(zip(ts.dayofweek, ts.hour)))
    ar = np.zeros(hours)
    for i in range(1, hours):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.03)
    return pd.DataFrame({
        "ts": ts,
        "speed": prof["speed"].to_numpy() * (1 + ar),
        "volume": prof["volume"].to_numpy() * (1 + rng.normal(0, 0.05, hours)),
    })


def send_batch(hub: str, batch: pd.DataFrame, label: str) -> dict:
    payload = {
        "corridor": hub,
        "observations": [
            {"ts": r.ts.isoformat(), "speed": round(float(r.speed), 1),
             "volume": None if pd.isna(r.volume) else round(float(r.volume))}
            for r in batch.itertuples(index=False)
        ],
    }
    resp = requests.post(f"{API_URL}/predict/batch-test", json=payload, timeout=900)
    resp.raise_for_status()
    result = resp.json()
    pts = pd.DataFrame(result["points"])
    err = (pts["predicted"] - pts["actual"]).abs()
    d = result["drift_check"]
    drift = d.get("drift", {})
    print(f"[{label}] {hub} {len(pts)}h  model={result['model_version']}  "
          f"MAE normal={err[~pts['known_event']].mean():.2f}  "
          f"event={err[pts['known_event']].mean() if pts['known_event'].any() else float('nan'):.2f}")
    ratio = drift.get("ratio")
    print(f"    drift_check: status={d['status']}  ratio={ratio if ratio is None else round(ratio, 2)} "
          f"(threshold {drift.get('threshold')})  kept={drift.get('n_hours')}h  excluded={drift.get('excluded_hours')}h")
    if d["status"] == "retrain_triggered":
        print(f"    retrain: promoted={d['promoted']} new_version={d.get('new_version')} "
              f"rmse_normal {d['production_rmse_normal']:.2f} -> {d['rmse_normal']:.2f}")
    return result


def scenario_normal(hub):
    return send_batch(hub, normal_batch(hub, 48), "normal")


def scenario_known_event(hub):
    batch = normal_batch(hub, 48)
    start = batch["ts"].iloc[0].normalize() + pd.Timedelta(hours=14)
    if start < batch["ts"].iloc[0]:
        start += pd.Timedelta(days=1)
    end = start + pd.Timedelta(hours=4)
    event = {
        "hub": hub_of(hub), "type": "rally", "title": f"[시뮬레이션] 대규모 집회 {start:%m/%d}",
        "start": start.isoformat(), "end": end.isoformat(), "expected_size": 80000,
        "description": "범국민 대회, 신고 인원 80,000명. 국회 → 여의도공원 방면 행진, 여의대로 전 차로 통제 예정.",
        "announced_at": (start.normalize() - pd.Timedelta(hours=6)).isoformat(),  # 전날 18시 공개
        "source": "simulation",
    }
    r = requests.post(f"{API_URL}/events", json=event, timeout=60)
    r.raise_for_status()
    ev = r.json()
    print(f"[known_event] 이벤트 등록: {ev['title']} {ev['start']}~{ev['end']} "
          f"(march={ev['march']}, lane_control={ev['lane_control']})")
    # 실제로는 모델 예상보다 더 심하게 막혔다고 가정 (이벤트 효과를 덜 맞히는 상황)
    c = (batch["ts"] - start) / pd.Timedelta(hours=1)
    impact = np.where((c >= -2) & (c <= 6), 0.6 * np.exp(-np.abs(c - 2) / 3), 0)
    batch["speed"] *= 1 - impact
    batch["volume"] *= 1 - 0.5 * impact
    return send_batch(hub, batch, "known_event")


def scenario_construction(hub):
    batch = normal_batch(hub, 24 * 5)
    h = batch["ts"].dt.hour.to_numpy() + 0.5
    weekday = batch["ts"].dt.dayofweek.to_numpy() < 5
    peak = np.exp(-0.5 * ((h - 8.3) / 1.5) ** 2) + np.exp(-0.5 * ((h - 18.5) / 1.8) ** 2)
    batch["speed"] *= 0.85 * (1 - 0.3 * peak * np.where(weekday, 1, 0.5))
    batch["volume"] *= 0.85
    print("[construction] 일정에 없는 도로 공사(차로 축소) 5일 주입 - 재학습이 걸리면 1~2분 걸립니다")
    return send_batch(hub, batch, "construction")


SCENARIOS = {"normal": scenario_normal, "known_event": scenario_known_event, "construction": scenario_construction}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor", default="yeouido_up")
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    args = ap.parse_args()
    for name in (SCENARIOS if args.scenario == "all" else [args.scenario]):
        SCENARIOS[name](args.corridor)
    print("결과 확인: 대시보드 '재학습 로그' 또는 logs/aiops.log")


if __name__ == "__main__":
    main()
