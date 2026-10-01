"""
비교 실험 (ablation) - 실제로 효과가 있었던 방법만 최종 모델에 채택하기 위한 근거.

    실험 ① 피처 단계: 속도만 → +교통량 → +캘린더 → +이벤트
    실험 ② 이벤트 인코딩: flag / scale(규모) / decay(시간 감쇠) / text(+행진·차로 통제)
    실험 ③ 드리프트 대응: 고정 모델 / 주기적 재학습 / 드리프트 감지 시에만 재학습 (오프라인 walk-forward)
    실험 ④ 예측 단위: 거점 평균(양방향을 합친 하나의 시계열) vs 방향별(corridor) - 같은 corridor 실측으로 평가
    실험 ⑤ 이벤트 데이터 출처: 경기만 / +KOPIS 공연 / +서울시 문화행사 - 학습에 쓰는 이벤트만 바꾸고,
           이벤트 시간대 판정은 모든 출처를 합친 같은 기준으로 평가 (출처를 더해도 평가 시간이 바뀌지 않게)

평가: MAE·RMSE 를 전체·평상·이벤트 시간대, 거점별로. 비교 기준은 단순 예측법(지난주 같은 시각).
결과는 config 의 seeds (기본 3개) 평균 ± 표준편차로 보고합니다.

    python scripts/run_experiments.py                 # ①②③④ 전부 (CPU 기준 1시간 내외)
    python scripts/run_experiments.py --exp 1 2       # 일부만
    python scripts/run_experiments.py --quick         # 시드 1개·짧은 학습 - 동작 확인용

출력: reports/experiments/{exp1,…,exp5}.csv, summary.md  (+ MLflow experiment "traffic-ablation")
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data import storage
from data.config import HORIZON, corridor_ids, corridors, hub_ids, hub_of, section
from data.features import FeatureSpec, TrafficScaler, Windows, build_frames, make_windows
from serving_app import training as T
from serving_app.monitoring.drift_detector import evaluate_drift

OUT_DIR = os.getenv("EXP_OUT", "reports/experiments")  # 병렬 실행 시 시드별 폴더로 나눠 쓰고 합칠 때 사용
SEGMENTS = ["all", "normal", "event"]


class Runner:
    """실험 ①②가 같은 설정을 다시 학습하지 않도록 (spec.key, seed) 결과를 캐시."""

    def __init__(self, obs, events, epochs, use_mlflow):
        self.obs, self.events = obs, events
        self.hubs = corridor_ids()
        self.splits = T.time_splits(obs["ts"].max(), HORIZON)
        self.scaler = TrafficScaler().fit(obs[obs["ts"] < self.splits["scaler_end"]])
        self.cfg = section("train")
        self.epochs = epochs
        self.frames: dict[str, dict] = {}
        self.results: dict[tuple, dict] = {}
        self.models: dict[tuple, object] = {}
        self.use_mlflow = use_mlflow
        if use_mlflow:
            import mlflow

            from serving_app import train_and_register  # noqa: F401  (tracking URI 설정)

            mlflow.set_experiment("traffic-ablation")

    def frames_for(self, spec: FeatureSpec):
        if spec.event_mode not in self.frames:
            self.frames[spec.event_mode] = build_frames(self.obs, self.events, spec, self.scaler)
        return self.frames[spec.event_mode]

    def run(self, spec: FeatureSpec, seed: int) -> dict:
        key = (spec.key, seed)
        if key in self.results:
            return self.results[key]
        frames = self.frames_for(spec)
        w = {k: T.windows_for(frames, spec, self.splits[k]) for k in ("train", "val", "test")}
        model = T.fit(spec, w["train"], w["val"], seed, self.epochs, self.cfg["lr"], self.cfg["batch_size"])
        m = T.metrics(T.predict_speed(model, w["test"], self.scaler), w["test"])
        self.results[key] = m
        self.models[key] = model
        print(f"  {spec.key:<12} seed={seed:<5} normal={m['rmse_normal']:.3f} event={m['rmse_event']:.3f}")
        if self.use_mlflow:
            import mlflow

            with mlflow.start_run(run_name=f"{spec.key}-s{seed}"):
                mlflow.log_params({"config": spec.key, "seed": seed, "epochs": self.epochs})
                mlflow.log_metrics({k: v for k, v in m.items() if not np.isnan(v)})
        return m

    def naive(self) -> dict:
        spec = FeatureSpec(self.hubs, "calendar")
        return T.naive_metrics(T.windows_for(self.frames_for(spec), spec, self.splits["test"]))


def summarize(rows: list[dict], order: list[str]) -> pd.DataFrame:
    """config × seed 행 → config 별 평균 ± 표준편차."""
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith(("rmse_", "mae_"))]
    agg = df.groupby("config")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    return agg.reindex([o for o in order if o in agg.index])


def fmt_table(agg: pd.DataFrame, metrics: list[str], labels: dict) -> str:
    head = "| 설정 | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics) + "\n"
    body = ""
    for cfg, row in agg.iterrows():
        cells = []
        for m in metrics:
            mean, std = row.get(f"{m}__mean"), row.get(f"{m}__std")
            cells.append("-" if pd.isna(mean) else f"{mean:.3f}" + ("" if pd.isna(std) else f" ± {std:.3f}"))
        body += f"| {labels.get(cfg, cfg)} | " + " | ".join(cells) + " |\n"
    return head + body


# ───────────────────────────────── 실험 ①② ─────────────────────────────────

LABELS = {
    "naive": "단순 예측법 (지난주 같은 시각)",
    "speed": "① 속도만", "volume": "① + 교통량", "calendar": "① + 캘린더 (이벤트 정보 없음)",
    "full-flag": "② 이벤트 0/1 플래그", "full-scale": "② 규모 가중치",
    "full-decay": "② 규모 × 시간 감쇠", "full-text": "② 감쇠 + 텍스트(행진·차로 통제)",
}


def exp_specs(hubs, which):
    if which == 1:
        mode = section("train")["event_mode"]
        return [FeatureSpec(hubs, s) for s in ("speed", "volume", "calendar")] + [FeatureSpec(hubs, "full", mode)]
    return [FeatureSpec(hubs, "calendar")] + [FeatureSpec(hubs, "full", m) for m in ("flag", "scale", "decay", "text")]


def run_ablation(runner: Runner, which: int, seeds: list[int]) -> pd.DataFrame:
    print(f"[실험 {'①' if which == 1 else '②'}]")
    rows = []
    naive = runner.naive()
    rows.append({"config": "naive", "seed": 0, **naive})
    for spec in exp_specs(runner.hubs, which):
        for seed in seeds:
            rows.append({"config": spec.key, "seed": seed, **runner.run(spec, seed)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"exp{which}.csv"), index=False)
    return df


# ───────────────────────────────── 실험 ③ ─────────────────────────────────

def _clone(model):
    from tensorflow import keras

    c = keras.models.clone_model(model)
    c.set_weights(model.get_weights())
    return c


def run_drift_strategies(obs, events, seeds: list[int], epochs: int, walk_days: int, period_days: int) -> pd.DataFrame:
    """
    walk_days 일 전에서 학습을 멈춘 모델로, 이후를 하루씩 예측하며 세 전략을 비교한다.
      fixed    : 그대로 사용
      periodic : period_days 일마다 최근 fine_tune_days 일로 fine-tuning
      drift    : 매일 거점별 드리프트 판정(이벤트·공휴일 제외) → 감지 시에만 fine-tuning
    """
    print("[실험 ③] 드리프트 대응 전략 (walk-forward)")
    cfg, rcfg, dcfg = section("train"), section("retrain"), section("drift")
    hubs = corridor_ids()
    spec = FeatureSpec(hubs, cfg["feature_stage"], cfg["event_mode"])
    end = obs["ts"].max()
    cut = (end - pd.Timedelta(days=walk_days)).floor("D")
    splits = T.time_splits(cut, HORIZON, {**cfg, "test_days": 0})
    scaler = TrafficScaler().fit(obs[obs["ts"] < splits["scaler_end"]])
    frames = build_frames(obs, events, spec, scaler)
    days = pd.date_range(cut, end.floor("D") - pd.Timedelta(days=1), freq="D")
    ev_by_hub = {c: events[events["hub"] == hub_of(c)] for c in hubs}
    inc = storage.load_incidents()
    inc_by = {c: inc[(inc["corridor"] == c) | (inc["corridor"].isna() & (inc["hub"] == hub_of(c)))] for c in hubs}

    rows = []
    for seed in seeds:
        base = T.fit(spec, T.windows_for(frames, spec, splits["train"]), T.windows_for(frames, spec, splits["val"]),
                     seed, epochs, cfg["lr"], cfg["batch_size"])
        w_ref = make_windows(frames, spec, issue_start=splits["val"][0], issue_end=splits["val"][1])
        ref_rmse = T.reference_rmse(T.metrics(T.predict_speed(base, w_ref, scaler), w_ref), hubs)  # val 구간 기준

        for strategy in ("fixed", "periodic", "drift"):
            model, since, retrains, preds = _clone(base), cut, [], []
            for i, day in enumerate(days):
                w = make_windows(frames, spec, issue_start=day, issue_end=day + pd.Timedelta(hours=23))
                if not len(w):
                    continue
                preds.append((w, T.predict_speed(model, w, scaler)))

                retrain = False
                if strategy == "periodic" and (i + 1) % period_days == 0:
                    retrain = True
                elif strategy == "drift":
                    now = day + pd.Timedelta(hours=23 + HORIZON)
                    for hub in hubs:
                        err = _errors(preds, hub, since, now, dcfg["window_hours"])
                        st = evaluate_drift(err, ev_by_hub[hub], ref_rmse.get(hub), dcfg, inc_by[hub])
                        if st["status"] == "drift":
                            retrain = True
                            break
                if retrain:
                    t_end = day + pd.Timedelta(hours=23)
                    w_ft = make_windows(frames, spec, issue_start=t_end - pd.Timedelta(days=rcfg["fine_tune_days"]),
                                        issue_end=t_end - pd.Timedelta(hours=HORIZON))
                    weights = T.recency_weights(w_ft.issued_at, rcfg["recency_half_life_days"])
                    model = T.fit(spec, w_ft, None, seed, rcfg["epochs"], rcfg["lr"], model=model, sample_weight=weights)
                    since = t_end + pd.Timedelta(hours=1)
                    retrains.append(str(day.date()))

            w_all = Windows.concat([p[0] for p in preds])
            pred_all = np.concatenate([p[1] for p in preds])
            m = T.metrics(pred_all, w_all)
            row = {"config": strategy, "seed": seed, "n_retrain": len(retrains),
                   "retrain_days": ",".join(retrains), **m}
            rows.append(row)
            print(f"  {strategy:<9} seed={seed:<5} retrains={len(retrains):<3} normal={m['rmse_normal']:.3f} "
                  f"event={m['rmse_event']:.3f}  {retrains[:6]}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "exp3.csv"), index=False)
    return df


def _errors(preds, hub, since, now, window_hours) -> pd.DataFrame:
    """walk-forward 예측 중 현재 모델 버전(since 이후)·최근 window 의 (target_ts, predicted, actual)."""
    lo = max(pd.Timestamp(since), now - pd.Timedelta(hours=window_hours - 1))
    out = []
    for w, p in preds[-5:]:
        m = w.units == hub
        if not m.any():
            continue
        issued = pd.to_datetime(w.issued_at[m])
        for h in range(p.shape[1]):
            t = issued + pd.Timedelta(hours=h + 1)
            out.append(pd.DataFrame({"target_ts": t, "horizon": h + 1, "predicted": p[m, h], "actual": w.actual[m, h]}))
    if not out:
        return pd.DataFrame(columns=["target_ts", "horizon", "predicted", "actual"])
    df = pd.concat(out)
    df = df[(df["target_ts"] >= lo) & (df["target_ts"] <= now)].dropna(subset=["actual"])
    return df.reset_index(drop=True)


# ───────────────────────────────── 실험 ④ ─────────────────────────────────

def hub_level_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """corridor 관측치 → 거점 평균 시계열 (속도는 길이 가중 조화평균, 교통량은 합). 방향 정보를 버린 기존 방식."""
    cfg = corridors()
    o = obs.assign(hub=obs["corridor"].map(lambda c: cfg[c]["hub"]), length=obs["corridor"].map(lambda c: cfg[c]["length_km"]))
    o["hours"] = o["length"] / o["speed"]
    g = o.groupby(["hub", "ts"]).agg(length=("length", "sum"), hours=("hours", "sum"), volume=("volume", "sum"),
                                     n=("speed", "count"))
    g = g[g["n"] == o.groupby(["hub", "ts"]).size().reindex(g.index)]  # 양방향 모두 있는 시간만
    out = g.reset_index().rename(columns={"hub": "corridor"})
    out["speed"] = out["length"] / out["hours"]
    out.loc[out["volume"] <= 0, "volume"] = np.nan
    return out[["corridor", "ts", "speed", "volume"]]


def run_unit_comparison(runner: Runner, seeds: list[int]) -> pd.DataFrame:
    """
    같은 설정·시드로 (a) 거점 평균 모델과 (b) 방향별 모델을 학습하고, 둘 다 방향별 실측으로 평가한다.
    거점 평균 모델은 양방향에 같은 속도를 안내하게 되므로, 귀가 방향 정체를 얼마나 놓치는지가 드러난다.
    """
    print("[실험 ④] 예측 단위: 거점 평균 vs 방향별")
    cfg = section("train")
    mode = cfg["event_mode"]
    spec_c = FeatureSpec(runner.hubs, "full", mode)
    obs_h = hub_level_obs(runner.obs)
    spec_h = FeatureSpec(hub_ids(), "full", mode)
    scaler_h = TrafficScaler().fit(obs_h[obs_h["ts"] < runner.splits["scaler_end"]])
    frames_h = build_frames(obs_h, runner.events, spec_h, scaler_h)
    w_c = T.windows_for(runner.frames_for(spec_c), spec_c, runner.splits["test"])
    roles = np.array([corridors()[u].get("sim_role", "through") for u in w_c.units])
    rows = []
    for seed in seeds:
        w = {k: T.windows_for(frames_h, spec_h, runner.splits[k]) for k in ("train", "val", "test")}
        model_h = T.fit(spec_h, w["train"], w["val"], seed, runner.epochs, cfg["lr"], cfg["batch_size"])
        pred_h = T.predict_speed(model_h, w["test"], scaler_h)
        lookup = {(u, t): i for i, (u, t) in enumerate(zip(w["test"].units, w["test"].issued_at))}
        idx = np.array([lookup.get((hub_of(u), t), -1) for u, t in zip(w_c.units, w_c.issued_at)])
        keep = idx >= 0
        mapped = pred_h[idx[keep]]
        wc = w_c.subset(keep)
        m_hub = T.metrics(mapped, wc)
        runner.run(spec_c, seed)  # 실험 ①②에서 학습한 같은 방향별 모델을 재사용
        pred_c = T.predict_speed(runner.models[(spec_c.key, seed)], wc, runner.scaler)
        m_cor = T.metrics(pred_c, wc)
        for name, m, pred in (("hub_avg", m_hub, mapped), ("corridor", m_cor, pred_c)):
            err = pred - wc.actual
            row = {"config": name, "seed": seed, **m}
            for role in ("inbound", "outbound", "through"):
                r = (roles[keep] == role)[:, None] & wc.is_event & ~np.isnan(err)
                row[f"rmse_event_{role}"] = float(np.sqrt(np.mean(err[r] ** 2))) if r.any() else np.nan
            rows.append(row)
        print(f"  seed={seed:<5} event RMSE 거점평균={m_hub['rmse_event']:.3f} 방향별={m_cor['rmse_event']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "exp4.csv"), index=False)
    return df


# ───────────────────────────────── 실험 ⑤ ─────────────────────────────────

_SPORTS = ("kbo_schedule", "kleague_schedule", "csv", "manual")
SOURCE_SETS = {  # 이름 → 학습에 쓸 이벤트 고르기 (None = 전부)
    "sports": lambda e: e["source"].isin(_SPORTS),
    "sports+kopis10k": lambda e: e["source"].isin(_SPORTS) | ((e["source"] == "kopis") & (e["expected_size"] >= 10_000)),
    "sports+kopis": lambda e: e["source"].isin(_SPORTS + ("kopis",)),
    "sports+kopis+culture": None,
}
SOURCE_LABELS = {"calendar": "이벤트 정보 없음", "sports": "경기 (KBO·K리그)",
                 "sports+kopis10k": "경기 + KOPIS 1만 석 이상 공연장", "sports+kopis": "경기 + KOPIS 공연 (3천 석 이상)",
                 "sports+kopis+culture": "경기 + KOPIS + 서울시 문화행사"}


def run_source_comparison(obs, events, seeds: list[int], epochs: int) -> pd.DataFrame:
    print("[실험 ⑤] 이벤트 데이터 출처")
    spec = FeatureSpec(corridor_ids(), "full", section("train")["event_mode"])
    full = Runner(obs, events, epochs, use_mlflow=False)
    mask = {u: f["is_event"] for u, f in full.frames_for(spec).items()}
    rows = [{"config": "naive", "seed": 0, **full.naive()}]
    for seed in seeds:
        rows.append({"config": "calendar", "seed": seed, **full.run(FeatureSpec(corridor_ids(), "calendar"), seed)})
    for name, pick in SOURCE_SETS.items():
        ev = events if pick is None else events[pick(events)]
        r = full if pick is None else Runner(obs, ev, epochs, use_mlflow=False)
        for u, f in r.frames_for(spec).items():
            f["is_event"] = mask[u]  # 평가 기준 통일
        print(f"  {name}: 학습 이벤트 {len(ev)}건")
        for seed in seeds:
            rows.append({"config": name, "seed": seed, "n_train_events": len(ev), **r.run(spec, seed)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "exp5.csv"), index=False)
    return df


# ───────────────────────────────── 리포트 ─────────────────────────────────

def write_summary(results: dict, seeds, epochs, source: str):
    lines = [f"# 비교 실험 결과 ({dt.datetime.now():%Y-%m-%d %H:%M})", ""]
    if "synthetic" in source:
        lines += ["> **주의: 합성 데이터(source=synthetic)로 실행한 결과입니다.** 파이프라인·실험 설계가 의도대로 "
                  "동작하는지 확인하는 용도이며, 실제 서울 교통에 대한 성능 근거로 쓰면 안 됩니다.", ""]
    else:
        n_by_src = storage.load_events().groupby("source").size().to_dict()
        lines += ["> 실데이터: TOPIS 링크 속도·지점 교통량 엑셀 + 이벤트 " + ", ".join(f"{k} {v}건" for k, v in n_by_src.items())
                  + ". 가정 - 과거 경기 일정 공개 시각 = 경기 7일 전, 우천취소 공지 = 시작 2시간 전, 규모 = 경기장·공연장 "
                  "수용 인원(관중 수 아님). KOPIS 공개 시각 = 첫 공연 전 마지막 갱신 시각(없으면 30일 전 가정), "
                  "서울시 문화행사 공개 시각 = 등록일. 집회 일정은 아직 없습니다.", ""]
    lines += [f"- 데이터 출처: `{source}`  ·  시드: {seeds}  ·  epochs: {epochs}",
              "- 지표: 향후 1~6시간 전체 예보 시차의 통행속도 오차 (km/h). 평상 = 이벤트·공휴일이 아닌 시간, "
              "이벤트 = 등록된 이벤트 시작 전 ~ 종료 후", ""]
    metrics = ["rmse_all", "rmse_normal", "rmse_event", "mae_event"]

    for which, title in ((1, "실험 ① 피처 단계별 추가"), (2, "실험 ② 이벤트 인코딩 방식")):
        if which in results:
            df = results[which]
            order = ["naive"] + [s.key for s in exp_specs(corridor_ids(), which)]
            agg = summarize(df.to_dict("records"), order)
            lines += [f"## {title}", "", fmt_table(agg, metrics, LABELS)]
            hub_cols = [f"rmse_event_{h}" for h in corridor_ids()]
            lines += ["corridor 별 이벤트 시간대 RMSE", "", fmt_table(agg, hub_cols, LABELS)]

    if 2 in results:
        agg = summarize(results[2].to_dict("records"), [s.key for s in exp_specs(corridor_ids(), 2)])
        cal = agg.loc["calendar"]
        rows = ["| 설정 | 이벤트 RMSE 변화 | 평상 RMSE 변화 | 판정 |", "|---|---|---|---|"]
        for key, r in agg.drop(index="calendar").iterrows():
            d_ev = r["rmse_event__mean"] - cal["rmse_event__mean"]
            d_no = r["rmse_normal__mean"] - cal["rmse_normal__mean"]
            ev_noise = max(r["rmse_event__std"] or 0, cal["rmse_event__std"] or 0)
            no_noise = max(r["rmse_normal__std"] or 0, cal["rmse_normal__std"] or 0)
            verdict = "이벤트 개선 (시드 편차 초과)" if d_ev < -ev_noise else "이벤트 개선 불확실"
            if d_no > no_noise:
                verdict += " · 평상 악화 (시드 편차 초과)"
            rows.append(f"| {LABELS.get(key, key)} | {d_ev:+.3f} | {d_no:+.3f} | {verdict} |")
        best = agg.drop(index="calendar")["rmse_event__mean"].idxmin()
        b = agg.loc[best]
        chosen = FeatureSpec(corridor_ids(), "full", section("train")["event_mode"]).key
        c = agg.loc[chosen] if chosen in agg.index else b
        std = c["rmse_normal__std"]
        gate = c["rmse_normal__mean"] + 2 * (0.0 if pd.isna(std) else std)  # 시드 1개면 σ 없음 → 평균만
        naive_normal = results[2].query("config == 'naive'")["rmse_normal"].mean()
        lines += ["## 채택 판단", "", "'이벤트 정보 없음'(캘린더 단계) 대비 변화 (km/h, 음수 = 개선)", "", *rows, "",
                  f"- 이벤트 시간대 RMSE 최소: **{LABELS.get(best, best)}** (`event_mode: {best.split('-', 1)[1]}`)",
                  f"- 현재 설정 `{chosen}` 의 평상 RMSE {c['rmse_normal__mean']:.3f} — 단순 예측법 {naive_normal:.3f} 보다 낮은지가 게이트 ① 기본 기준",
                  f"- 배포 게이트 ① 권장값 (현재 설정의 평상 RMSE 평균 + 2σ{', 시드 1개라 σ 미반영' if pd.isna(std) else ''}): "
                  f"**{gate:.2f} km/h** → `config/hubs.yaml` gates.normal_rmse_max",
                  "- 평상 시간대가 시드 편차 이상 나빠지는 설정은 이벤트 개선폭과 함께 보고 채택 여부를 결정하세요 "
                  "(이벤트 피처가 평상 시간대에도 0이 아닌 값을 남기거나, 모델 용량을 이벤트 패턴에 나눠 쓰는 영향).",
                  ""]

    if 3 in results:
        df = results[3]
        # drift 는 임계값별 변형(drift-1.5, drift-2.0 …)을 합쳐 보고할 수 있음
        drift_cfgs = sorted(c for c in df["config"].unique() if str(c).startswith("drift"))
        agg = summarize(df.to_dict("records"), ["fixed", "periodic", *drift_cfgs])
        n = df.groupby("config")["n_retrain"].mean()
        labels = {"fixed": "고정 모델", "periodic": "주기적 재학습", "drift": "드리프트 감지 시 재학습",
                  **{c: f"드리프트 감지 시 재학습 (임계 {c.split('-', 1)[1]})" for c in drift_cfgs if "-" in c}}
        lines += ["## 실험 ③ 드리프트 대응 전략 (walk-forward)", "",
                  fmt_table(agg, ["rmse_all", "rmse_normal", "rmse_event"] + [f"rmse_normal_{h}" for h in corridor_ids()],
                            labels),
                  "평균 재학습 횟수: " + ", ".join(f"{k} {v:.1f}회" for k, v in n.items()), ""]
        for r in df[df["config"].isin(drift_cfgs)].itertuples():
            lines.append(f"- {r.config} seed {r.seed}: 드리프트 재학습일 {r.retrain_days or '-'}")
        best = agg["rmse_normal__mean"].idxmin()
        lines += ["", f"- 평상 RMSE 최소: **{labels.get(best, best)}** → `config/hubs.yaml` retrain.on_drift "
                      f"({'alert + 주기적 fine-tuning' if best == 'periodic' else 'retrain'})", ""]

    if 4 in results:
        agg = summarize(results[4].to_dict("records"), ["hub_avg", "corridor"])
        lines += ["## 실험 ④ 예측 단위: 거점 평균 vs 방향별 (방향별 실측으로 평가)", "",
                  fmt_table(agg, ["rmse_normal", "rmse_event", "rmse_event_inbound", "rmse_event_outbound",
                                  "rmse_event_through"],
                            {"hub_avg": "거점 평균 (양방향 합침)", "corridor": "방향별 (corridor)"}),
                  "- inbound/outbound/through 는 hubs.yaml 의 sim_role(합성 데이터용 방향 가정) 기준 분류입니다. "
                  "실데이터 corridor 는 TOPIS 상행/하행이라 이 열은 참고용입니다.", ""]

    if 5 in results:
        df = results[5]
        order = ["naive", "calendar", *SOURCE_SETS]
        agg = summarize(df.to_dict("records"), order)
        hubs_ev = [f"rmse_event_{c}" for c in corridor_ids()]
        lines += ["## 실험 ⑤ 이벤트 데이터 출처 (평가 이벤트 시간은 모든 출처 기준으로 동일)", "",
                  fmt_table(agg, ["rmse_all", "rmse_normal", "rmse_event", "mae_event"],
                            {**LABELS, **SOURCE_LABELS}),
                  "구간별 이벤트 시간 RMSE", "",
                  fmt_table(agg, hubs_ev, {**LABELS, **SOURCE_LABELS}),
                  "- 출처를 더할수록 학습 때 '이벤트인 줄 모르던' 정체가 이벤트로 설명되는지를 봅니다. "
                  "평가 기준이 같으므로 행끼리 바로 비교할 수 있습니다.", ""]

    path = os.path.join(OUT_DIR, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"→ {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--walk-days", type=int, default=75, help="실험 ③ walk-forward 기간")
    ap.add_argument("--period-days", type=int, default=14, help="실험 ③ 주기적 재학습 간격")
    ap.add_argument("--quick", action="store_true", help="시드 1개·3 epoch 동작 확인")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="학습 없이 저장된 exp*.csv 로 summary.md 만 다시 생성")
    args = ap.parse_args()

    cfg = section("train")
    seeds = args.seeds or (cfg["seeds"][:1] if args.quick else cfg["seeds"])
    epochs = args.epochs or (3 if args.quick else cfg["epochs"])
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.report_only:
        results = {w: pd.read_csv(os.path.join(OUT_DIR, f"exp{w}.csv")) for w in args.exp
                   if os.path.exists(os.path.join(OUT_DIR, f"exp{w}.csv"))}
        write_summary(results, seeds, epochs, T.data_source())
        return

    obs, events = T.load_data()
    source = T.data_source()
    results = {}
    if {1, 2, 4} & set(args.exp):
        runner = Runner(obs, events, epochs, use_mlflow=not args.no_mlflow)
        for which in (1, 2):
            if which in args.exp:
                results[which] = run_ablation(runner, which, seeds)
        if 4 in args.exp:
            results[4] = run_unit_comparison(runner, seeds)
    if 5 in args.exp:
        results[5] = run_source_comparison(obs, events, seeds, epochs)
    if 3 in args.exp:
        results[3] = run_drift_strategies(obs, events, seeds[:1] if args.quick else seeds, epochs,
                                          args.walk_days, args.period_days)
    write_summary(results, seeds, epochs, source)


if __name__ == "__main__":
    main()
