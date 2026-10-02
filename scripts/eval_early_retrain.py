"""
조기 재학습(C안)이 성능에 도움이 되는지 오프라인 walk-forward 로 검증 - 실험 ③ 의 확장.

실험 ③(scripts/run_experiments.py)과 같은 데이터·분할·판정을 쓰고, 운영에 들어간 C안 규칙을 전략으로 추가합니다.

    fixed         : 재학습 없음
    periodic      : period_days(14) 일마다 fine-tuning                              (실험 ③ 과 동일)
    immediate     : 하루 한 번 판정해 어느 corridor 든 감지되면 즉시 fine-tuning       (실험 ③ 의 drift 와 동일)
    hybrid        : periodic + 조기 재학습 = 같은 corridor 가 연속 consecutive_days 일 감지되면 신청,
                    마지막 학습 후 cooldown_days 가 지나면 실행 (신청은 request_expire_days 일 뒤 만료)
    periodic_gate / hybrid_gate : 위와 같고, 운영처럼 배포 게이트 ③(마지막 eval_days 일 평상 RMSE 가 현재 모델보다
                    나빠지지 않을 때만 교체)을 적용. 탈락해도 마지막 학습 시각은 갱신(쿨다운), 모델은 그대로

평가: walk-forward 전 구간의 평상/이벤트 시간대 RMSE (낮을수록 좋음), 재학습 횟수, 시드별 짝 비교.
빠진 것: "최근 공사" 조건은 평가하지 않습니다 (DB 의 incidents 가 비어 있어 발동할 수 없음).
결과는 시드 편차(README 기준 ±0.02~0.03 수준)와 함께 읽어야 하며, 이 평가는 현재 데이터 75일 구간 하나뿐입니다.

    python scripts/eval_early_retrain.py                  # 시드 3개 (CPU 기준 시드당 20~30분 추정)
    python scripts/eval_early_retrain.py --quick          # 시드 1개·3 epoch, 동작 확인용
출력: reports/early_retrain/eval.csv, report.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import run_experiments as X
from data import storage
from data.config import HORIZON, corridor_ids, hub_of, section
from data.features import FeatureSpec, TrafficScaler, Windows, build_frames, make_windows
from serving_app import training as T
from serving_app.monitoring.drift_detector import evaluate_drift

OUT_DIR = os.getenv("EARLY_EVAL_OUT", "reports/early_retrain")
STRATEGIES = ["fixed", "periodic", "immediate", "hybrid", "periodic_gate", "hybrid_gate"]
LABELS = {"fixed": "고정", "periodic": "14일 주기", "immediate": "감지 즉시", "hybrid": "C안 (주기 + 조기)",
          "periodic_gate": "14일 주기 + 게이트", "hybrid_gate": "C안 + 게이트"}


def retrain_step(model, frames, spec, scaler, t_end, seed, rcfg, gated: bool):
    """fine-tuning 한 번. 반환: (새 모델 또는 기존 모델, 교체 여부)."""
    ft_start = t_end - pd.Timedelta(days=rcfg["fine_tune_days"])
    if not gated:  # 실험 ③ 과 같은 방식: 최근 fine_tune_days 일 전부로 학습, 항상 교체
        w = make_windows(frames, spec, issue_start=ft_start, issue_end=t_end - pd.Timedelta(hours=HORIZON))
        weights = T.recency_weights(w.issued_at, rcfg["recency_half_life_days"])
        return T.fit(spec, w, None, seed, rcfg["epochs"], rcfg["lr"], model=model, sample_weight=weights), True

    # 운영(fine_tune)과 같은 분할: 마지막 eval_days 일은 평가용으로 남기고 그 앞으로 학습
    last = t_end - pd.Timedelta(hours=HORIZON)
    eval_start = t_end - pd.Timedelta(days=rcfg["eval_days"]) - pd.Timedelta(hours=HORIZON)
    train_end = eval_start - pd.Timedelta(hours=HORIZON + 1)
    w_tr = make_windows(frames, spec, issue_start=ft_start, issue_end=train_end)
    w_ev = make_windows(frames, spec, issue_start=eval_start, issue_end=last)
    weights = T.recency_weights(w_tr.issued_at, rcfg["recency_half_life_days"])
    cand = T.fit(spec, w_tr, None, seed, rcfg["epochs"], rcfg["lr"], model=X._clone(model), sample_weight=weights)
    cur = T.metrics(T.predict_speed(model, w_ev, scaler), w_ev)["rmse_normal"]
    new = T.metrics(T.predict_speed(cand, w_ev, scaler), w_ev)["rmse_normal"]
    return (cand, True) if new <= cur else (model, False)  # NaN 이면 False → 탈락 (운영 게이트 ③ 과 같음)


def run(obs, events, seeds, epochs, walk_days, early_cfg, period_days):
    cfg, rcfg, dcfg = section("train"), section("retrain"), section("drift")
    hubs = corridor_ids()
    spec = FeatureSpec(hubs, cfg["feature_stage"], cfg["event_mode"])
    end = obs["ts"].max()
    cut = (end - pd.Timedelta(days=walk_days)).floor("D")
    splits = T.time_splits(cut, HORIZON, {**cfg, "test_days": 0})
    scaler = TrafficScaler().fit(obs[obs["ts"] < splits["scaler_end"]])
    frames = build_frames(obs, events, spec, scaler)
    days = pd.date_range(cut, end.floor("D") - pd.Timedelta(days=1), freq="D")
    ev_by = {c: events[events["hub"] == hub_of(c)] for c in hubs}
    inc = storage.load_incidents()
    inc_by = {c: inc[(inc["corridor"] == c) | (inc["corridor"].isna() & (inc["hub"] == hub_of(c)))] for c in hubs}

    rows = []
    for seed in seeds:
        base = T.fit(spec, T.windows_for(frames, spec, splits["train"]), T.windows_for(frames, spec, splits["val"]),
                     seed, epochs, cfg["lr"], cfg["batch_size"])
        w_ref = make_windows(frames, spec, issue_start=splits["val"][0], issue_end=splits["val"][1])
        ref = T.reference_rmse(T.metrics(T.predict_speed(base, w_ref, scaler), w_ref), hubs)

        for strategy in STRATEGIES:
            gated = strategy.endswith("_gate")
            kind = strategy.removesuffix("_gate")
            model, since, preds = X._clone(base), cut, []
            streak = {c: 0 for c in hubs}
            last_idx, request_idx = -1, None
            log = {"periodic": [], "early": [], "rejected": []}
            for i, day in enumerate(days):
                w = make_windows(frames, spec, issue_start=day, issue_end=day + pd.Timedelta(hours=23))
                if not len(w):
                    continue
                preds.append((w, T.predict_speed(model, w, scaler)))

                trigger = None
                due_periodic = i - last_idx >= period_days
                if kind == "periodic" and due_periodic:
                    trigger = "periodic"
                elif kind in ("immediate", "hybrid"):
                    now = day + pd.Timedelta(hours=23 + HORIZON)
                    drifted = []
                    for c in hubs:
                        st = evaluate_drift(X._errors(preds, c, since, now, dcfg["window_hours"]), ev_by[c],
                                            ref.get(c), dcfg, inc_by[c])
                        streak[c] = streak[c] + 1 if st["status"] == "drift" else 0
                        if st["status"] == "drift":
                            drifted.append(c)
                    if kind == "immediate" and drifted:
                        trigger = "early"
                    elif kind == "hybrid":
                        if request_idx is None and any(streak[c] >= early_cfg["consecutive_days"] for c in hubs):
                            request_idx = i
                        if request_idx is not None and i - request_idx > early_cfg["request_expire_days"]:
                            request_idx = None
                        if due_periodic:
                            trigger = "periodic"
                        elif request_idx is not None and i - last_idx >= early_cfg["cooldown_days"]:
                            trigger = "early"
                if trigger:
                    t_end = day + pd.Timedelta(hours=23)
                    model, replaced = retrain_step(model, frames, spec, scaler, t_end, seed, rcfg, gated)
                    last_idx, request_idx = i, None
                    if replaced:
                        since = t_end + pd.Timedelta(hours=1)
                        streak = {c: 0 for c in hubs}  # 새 버전은 이전 버전 오차와 이어지지 않음
                        log[trigger].append(str(day.date()))
                    else:
                        log["rejected"].append(str(day.date()))

            m = T.metrics(np.concatenate([p[1] for p in preds]), Windows.concat([p[0] for p in preds]))
            rows.append({"config": strategy, "seed": seed, "rmse_normal": m["rmse_normal"], "rmse_event": m["rmse_event"],
                         "rmse_all": m["rmse_all"], "n_periodic": len(log["periodic"]), "n_early": len(log["early"]),
                         "n_rejected": len(log["rejected"]), "early_days": ",".join(log["early"]),
                         "rejected_days": ",".join(log["rejected"])})
            print(f"  seed={seed:<5} {strategy:<14} normal={m['rmse_normal']:.4f} event={m['rmse_event']:.4f} "
                  f"periodic={len(log['periodic'])} early={len(log['early'])} rejected={len(log['rejected'])} "
                  f"{log['early'][:6]}", flush=True)
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, meta: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "eval.csv"), index=False)
    agg = df.groupby("config")[["rmse_normal", "rmse_event", "rmse_all", "n_periodic", "n_early", "n_rejected"]].agg(["mean", "std"])
    lines = [f"# 조기 재학습(C안) 오프라인 평가 - walk-forward {meta['walk_days']}일, 시드 {meta['seeds']}, "
             f"epochs {meta['epochs']}", "",
             f"- 데이터: {meta['source']} (마지막 관측 {meta['end']}), 조기 재학습 규칙: 연속 {meta['early']['consecutive_days']}일 · "
             f"쿨다운 {meta['early']['cooldown_days']}일 · 신청 만료 {meta['early']['request_expire_days']}일, 주기 {meta['period_days']}일",
             "- 공사 조건은 평가하지 않음 (incidents 비어 있음). RMSE 단위 km/h, 낮을수록 좋음", "",
             "| 전략 | 평상 RMSE | 이벤트 RMSE | 전체 RMSE | 주기 재학습 | 조기 재학습 | 게이트 탈락 |", "|---|---|---|---|---|---|---|"]

    def cell(r, k, p=3):
        return f"{r[(k, 'mean')]:.{p}f}" + ("" if pd.isna(r[(k, 'std')]) else f" ± {r[(k, 'std')]:.{p}f}")

    for s in STRATEGIES:
        if s in agg.index:
            r = agg.loc[s]
            lines.append(f"| {LABELS[s]} | {cell(r, 'rmse_normal')} | {cell(r, 'rmse_event')} | {cell(r, 'rmse_all')} | "
                         f"{cell(r, 'n_periodic', 1)} | {cell(r, 'n_early', 1)} | {cell(r, 'n_rejected', 1)} |")

    piv = df.pivot(index="seed", columns="config", values="rmse_normal")
    lines += ["", "## 평상 RMSE 시드별 짝 비교 (A - B, 음수 = A가 더 좋음)", "", "| A | B | " +
              " | ".join(f"seed {s}" for s in piv.index) + " | 평균 | A가 좋은 시드 |", "|---|---|" + "---|" * (len(piv.index) + 2)]
    for a, b in (("hybrid", "periodic"), ("hybrid", "fixed"), ("hybrid_gate", "periodic_gate"), ("hybrid_gate", "fixed"),
                 ("periodic", "fixed"), ("immediate", "periodic")):
        if a in piv and b in piv:
            d = piv[a] - piv[b]
            lines.append(f"| {LABELS[a]} | {LABELS[b]} | " + " | ".join(f"{v:+.4f}" for v in d) +
                         f" | {d.mean():+.4f} | {(d < 0).sum()}/{len(d)} |")
    lines += ["", "## 재학습 시점 (hybrid 계열)", ""]
    for r in df[df["config"].isin(["hybrid", "hybrid_gate"])].itertuples():
        lines.append(f"- {r.config} seed {r.seed}: 조기 {r.early_days or '-'} / 게이트 탈락 {r.rejected_days or '-'}")
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--walk-days", type=int, default=75)
    ap.add_argument("--quick", action="store_true", help="시드 1개·3 epoch (동작 확인용)")
    args = ap.parse_args()
    cfg = section("train")
    seeds = args.seeds or (cfg["seeds"][:1] if args.quick else cfg["seeds"])
    epochs = args.epochs or (3 if args.quick else cfg["epochs"])
    early = section("retrain")["early"]
    period = section("retrain")["period_days"]
    obs, events = T.load_data()
    print(f"[조기 재학습 평가] seeds={seeds} epochs={epochs} walk_days={args.walk_days} early={early} period={period}", flush=True)
    df = run(obs, events, seeds, epochs, args.walk_days, early, period)
    write_report(df, {"seeds": seeds, "epochs": epochs, "walk_days": args.walk_days, "early": early,
                      "period_days": period, "source": T.data_source(), "end": str(obs["ts"].max())})


if __name__ == "__main__":
    main()
