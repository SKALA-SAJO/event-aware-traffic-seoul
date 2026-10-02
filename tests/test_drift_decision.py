"""드리프트 판정 · 조기 재학습 신청 · 재학습 시점 판단 · 서버 reload 회귀 테스트.

    python -m unittest tests.test_drift_decision -v

실행에 MLflow 서버·학습·실데이터 DB 가 필요 없습니다 (임시 폴더의 sqlite 와 가짜 판정 결과만 씀).
"""
import datetime as dt
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from serving_app.monitoring import drift_state, retrain_schedule, retrain_trigger
from serving_app.monitoring.drift_detector import evaluate_drift

DRIFT_CFG = {"window_hours": 72, "min_samples": 36, "rmse_ratio_threshold": 1.5,
             "exclude_holidays": True, "incident_buffer_hours": 1}
EARLY_CFG = {"enabled": True, "consecutive_days": 2, "recent_construction_days": 14,
             "cooldown_days": 3, "request_expire_days": 7}
REF = 1.0  # 기준 평상 RMSE


def make_errors(start: str, hours: int = 72, sigma: float = 1.0, big: tuple | None = None, big_sigma: float = 6.0):
    """시간당 오차 표본. big=(시작, 끝) 시각 사이만 오차가 큼. 실제=예측+오차."""
    rng = np.random.default_rng(0)
    ts = pd.date_range(start, periods=hours, freq="h")
    err = rng.normal(0, sigma, hours)
    if big:
        m = (ts >= pd.Timestamp(big[0])) & (ts <= pd.Timestamp(big[1]))
        err[m] = rng.choice([-1, 1], m.sum()) * big_sigma
    pred = np.full(hours, 20.0)
    return pd.DataFrame({"target_ts": ts, "horizon": 1, "predicted": pred, "actual": pred + err})


def event(start, end, status="scheduled"):
    return pd.DataFrame([{"start": pd.Timestamp(start), "end": pd.Timestamp(end), "status": status}])


def incident(category, start, end):
    return pd.DataFrame([{"category": category, "start": pd.Timestamp(start), "last_seen": pd.Timestamp(end),
                          "expected_end": pd.NaT, "type_name": category, "info": "x"}])


class EvaluateDriftTests(unittest.TestCase):
    """기존 판정(evaluate_drift): 무엇이 드리프트이고 무엇을 제외하는가."""

    def judge(self, errors, events=None, incidents=None):
        return evaluate_drift(errors, events, REF, DRIFT_CFG, incidents)

    def test_normal_error_is_ok(self):
        r = self.judge(make_errors("2026-09-21"))
        self.assertEqual(r["status"], "ok")
        self.assertLess(r["ratio"], 1.5)

    def test_persistent_large_error_is_drift(self):
        r = self.judge(make_errors("2026-09-21", sigma=3.0))
        self.assertEqual(r["status"], "drift")
        self.assertGreater(r["ratio"], 1.5)

    def test_too_few_hours_is_insufficient(self):
        r = self.judge(make_errors("2026-09-21", hours=20, sigma=3.0))
        self.assertEqual(r["status"], "insufficient_data")
        self.assertEqual(r["reason"], "too_few_hours")

    def test_no_predictions_is_insufficient(self):
        r = self.judge(make_errors("2026-09-21", hours=0))
        self.assertEqual(r["status"], "insufficient_data")

    def test_error_during_known_event_is_excluded(self):
        errors = make_errors("2026-09-21", big=("2026-09-21 18:00", "2026-09-21 23:00"))
        r = self.judge(errors, event("2026-09-21 19:00", "2026-09-21 22:00"))
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["rmse_known_event_hours"], 4.0)  # 이벤트 시간 오차는 크지만 판정에서 빠짐

    def test_same_error_without_event_registered_is_drift(self):
        errors = make_errors("2026-09-21", big=("2026-09-21 18:00", "2026-09-22 20:00"))
        self.assertEqual(self.judge(errors)["status"], "drift")

    def test_cancelled_event_does_not_exclude(self):
        errors = make_errors("2026-09-21", big=("2026-09-21 18:00", "2026-09-22 20:00"))
        r = self.judge(errors, event("2026-09-21 19:00", "2026-09-22 19:00", status="cancelled"))
        self.assertEqual(r["status"], "drift")

    def test_holiday_is_excluded(self):
        # 2026-10-03 개천절 하루 전체에만 큰 오차
        errors = make_errors("2026-10-02", big=("2026-10-03 00:00", "2026-10-03 23:00"))
        self.assertEqual(self.judge(errors)["status"], "ok")

    def test_transient_incident_is_excluded(self):
        errors = make_errors("2026-09-21", big=("2026-09-21 10:00", "2026-09-21 20:00"))
        r = self.judge(errors, incidents=incident("accident", "2026-09-21 10:00", "2026-09-21 20:00"))
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["excluded_incident_hours"], 0)

    def test_construction_is_not_excluded_and_reported_as_cause(self):
        errors = make_errors("2026-09-21", sigma=3.0)
        inc = incident("construction", "2026-09-20", "2026-09-30")
        r = self.judge(errors, incidents=inc)
        self.assertEqual(r["status"], "drift")
        self.assertEqual(len(r["constructions"]), 1)


class DriftStateTests(unittest.TestCase):
    """일별 감지 기록 · 연속일 · 최근 공사 · 신청 파일."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for name, fname in (("STATE_PATH", "state.json"), ("REQUEST_PATH", "request.json")):
            p = mock.patch.object(drift_state, name, os.path.join(self.tmp.name, fname))
            p.start()
            self.addCleanup(p.stop)

    def test_consecutive_days_counts_calendar_days(self):
        self.assertEqual(drift_state.record_drift("c", "2026-10-10 23:00", "1", False), 1)
        self.assertEqual(drift_state.record_drift("c", "2026-10-11 23:00", "1", False), 2)
        self.assertEqual(drift_state.record_drift("c", "2026-10-12 05:00", "1", False), 3)

    def test_same_day_twice_counts_once(self):
        drift_state.record_drift("c", "2026-10-10 10:00", "1", False)
        self.assertEqual(drift_state.record_drift("c", "2026-10-10 23:00", "1", False), 1)

    def test_missing_day_breaks_streak(self):
        drift_state.record_drift("c", "2026-10-10 23:00", "1", False)
        self.assertEqual(drift_state.record_drift("c", "2026-10-12 23:00", "1", False), 1)

    def test_streak_is_per_corridor(self):
        drift_state.record_drift("a", "2026-10-10 23:00", "1", False)
        self.assertEqual(drift_state.record_drift("b", "2026-10-11 23:00", "1", False), 1)

    def test_new_model_version_resets_history(self):
        drift_state.record_drift("c", "2026-10-10 23:00", "1", False)
        self.assertEqual(drift_state.record_drift("c", "2026-10-11 23:00", "2", False), 1)

    def test_first_today_logs_once_per_day(self):
        self.assertTrue(drift_state.first_today("drift", "c", "2026-10-10 01:00", "1"))
        self.assertFalse(drift_state.first_today("drift", "c", "2026-10-10 22:00", "1"))
        self.assertTrue(drift_state.first_today("drift", "c", "2026-10-11 01:00", "1"))
        self.assertTrue(drift_state.first_today("drift", "other", "2026-10-10 01:00", "1"))

    def test_early_reason_thresholds(self):
        self.assertIsNone(drift_state.early_reason(1, False, EARLY_CFG))
        self.assertIsNotNone(drift_state.early_reason(2, False, EARLY_CFG))
        self.assertIsNotNone(drift_state.early_reason(1, True, EARLY_CFG))  # 최근 공사면 1일로 충분

    def test_recent_construction_filters(self):
        now = pd.Timestamp("2026-10-20 12:00")
        inc = pd.DataFrame([
            {"category": "construction", "corridor": None, "start": pd.Timestamp("2026-10-10"), "type_name": "공사", "info": "최근"},
            {"category": "construction", "corridor": None, "start": pd.Timestamp("2026-09-01"), "type_name": "공사", "info": "오래됨"},
            {"category": "construction", "corridor": "other", "start": pd.Timestamp("2026-10-10"), "type_name": "공사", "info": "다른구간"},
            {"category": "accident", "corridor": None, "start": pd.Timestamp("2026-10-10"), "type_name": "사고", "info": "사고"},
            {"category": "construction", "corridor": "c", "start": pd.Timestamp("2026-10-25"), "type_name": "공사", "info": "미래"},
        ])
        got = drift_state.recent_construction(inc, "c", now, 14)
        self.assertEqual(len(got), 1)
        self.assertIn("최근", got[0])
        self.assertEqual(drift_state.recent_construction(pd.DataFrame(), "c", now, 14), [])

    def test_request_keeps_first_and_clears(self):
        self.assertTrue(drift_state.write_request("c", "first"))
        self.assertFalse(drift_state.write_request("c", "second"))
        self.assertEqual(drift_state.read_request()["reason"], "first")
        drift_state.clear_request()
        self.assertIsNone(drift_state.read_request())

    def test_broken_state_file_is_treated_as_empty(self):
        with open(drift_state.STATE_PATH, "w") as f:
            f.write("{not json")
        self.assertEqual(drift_state.record_drift("c", "2026-10-10 23:00", "1", False), 1)


class CheckAlertOnlyTests(unittest.TestCase):
    """운영 경로(/monitoring/check)의 판단: 경보 + 신청만 하고 재학습은 하지 않는다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for name, fname in (("STATE_PATH", "state.json"), ("REQUEST_PATH", "request.json")):
            p = mock.patch.object(drift_state, name, os.path.join(self.tmp.name, fname))
            p.start()
            self.addCleanup(p.stop)
        self.on_drift = "alert"
        self.incidents = pd.DataFrame(columns=["category", "corridor", "start", "type_name", "info"])
        fake_model = mock.Mock(version="1", units=["c"])
        for target, kw in (
            (retrain_trigger.model_loader, {"get_model": mock.Mock(return_value=fake_model)}),
            (retrain_trigger.storage, {"load_incidents": lambda **k: self.incidents}),
            (retrain_trigger, {"hub_of": lambda c: "hub", "section": self._section}),
        ):
            for attr, val in kw.items():
                p = mock.patch.object(target, attr, val)
                p.start()
                self.addCleanup(p.stop)
        # 테스트 로그가 실제 logs/aiops.log 에 섞이지 않게, 기존 핸들러(main.py 가 붙인 파일 핸들러)를 떼고 수집용만 붙임
        self.logs = []
        handler = mock.Mock(level=0)
        handler.handle = lambda r: self.logs.append(r.getMessage())
        log = retrain_trigger.logger
        saved = (log.handlers[:], log.level, log.propagate)
        log.handlers[:] = [handler]
        log.setLevel(10)
        log.propagate = False
        self.addCleanup(lambda: (log.handlers.__setitem__(slice(None), saved[0]), log.setLevel(saved[1]),
                                 setattr(log, "propagate", saved[2])))

    def _section(self, name):
        return {"retrain": {"on_drift": self.on_drift, "early": EARLY_CFG}, "drift": DRIFT_CFG}[name]

    def check(self, day, status="drift"):
        st = {"corridor": "c", "status": status, "window_end": f"{day} 23:00:00", "rmse": 3.0, "reference_rmse": 1.5,
              "ratio": 2.0, "excluded_hours": 0, "excluded_incident_hours": 0, "constructions": [],
              "rmse_known_event_hours": None}
        with mock.patch.object(retrain_trigger, "drift_status", lambda *a, **k: st):
            return retrain_trigger.check_alert_only("c")

    def test_first_day_alerts_but_does_not_request(self):
        r = self.check("2026-10-10")
        self.assertEqual(r["status"], "drift_detected")
        self.assertIsNone(r["early_retrain"])
        self.assertIsNone(drift_state.read_request())
        self.assertEqual(sum("[WARN] drift detected" in m for m in self.logs), 1)

    def test_second_consecutive_day_requests_retrain(self):
        self.check("2026-10-10")
        r = self.check("2026-10-11")
        self.assertIn("2 consecutive days", r["early_retrain"])
        self.assertIsNotNone(drift_state.read_request())
        self.assertTrue(any("[INFO] early retrain requested" in m for m in self.logs))

    def test_gap_day_does_not_request(self):
        self.check("2026-10-10")
        self.assertIsNone(self.check("2026-10-12")["early_retrain"])

    def test_repeated_checks_same_day_log_once(self):
        for _ in range(5):
            self.check("2026-10-10")
        self.assertEqual(sum("[WARN] drift detected" in m for m in self.logs), 1)

    def test_recent_construction_requests_on_first_day(self):
        self.incidents = pd.DataFrame([{"category": "construction", "corridor": None, "start": pd.Timestamp("2026-10-05"),
                                        "type_name": "공사", "info": "차로 차단"}])
        r = self.check("2026-10-10")
        self.assertIn("recent construction", r["early_retrain"])

    def test_old_construction_does_not_shorten(self):
        self.incidents = pd.DataFrame([{"category": "construction", "corridor": None, "start": pd.Timestamp("2026-06-01"),
                                        "type_name": "공사", "info": "오래된 공사"}])
        self.assertIsNone(self.check("2026-10-10")["early_retrain"])

    def test_on_drift_retrain_never_requests(self):
        self.on_drift = "retrain"  # 시연용 즉시 재학습 설정에서는 조기 재학습 신청을 쓰지 않음
        self.check("2026-10-10")
        self.assertIsNone(self.check("2026-10-11")["early_retrain"])

    def test_ok_status_makes_no_request(self):
        self.check("2026-10-10")
        self.assertIsNone(self.check("2026-10-11", status="ok")["early_retrain"])
        self.assertIsNone(drift_state.read_request())


class RetrainScheduleTests(unittest.TestCase):
    """retrain_schedule.decide: 14일 주기 · 신청 + 쿨다운 · 만료 · 모델 없음."""

    NOW = dt.datetime(2026, 10, 20, 12, 0)

    def setUp(self):
        from mlflow.tracking import MlflowClient

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        uri = f"sqlite:///{self.tmp.name}/mlflow.db"
        for target, name, val in (
            (retrain_schedule, "_tracking_uri", lambda: uri),
            (retrain_schedule, "section", lambda n: {"retrain": {"period_days": 14, "on_drift": "alert", "early": EARLY_CFG}}[n]),
            (drift_state, "REQUEST_PATH", os.path.join(self.tmp.name, "request.json")),
            (drift_state, "STATE_PATH", os.path.join(self.tmp.name, "state.json")),
        ):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        import mlflow
        mlflow.set_tracking_uri(uri)
        self.client = MlflowClient(uri)
        self.exp = self.client.create_experiment("traffic-speed")

    def train_run(self, days_ago: float, name="base-train"):
        start = int((self.NOW - dt.timedelta(days=days_ago)).timestamp() * 1000)
        return self.client.create_run(self.exp, start_time=start, run_name=name)

    def register(self, run):
        mv = self.client.create_model_version(retrain_schedule.MODEL_NAME, f"runs:/{run.info.run_id}/model") \
            if self._model_exists() else None
        self.client.set_registered_model_alias(retrain_schedule.MODEL_NAME, retrain_schedule.PRODUCTION_ALIAS, mv.version)

    def _model_exists(self):
        try:
            self.client.create_registered_model(retrain_schedule.MODEL_NAME)
        except Exception:
            pass
        return True

    def request(self, days_ago: float):
        drift_state.write_request("c", "drift 2 consecutive days, corridor=c", self.NOW - dt.timedelta(days=days_ago))

    def test_no_production_model(self):
        self.train_run(20)
        code, msg = retrain_schedule.decide(self.NOW)
        self.assertEqual(code, 2)
        self.assertIn("no production model", msg)

    def test_recent_training_is_skipped(self):
        self.register(self.train_run(5))
        self.assertEqual(retrain_schedule.decide(self.NOW)[0], 1)

    def test_period_elapsed_is_due(self):
        self.register(self.train_run(15))
        code, msg = retrain_schedule.decide(self.NOW)
        self.assertEqual(code, 0)
        self.assertIn("mode=periodic", msg)

    def test_period_is_measured_from_latest_run_including_fine_tune(self):
        run = self.train_run(30)
        self.register(run)
        self.train_run(2, name="fine-tune")  # 게이트 탈락한 fine-tune 이어도 기준 시각으로 침
        self.assertEqual(retrain_schedule.decide(self.NOW)[0], 1)

    def test_request_waits_for_cooldown(self):
        self.register(self.train_run(1))
        self.request(0.5)
        code, msg = retrain_schedule.decide(self.NOW)
        self.assertEqual(code, 1)
        self.assertIn("cooldown", msg)

    def test_request_after_cooldown_is_due_early(self):
        self.register(self.train_run(4))
        self.request(0.5)
        code, msg = retrain_schedule.decide(self.NOW)
        self.assertEqual(code, 0)
        self.assertIn("mode=drift-early", msg)

    def test_expired_request_is_dropped(self):
        self.register(self.train_run(5))
        self.request(8)
        self.assertEqual(retrain_schedule.decide(self.NOW)[0], 1)
        self.assertIsNone(drift_state.read_request())

    def test_failed_gate_run_still_triggers_cooldown(self):
        # 방금 fine-tune(실패 포함)했다면 신청이 있어도 쿨다운 동안은 다시 하지 않음 → 실패 후 반복 재시도 방지
        self.register(self.train_run(20))
        self.train_run(1, name="fine-tune")
        self.request(0.2)
        self.assertEqual(retrain_schedule.decide(self.NOW)[0], 1)


class AdminReloadTests(unittest.TestCase):
    def test_reload_is_localhost_only(self):
        from fastapi.testclient import TestClient

        from serving_app.main import app

        remote = TestClient(app).post("/admin/reload")  # 클라이언트 주소가 localhost 가 아님
        self.assertEqual(remote.status_code, 403)
        local = TestClient(app, client=("127.0.0.1", 50000)).post("/admin/reload")
        self.assertEqual(local.status_code, 200)
        self.assertIn("model_version", local.json())


if __name__ == "__main__":
    unittest.main()
