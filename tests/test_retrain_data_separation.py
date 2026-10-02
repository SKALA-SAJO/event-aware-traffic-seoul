"""신호용 / 재학습용 데이터 분리 테스트: 시뮬레이션 주입 관측치는 드리프트 신호로만 쓰고 재학습에는 쓰지 않는다.

    python -m unittest tests.test_retrain_data_separation -v

임시 DB 만 사용합니다 (실제 data/traffic.db·mlflow.db 를 바꾸지 않음).
"""
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

_TMP_MLFLOW = tempfile.mkdtemp()
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_TMP_MLFLOW}/mlflow.db")  # import 시 실제 mlflow.db 를 건드리지 않게

from data import storage
from data.config import section
from serving_app import train_and_register as tr
from serving_app import training as T

REAL_END = pd.Timestamp("2026-09-30 23:00")
SIM_END = pd.Timestamp("2026-10-05 23:00")


def obs_frame(start, end, corridor="c1", speed=20.0):
    ts = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"corridor": corridor, "ts": ts, "speed": speed, "volume": float("nan")})


class SeparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "t.db")
        # 실측(topis) 9/1~9/30 + 시뮬레이션 주입(simulation) 10/1~10/5 + 2023 이전 이력(topis_history)
        storage.upsert_observations(obs_frame("2022-12-01", "2022-12-31 23:00", speed=15.0), "topis_history", self.db)
        storage.upsert_observations(obs_frame("2026-09-01", REAL_END), "topis", self.db)
        storage.upsert_observations(obs_frame("2026-10-01", SIM_END, speed=8.0), "simulation", self.db)

    def test_last_observation_includes_simulation_by_default(self):
        self.assertEqual(storage.last_observation_ts(db_path=self.db), SIM_END)

    def test_last_observation_can_exclude_sources(self):
        self.assertEqual(storage.last_observation_ts(db_path=self.db, exclude_sources=("simulation",)), REAL_END)
        self.assertEqual(storage.last_observation_ts("c1", self.db, ("simulation", "topis")), pd.Timestamp("2022-12-31 23:00"))
        self.assertIsNone(storage.last_observation_ts("none", self.db, ("simulation",)))

    def test_load_observations_excludes_simulation(self):
        full = storage.load_observations(db_path=self.db)
        kept = storage.load_observations(db_path=self.db, exclude_sources=("simulation", "topis_history"))
        self.assertTrue((full["speed"] == 8.0).any())
        self.assertFalse((kept["speed"] == 8.0).any())  # 시뮬레이션 속도(8.0)가 학습 데이터에 없음
        self.assertFalse((kept["speed"] == 15.0).any())  # 2023 이전 이력도 없음
        self.assertEqual(pd.Timestamp(kept["ts"].max()), REAL_END)

    def test_retrain_window_uses_confirmed_data_end(self):
        rcfg = {"fine_tune_days": 14, "eval_days": 1, "exclude_sources": ["simulation", "topis_history"]}
        exclude, end, ft_start, eval_start = tr.retrain_window(rcfg, self.db)
        self.assertEqual(exclude, ("simulation", "topis_history"))
        self.assertEqual(end, REAL_END)  # 시뮬레이션 때문에 구간이 뒤로 밀리지 않음
        self.assertEqual(ft_start, REAL_END - pd.Timedelta(days=14))
        self.assertLess(eval_start, end)

    def test_demo_mode_includes_simulation(self):
        rcfg = {"fine_tune_days": 14, "eval_days": 1, "exclude_sources": ["topis_history", "synthetic"]}
        _, end, ft_start, _ = tr.retrain_window(rcfg, self.db)
        self.assertEqual(end, SIM_END)  # 시연 모드: 목록에서 simulation 만 빼면 주입 데이터가 학습 구간이 됨
        self.assertEqual(ft_start, SIM_END - pd.Timedelta(days=14))

    def test_no_exclusion_configured(self):
        _, end, _, _ = tr.retrain_window({"fine_tune_days": 14, "eval_days": 1}, self.db)
        self.assertEqual(end, SIM_END)

    def test_nothing_left_after_exclusion_raises(self):
        db2 = os.path.join(self.tmp.name, "only_sim.db")
        storage.upsert_observations(obs_frame("2026-10-01", "2026-10-02"), "simulation", db2)
        with self.assertRaises(ValueError):
            tr.retrain_window({"fine_tune_days": 14, "eval_days": 1, "exclude_sources": ["simulation"]}, db2)

    def test_data_source_summary_respects_exclusion(self):
        with mock.patch.object(storage, "DB_PATH", self.db):
            self.assertEqual(T.data_source(("simulation", "topis_history")), "topis")
            self.assertIn("simulation", T.data_source(()))


class ConfigDefaultTests(unittest.TestCase):
    def test_default_config_separates_signal_and_training_data(self):
        excl = section("retrain")["exclude_sources"]
        self.assertIn("simulation", excl)      # 주입 데이터는 신호로만
        self.assertIn("topis_history", excl)   # 코로나 이력은 학습 제외 (기존 base 학습 규칙과 같음)

    def test_base_training_default_already_excludes_simulation(self):
        self.assertIn("simulation", T.DEFAULT_EXCLUDE)


if __name__ == "__main__":
    unittest.main()
