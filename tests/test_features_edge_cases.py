"""이벤트·관측치가 없는 시간대의 전처리 회귀 테스트."""

import unittest

import numpy as np
import pandas as pd

from data.features import FeatureSpec, TrafficScaler, build_frames, make_windows


class FeatureEdgeCaseTests(unittest.TestCase):
    def test_build_frames_without_events(self):
        unit = "gwanghwamun_up"
        hours = pd.date_range("2026-09-30", periods=40, freq="h")
        obs = pd.DataFrame({
            "corridor": unit,
            "ts": hours,
            "speed": np.linspace(20.0, 25.0, len(hours)),
            "volume": np.nan,
        })
        spec = FeatureSpec([unit], "full", "decay")

        frames = build_frames(obs, pd.DataFrame(), spec, TrafficScaler().fit(obs))

        frame = frames[unit]
        self.assertEqual(len(frame), len(hours))
        self.assertFalse(frame["is_event"].any())
        self.assertTrue((frame[spec.event_names] == 0).all().all())
        self.assertGreater(len(make_windows(frames, spec)), 0)

    def test_make_windows_with_only_invalid_inputs_returns_empty(self):
        unit = "gwanghwamun_up"
        hours = pd.date_range("2026-09-30", periods=40, freq="h")
        frame = pd.DataFrame(index=hours, data={
            "speed_z": np.nan,
            "speed": np.nan,
            "is_event": False,
            "is_holiday": False,
            "naive": np.nan,
        })
        spec = FeatureSpec([unit], "speed")

        windows = make_windows({unit: frame}, spec)

        self.assertEqual(len(windows), 0)
        self.assertEqual(windows.X_past.shape, (0, spec.lookback, len(spec.past_names())))
        self.assertEqual(windows.X_fut.shape, (0, spec.horizon, len(spec.future_names())))

    def test_make_windows_skips_invalid_unit_when_other_unit_is_valid(self):
        units = ["gwanghwamun_up", "jamsil_up"]
        hours = pd.date_range("2026-09-30", periods=40, freq="h")
        frames = {}
        for unit, speed in zip(units, (np.nan, 25.0)):
            frames[unit] = pd.DataFrame(index=hours, data={
                "speed_z": speed,
                "speed": speed,
                "is_event": False,
                "is_holiday": False,
                "naive": np.nan,
            })

        windows = make_windows(frames, FeatureSpec(units, "speed"))

        self.assertGreater(len(windows), 0)
        self.assertEqual(set(windows.units), {"jamsil_up"})


if __name__ == "__main__":
    unittest.main()
