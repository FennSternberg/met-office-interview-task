import unittest

import pandas as pd

from min_temp import CraddockAndPritchardOptimizer, KTable


class TestCalibration(unittest.TestCase):
    def test_fit_recovers_parameters(self):
        # Simple synthetic setup with 2x2 K-table and known coefficients.
        true_a, true_b, true_c = 0.3, 0.5, -1.0
        true_k_table = KTable(pd.DataFrame(
            data={
                2.0: [1.0, 3.0],
                5.0: [2.0, 4.0],
            },
            index=[10.0, 20.0],
        ))

        # Build data that hits each bin.
        rows = [
            {"midday_temp_c": 10, "midday_dew_point_c": 4, "wind_kn": 8, "cloud_oktas": 1},    # k=1.0
            {"midday_temp_c": 12, "midday_dew_point_c": 7, "wind_kn": 9, "cloud_oktas": 4},    # k=2.0
            {"midday_temp_c": 8,  "midday_dew_point_c": 3.5, "wind_kn": 15, "cloud_oktas": 1}, # k=3.0
            {"midday_temp_c": 6,  "midday_dew_point_c": 2.0, "wind_kn": 18, "cloud_oktas": 4}, # k=4.0
            {"midday_temp_c": 11, "midday_dew_point_c": 5, "wind_kn": 8, "cloud_oktas": 1},    # k=1.0
            {"midday_temp_c": 13, "midday_dew_point_c": 6, "wind_kn": 9, "cloud_oktas": 4},    # k=2.0
            {"midday_temp_c": 9,  "midday_dew_point_c": 5, "wind_kn": 15, "cloud_oktas": 1},   # k=3.0
            {"midday_temp_c": 7,  "midday_dew_point_c": 2.5, "wind_kn": 18, "cloud_oktas": 4}, # k=4.0
        ]
        df = pd.DataFrame(rows)
        df["observed_min_temp_c"] = (
            true_a * df["midday_temp_c"]
            + true_b * df["midday_dew_point_c"]
            + true_c
            + df.apply(lambda r: true_k_table.lookup(r["wind_kn"], r["cloud_oktas"]), axis=1)
        )

        fitted = CraddockAndPritchardOptimizer(
            a=0, b=0, c=0, k_table=true_k_table
        ).fit(df)

        self.assertAlmostEqual(fitted.model.a, true_a, places=6)
        self.assertAlmostEqual(fitted.model.b, true_b, places=6)
        self.assertAlmostEqual(fitted.model.c, true_c, places=6)

        pd.testing.assert_frame_equal(fitted.model.k_table.values, true_k_table.values)

      
    
    def test_fit_recovers_k_table(self):
        # Simple synthetic setup with 2x2 K-table and known coefficients.
        true_a, true_b, true_c = 0.3, 0.5, -1.0
        true_k_table = KTable(pd.DataFrame(
            data={
                2.0: [1.0, 3.0],
                5.0: [2.0, 4.0],
            },
            index=[10.0, 20.0],
        ))

        # Build data that hits each bin.
        rows = [
            {"midday_temp_c": 10, "midday_dew_point_c": 4, "wind_kn": 8, "cloud_oktas": 1},    # k=1.0
            {"midday_temp_c": 12, "midday_dew_point_c": 7, "wind_kn": 9, "cloud_oktas": 4},    # k=2.0
            {"midday_temp_c": 8,  "midday_dew_point_c": 3.5, "wind_kn": 15, "cloud_oktas": 1}, # k=3.0
            {"midday_temp_c": 6,  "midday_dew_point_c": 2.0, "wind_kn": 18, "cloud_oktas": 4}, # k=4.0
            {"midday_temp_c": 11, "midday_dew_point_c": 5, "wind_kn": 8, "cloud_oktas": 1},    # k=1.0
            {"midday_temp_c": 13, "midday_dew_point_c": 6, "wind_kn": 9, "cloud_oktas": 4},    # k=2.0
            {"midday_temp_c": 9,  "midday_dew_point_c": 5, "wind_kn": 15, "cloud_oktas": 1},   # k=3.0
            {"midday_temp_c": 7,  "midday_dew_point_c": 2.5, "wind_kn": 18, "cloud_oktas": 4}, # k=4.0
        ]
        
        df = pd.DataFrame(rows)
        df["observed_min_temp_c"] = (
            true_a * df["midday_temp_c"]
            + true_b * df["midday_dew_point_c"]
            + true_c
            + df.apply(lambda r: true_k_table.lookup(r["wind_kn"], r["cloud_oktas"]), axis=1)
        )

        wrong_k_table = KTable(pd.DataFrame(
            data={
                2.0: [0.0, 0.0],
                5.0: [0.0, 0.0],
            },
            index=[10.0, 20.0],
        ))

        fitted = CraddockAndPritchardOptimizer(
            k_table=wrong_k_table
        ).fit(df, fixed={"c": true_c})

        self.assertAlmostEqual(fitted.model.a, true_a, places=6)
        self.assertAlmostEqual(fitted.model.b, true_b, places=6)
        self.assertAlmostEqual(fitted.model.c, true_c, places=6)
        pd.testing.assert_frame_equal(fitted.model.k_table.values, true_k_table.values)


if __name__ == "__main__":
    unittest.main()
