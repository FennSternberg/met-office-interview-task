import unittest

import pandas as pd

from min_temp import KTable, parse_initial_data, parse_k_table


class TestParsers(unittest.TestCase):
    def test_parse_k_table_matches_embedded_table(self):
        parsed = parse_k_table("k_table.csv")
        expected =  KTable(
            pd.DataFrame(
                data={
                    2.0: [-2.2, -1.1, -0.6, 1.1],
                    4.0: [-1.7, 0.0, 0.0, 1.7],
                    6.0: [-0.6, 0.6, 0.6, 2.8],
                    8.0: [0.0, 1.1, 1.1, float("nan")],
                },
                index=[12.5, 25.5, 38.5, 51],
            )
        )
        pd.testing.assert_frame_equal(parsed.values, expected.values)

    def test_parse_initial_data_fields(self):
        df = parse_initial_data("data/initial_data.csv")
        self.assertEqual(len(df), 4)
        first = df.iloc[0]
        for key in {"Date", "Location", "midday_temp_c", "midday_dew_point_c", "wind_kn", "cloud_oktas"}:
            self.assertIn(key, df.columns)
        self.assertEqual(first["Date"], "1")
        self.assertEqual(first["Location"], "A")
        self.assertAlmostEqual(first["midday_temp_c"], 22.4)
        self.assertAlmostEqual(first["midday_dew_point_c"], 10.9)
        self.assertAlmostEqual(first["wind_kn"], 14.56)
        self.assertAlmostEqual(first["cloud_oktas"], 3.9)


if __name__ == "__main__":
    unittest.main()
