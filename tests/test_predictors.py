import unittest

import pandas as pd

from min_temp import (
    KTable,
    CraddockAndPritchardModel,
    parse_initial_data,
)

class TestKTable(unittest.TestCase):
    def test_lookup_basic(self):
        data = {
            # cloud  edges : K values
            2.0: [1.0, 2.0, 3.0],
            5.0: [4.0, 5.0, 6.0],
            8.0: [7.0, 8.0, 9.0],
        }
        index = [10.0, 20.0, 30.0]  # wind speed edges
        k_table = KTable(pd.DataFrame(data=data, index=index))

        self.assertEqual(k_table.lookup(5.0, 2.0), 1.0)   # wind<=10, cloud<=2
        self.assertEqual(k_table.lookup(15.0, 1.0), 2.0)  # wind<=20, cloud<=2
        self.assertEqual(k_table.lookup(25.0, 6.0), 9.0)  # wind<=30, cloud<=8

    def test_lookup_out_of_bounds(self):
        data = {
            # cloud  edges : K values
            2.0: [1.0, 2.0],
            5.0: [3.0, 4.0],
        }
        index = [10.0, 20.0]  # wind speed edges
        k_table = KTable(pd.DataFrame(data=data, index=index))

        with self.assertRaises(ValueError):
            k_table.lookup(25.0, 1.0)

        with self.assertRaises(ValueError):
            k_table.lookup(5.0, 6.0)
    
    def test_lookup_nan(self):
        data = {
            # cloud  edges : K values
            2.0: [1.0, float("nan")],
            5.0: [3.0, 4.0],
        }
        index = [10.0, 20.0]  # wind speed edges
        k_table = KTable(pd.DataFrame(data=data, index=index))

        with self.assertRaises(ValueError):
            k_table.lookup(15.0, 2.0)  # wind<=20, cloud<=2 -> nan
    
    def test_duplicate_edges(self):
        data = {
            # cloud  edges : K values
            2.0: [1.0, 2.0],
            2.0: [3.0, 4.0],  # duplicate
        }
        index = [10.0, 10.0]  # duplicate wind speed edges

        with self.assertRaises(ValueError):
            KTable(pd.DataFrame(data=data, index=index))
    
    def test_unsorted_edges(self):
        data = {
            # cloud  edges : K values
            5.0: [3.0, 4.0],
            2.0: [1.0, 2.0],  # unsorted
        }
        index = [20.0, 10.0]  # unsorted wind speed edges

        with self.assertRaises(ValueError):
            KTable(pd.DataFrame(data=data, index=index))
       


class TestCraddockAndPritchard(unittest.TestCase):
    def test_pdf_example(self):
        model = CraddockAndPritchardModel()
        t12 = 18.0
        td12 = 10.0
        cloud_oktas = 3.0
        wind_kn = 30.0

        k_val = model.k_table.lookup(wind_kn, cloud_oktas)
        self.assertEqual(k_val, 0.0)

        predicted = model.predict(
            {
                "midday_temp_c": t12,
                "midday_dew_point_c": td12,
                "wind_kn": wind_kn,
                "cloud_oktas": cloud_oktas,
            }
        )
        expected_tmin = 0.316 * t12 + 0.548 * td12 - 1.24
        self.assertAlmostEqual(predicted, expected_tmin, places=6)
    
    def test_predict_from_initial_data_df(self):
        df = parse_initial_data("data/initial_data.csv")
        model = CraddockAndPritchardModel()
        preds = df.apply(model.predict, axis=1)
        expected = (
            11.8116,  # Date 1 Location A
            10.9698,  # Date 1 Location B
            9.4340,   # Date 2 Location B
            7.4824,   # Date 2 Location C
        )
        self.assertEqual(len(preds), len(expected))
        for got, want in zip(preds, expected):
            self.assertAlmostEqual(got, want, places=3)

    def test_k_table_is_not_shared_between_model_instances(self):
        """
        Guard against the 'mutable default' pitfall:
        """
        m1 = CraddockAndPritchardModel()
        m2 = CraddockAndPritchardModel()

        self.assertIsNot(
            m1.k_table.values,
            m2.k_table.values,
        )

        # Mutate m1 and ensure m2 doesn't change
        before_m2 = m2.k_table.values.copy(deep=True)
        first_wind_edge = m1.k_table.values.index[0]
        first_cloud_edge = m1.k_table.values.columns[0]
        m1.k_table.values.loc[first_wind_edge, first_cloud_edge] = 12345.0

        pd.testing.assert_frame_equal(
            m2.k_table.values,
            before_m2,
            check_dtype=True,
            check_exact=True,
            obj="m2.k_table.values",
        )

    def test_custom_k_table_is_not_shared_between_model_instances(self):
        custom_k = KTable(
            pd.DataFrame(
                data={
                    # cloud_oktas edges : K values
                    2.0: [1.0, 3.0],
                    5.0: [2.0, 4.0],
                },
                index=[10.0, 20.0],  # wind speed edges
            )
        )
        model = CraddockAndPritchardModel(k_table=custom_k)
        k0 = model.k_table.lookup(15.0, 3.0)  # should be 4.0

        custom_k.values.loc[20.0, 5.0] = 999.0  # mutate original custom_k
        k1 = model.k_table.lookup(15.0, 3.0)  # should still be 4.0
        self.assertEqual(
            k0,
            k1,
        )

    def test_custom_coefficients(self):
        model = CraddockAndPritchardModel(a=0.5, b=0.5, c=0.0)
        t12, td12, wind_kn, cloud_oktas = 10.0, 5.0, 10.0, 1.0
        expected_k = -2.2  # wind<=12, cloud<=2
        expected = 0.5 * t12 + 0.5 * td12 + expected_k  # 5.3
        features = {
            "midday_temp_c": t12,
            "midday_dew_point_c": td12,
            "wind_kn": wind_kn,
            "cloud_oktas": cloud_oktas,
        }
        self.assertAlmostEqual(model.predict(features), expected, places=6)

    def test_custom_k_table(self):
        custom_k = KTable(
            pd.DataFrame(
                data={
                    # cloud  edges : K values
                    2.0: [1.0, 3.0],
                    5.0: [2.0, 4.0],
                },
                index=[10.0, 20.0], # wind speed edges
            )
        )
        model = CraddockAndPritchardModel(k_table=custom_k)
        # falls into wind<=20, cloud<=5 -> k = 4.0
        t12, td12, wind_kn, cloud_oktas = 12.0, 6.0, 15.0, 3.0
        linear = model.a * t12 + model.b * td12 + model.c
        expected = linear + 4.0
        features = {
            "midday_temp_c": t12,
            "midday_dew_point_c": td12,
            "wind_kn": wind_kn,
            "cloud_oktas": cloud_oktas,
        }
        self.assertAlmostEqual(model.predict(features), expected, places=6)

    def test_custom_coefficients_and_k_table(self):
        custom_k = KTable(
            pd.DataFrame(
                data={
                    3.0: [0.5, 1.5],
                    7.0: [1.0, 2.0],
                },
                index=[8.0, 16.0],
            )
        )
        model = CraddockAndPritchardModel(a=0.2, b=0.3, c=1.0, k_table=custom_k)
        # falls into wind<=16, cloud<=7 -> k = 2.0
        t12, td12, wind_kn, cloud_oktas = 15.0, 5.0, 10.0, 6.0
        expected = 0.2 * t12 + 0.3 * td12 + 1.0 + 2.0  # 7.5
        features = {
            "midday_temp_c": t12,
            "midday_dew_point_c": td12,
            "wind_kn": wind_kn,
            "cloud_oktas": cloud_oktas,
        }
        self.assertAlmostEqual(model.predict(features), expected, places=6)


if __name__ == '__main__':
    unittest.main()
