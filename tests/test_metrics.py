import math
import unittest
import pandas as pd

from min_temp import (
    MinTempPredictor,
    PredictionResult,
    compute_metrics,
    evaluate_predictors,
    evaluate_predictor,
)


class TestComputeMetrics(unittest.TestCase):
    def test_values(self):
        result = compute_metrics([1.0, 2.0], [1.0, 4.0])
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.y_true, (1.0, 2.0))
        self.assertEqual(result.y_pred, (1.0, 4.0))
        self.assertEqual(result.errors, (0.0, 2.0))
        self.assertAlmostEqual(result.mae, 1.0)
        self.assertAlmostEqual(result.rmse, math.sqrt(2.0))
        self.assertAlmostEqual(result.bias, 1.0)
    
    def test_different_lengths_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics([1.0, 2.0], [1.0])

class TestEvaluatePredictor(unittest.TestCase):
    def test_evaluate_predictor_with_callable(self):
        df = pd.DataFrame({"observed_min_temp_c": [1.0, 2.0]})

        def predict(df_):
            return df_["observed_min_temp_c"] + 1.0

        result = evaluate_predictor(df, predict)
        self.assertEqual(result.method, "predict")
        self.assertEqual(result.errors, (1.0, 1.0))
        self.assertAlmostEqual(result.mae, 1.0)

    def test_evaluate_predictor_with_min_temp_predictor(self):
        df = pd.DataFrame({"observed_min_temp_c": [0.0, 2.0]})

        class DummyPredictor(MinTempPredictor):
            name = "dummy"

            def predict(self, row):
                return 1.0

        predictor = DummyPredictor()
        result = evaluate_predictor(df, predictor)
        self.assertEqual(result.method, "dummy")
        self.assertEqual(result.y_pred, (1.0, 1.0))
        self.assertEqual(result.errors, (1.0, -1.0))
        self.assertAlmostEqual(result.mae, 1.0)
        self.assertAlmostEqual(result.rmse, 1.0)
        self.assertAlmostEqual(result.bias, 0.0)

    
    def test_evaluate_predictor_with_method_name(self):
        df = pd.DataFrame({"observed_min_temp_c": [1.0, 2.0]})

        def predict(df_):
            return df_["observed_min_temp_c"] + 2.0

        result = evaluate_predictor(df, predict, method_name="custom_method")
        self.assertEqual(result.method, "custom_method")
        self.assertEqual(result.errors, (2.0, 2.0))
        self.assertAlmostEqual(result.mae, 2.0)
        self.assertAlmostEqual(result.rmse, 2.0)
        self.assertAlmostEqual(result.bias, 2.0)
 
class TestEvaluatePredictors(unittest.TestCase):
    def test_evaluate_methods_order_preserved(self):
        df = pd.DataFrame({"observed_min_temp_c": [1.0, 2.0]})

        def predict_a(d):
            return d["observed_min_temp_c"] + 1

        def predict_b(d):
            return d["observed_min_temp_c"] - 1

        methods = {"A": predict_a, "B": predict_b}
        results = evaluate_predictors(df, methods)

        self.assertEqual(tuple(r.method for r in results), ("A", "B"))
        self.assertEqual(results[0].errors, (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
