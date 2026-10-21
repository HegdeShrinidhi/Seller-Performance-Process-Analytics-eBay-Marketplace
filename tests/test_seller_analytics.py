import json

import pandas as pd

from seller_analytics import calculate_seller_metrics, generate_orders, run_pipeline


def test_generate_orders_is_reproducible():
    first = generate_orders(sellers=10, orders_per_seller=8, seed=7)
    second = generate_orders(sellers=10, orders_per_seller=8, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 80


def test_calculate_seller_metrics_aggregates_kpis():
    orders = pd.DataFrame(
        {
            "seller_id": ["A", "A", "B"],
            "order_id": ["1", "2", "3"],
            "promised_ship_date": pd.to_datetime(
                ["2025-01-02", "2025-01-02", "2025-01-02"]
            ),
            "actual_ship_date": pd.to_datetime(
                ["2025-01-03", "2025-01-01", "2025-01-02"]
            ),
            "returned": [True, False, False],
            "rating": [4, 2, 5],
            "negative_feedback": [True, False, False],
            "is_high_risk": [True, True, False],
        }
    )

    metrics = calculate_seller_metrics(orders).set_index("seller_id")

    assert metrics.loc["A", "order_count"] == 2
    assert metrics.loc["A", "seller_rating"] == 3
    assert metrics.loc["A", "late_shipment_rate"] == 0.5
    assert metrics.loc["A", "return_ratio"] == 0.5
    assert metrics.loc["A", "negative_feedback_rate"] == 0.5
    assert bool(metrics.loc["A", "is_high_risk"])
    assert metrics.loc["B", "late_shipment_rate"] == 0


def test_pipeline_writes_metrics_evaluation_and_model(tmp_path):
    paths = run_pipeline(
        tmp_path,
        sellers=40,
        orders_per_seller=20,
        seed=11,
    )

    assert all(path.exists() for path in paths.values())
    seller_metrics = pd.read_csv(paths["seller_metrics"])
    evaluation = json.loads(paths["evaluation"].read_text())
    assert len(seller_metrics) == 40
    assert seller_metrics["risk_probability"].between(0, 1).all()
    assert 0 <= evaluation["accuracy"] <= 1
    assert 0 <= evaluation["roc_auc"] <= 1