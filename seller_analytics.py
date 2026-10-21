"""Generate seller-performance analytics and a baseline risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "seller_rating",
    "late_shipment_rate",
    "return_ratio",
    "negative_feedback_rate",
]


def generate_orders(
    sellers: int = 100,
    orders_per_seller: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic, synthetic order-level data for development."""
    if sellers < 10:
        raise ValueError("sellers must be at least 10 to train and evaluate the model")
    if orders_per_seller < 1:
        raise ValueError("orders_per_seller must be positive")

    rng = np.random.default_rng(seed)
    high_risk_count = max(2, round(sellers * 0.2))
    high_risk_sellers = set(
        rng.choice(sellers, size=high_risk_count, replace=False).tolist()
    )
    start_date = pd.Timestamp("2025-01-01")
    seller_orders = []

    for seller_number in range(sellers):
        is_high_risk = seller_number in high_risk_sellers
        count = orders_per_seller
        order_dates = start_date + pd.to_timedelta(
            rng.integers(0, 365, size=count), unit="D"
        )
        promised_dates = order_dates + pd.Timedelta(days=2)
        late_probability = 0.30 if is_high_risk else 0.07
        late = rng.random(count) < late_probability
        shipping_lag = np.where(
            late,
            rng.integers(1, 6, size=count),
            rng.integers(-1, 2, size=count),
        )
        actual_dates = promised_dates + pd.to_timedelta(shipping_lag, unit="D")
        ratings = np.rint(
            np.clip(
                rng.normal(3.7 if is_high_risk else 4.6, 0.65, size=count),
                1,
                5,
            )
        ).astype(int)

        seller_orders.append(
            pd.DataFrame(
                {
                    "seller_id": f"S{seller_number + 1:04d}",
                    "order_id": [
                        f"O{seller_number + 1:04d}-{order_number + 1:05d}"
                        for order_number in range(count)
                    ],
                    "order_date": order_dates,
                    "promised_ship_date": promised_dates,
                    "actual_ship_date": actual_dates,
                    "returned": rng.random(count)
                    < (0.20 if is_high_risk else 0.06),
                    "rating": ratings,
                    "negative_feedback": rng.random(count)
                    < (0.14 if is_high_risk else 0.02),
                    "is_high_risk": is_high_risk,
                }
            )
        )

    return pd.concat(seller_orders, ignore_index=True)


def calculate_seller_metrics(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order records into seller KPIs and the synthetic target."""
    required = {
        "seller_id",
        "order_id",
        "promised_ship_date",
        "actual_ship_date",
        "returned",
        "rating",
        "negative_feedback",
        "is_high_risk",
    }
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Missing required order columns: {', '.join(sorted(missing))}")

    prepared = orders.assign(
        _late=orders["actual_ship_date"] > orders["promised_ship_date"]
    )
    return (
        prepared.groupby("seller_id", as_index=False)
        .agg(
            order_count=("order_id", "count"),
            seller_rating=("rating", "mean"),
            late_shipment_rate=("_late", "mean"),
            return_ratio=("returned", "mean"),
            negative_feedback_rate=("negative_feedback", "mean"),
            is_high_risk=("is_high_risk", "max"),
        )
        .sort_values("seller_id", ignore_index=True)
    )


def _new_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
    )


def run_pipeline(
    output_dir: str | Path,
    sellers: int = 100,
    orders_per_seller: int = 200,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate data, evaluate a baseline model, and write project artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    orders = generate_orders(sellers, orders_per_seller, seed)
    metrics = calculate_seller_metrics(orders)
    features = metrics[FEATURE_COLUMNS]
    target = metrics["is_high_risk"].astype(int)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=seed,
        stratify=target,
    )

    evaluation_model = _new_model(seed)
    evaluation_model.fit(train_features, train_target)
    test_probabilities = evaluation_model.predict_proba(test_features)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    evaluation = {
        "accuracy": float(accuracy_score(test_target, test_predictions)),
        "roc_auc": float(roc_auc_score(test_target, test_probabilities)),
        "classification_report": classification_report(
            test_target,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
        "feature_columns": FEATURE_COLUMNS,
        "training_note": "Synthetic data; evaluation is a baseline, not production evidence.",
    }

    model = _new_model(seed)
    model.fit(features, target)
    metrics["risk_probability"] = model.predict_proba(features)[:, 1]
    metrics["risk_flag"] = metrics["risk_probability"] >= 0.5

    paths = {
        "orders": output_path / "orders.csv",
        "seller_metrics": output_path / "seller_metrics.csv",
        "evaluation": output_path / "evaluation.json",
        "model": output_path / "seller_risk_model.joblib",
    }
    orders.to_csv(paths["orders"], index=False)
    metrics.to_csv(paths["seller_metrics"], index=False)
    paths["evaluation"].write_text(json.dumps(evaluation, indent=2) + "\n")
    joblib.dump(model, paths["model"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--sellers", type=int, default=100)
    parser.add_argument("--orders-per-seller", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = run_pipeline(
        args.output_dir,
        sellers=args.sellers,
        orders_per_seller=args.orders_per_seller,
        seed=args.seed,
    )
    print("Pipeline complete:")
    for artifact, path in paths.items():
        print(f"  {artifact}: {path}")


if __name__ == "__main__":
    main()