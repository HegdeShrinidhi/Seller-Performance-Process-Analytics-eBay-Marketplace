# eBay Marketplace Seller Performance Analytics

# eBay Marketplace Seller Performance Analytics

A reproducible Python starter project for seller KPIs and a baseline seller-risk model. It generates synthetic order data so the pipeline can run before real marketplace data is available. The synthetic risk label is used only as the training target, not as a model feature.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python seller_analytics.py
```

The default run creates 100 synthetic sellers with 200 orders each. To change the sample size or output location:

```bash
python seller_analytics.py --sellers 250 --orders-per-seller 100 --seed 7 --output-dir data/output
```

The output directory contains:

- `orders.csv`: synthetic order-level source data.
- `seller_metrics.csv`: seller rating, late shipment rate, return ratio, negative feedback rate, and baseline risk scores.
- `evaluation.json`: holdout accuracy, ROC AUC, and classification report.
- `seller_risk_model.joblib`: fitted scikit-learn Random Forest model.

Late shipment rate is the share of orders shipped after the promised ship date. Return ratio and negative feedback rate are order-level proportions. Model evaluation on generated data is for pipeline validation only and is not evidence of real-world performance.

## Test

```bash
python -m pytest
```
