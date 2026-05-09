# IPL Sports Analytics Platform

End-to-end IPL analytics project with:

- Real-time win probability prediction
- Match-level Logistic Regression baseline plus live-context Random Forest and XGBoost models
- K-Means player segmentation
- Interactive Streamlit dashboard for player, team, and venue analytics

The project works in two modes:

1. Real dataset mode: place IPL CSVs in `data/raw/`
2. Demo mode: automatically generates a realistic synthetic 2008-2024 IPL-style dataset and trains models

## Expected raw files

Place files here:

- `data/raw/matches.csv`
- `data/raw/deliveries.csv`

Recommended schemas are compatible with common IPL Kaggle datasets. The loader includes column normalization for typical variants.

## Quick start

```bash
python scripts/train_models.py
streamlit run app.py
```

If raw data is missing, the training script will generate demo data first.

## Features

- Live match simulator using target, overs remaining, wickets remaining, and run rate context
- Model comparison table covering the baseline and live-context classifiers
- Season-wise player batting and bowling analysis
- Venue scoring and chase/defend pattern heatmap
- Head-to-head team comparison with top performers
- Player clustering into roles such as anchors, power hitters, finishers, and economy bowlers
