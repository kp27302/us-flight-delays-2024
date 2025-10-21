# US Flight Delays 2024 - Machine Learning Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Data-Kaggle%20Hub-purple)](https://kaggle.com)

## 📊 Data Card

### Dataset Information
- **Name**: Flight Delay and Cancellation Data (1 Million Records, 2024)
- **Source**: [Kaggle Hub - nalisha/flight-delay-and-cancellation-data-1-million-2024](https://kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024)
- **Type**: Time Series, Tabular
- **Size**: ~1M records
- **Temporal Coverage**: 2024 (Full Year)
- **Geographic Coverage**: United States
- **Update Frequency**: Monthly

### Data Schema
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `flight_number` | Integer | Flight number | 1234 |
| `tail_number` | String | Aircraft tail number | N123AB |
| `carrier` | String | Airline carrier code | AA, DL, UA |
| `origin` | String | Origin airport code | ATL, LAX, ORD |
| `dest` | String | Destination airport code | JFK, SFO, SEA |
| `flight_date` | Date | Flight date | 2024-01-15 |
| `sched_dep_time` | Integer | Scheduled departure time (HHMM) | 1430 |
| `dep_time` | Integer | Actual departure time (HHMM) | 1445 |
| `arr_time` | Integer | Actual arrival time (HHMM) | 1720 |
| `dep_delay` | Integer | Departure delay in minutes | 15 |
| `weather_delay` | Integer | Weather-related delay (minutes) | 5 |
| `carrier_delay` | Integer | Carrier-related delay (minutes) | 3 |
| `late_aircraft_delay` | Integer | Late aircraft delay (minutes) | 7 |
| `cancelled` | Binary | Flight cancellation flag | 0, 1 |
| `diverted` | Binary | Flight diversion flag | 0, 1 |

### Data Quality
- **Completeness**: >95% for core fields
- **Accuracy**: Validated against official sources
- **Consistency**: Standardized formats across all records
- **Timeliness**: Updated monthly with 1-month lag
- **Bias**: Representative of US domestic flights

### Target Variable
- **Definition**: Binary classification of flight delays >15 minutes
- **Distribution**: ~35% delayed, ~65% on-time
- **Business Impact**: Critical for operational planning and customer satisfaction

## 🎯 Project Overview

This comprehensive analysis examines US flight delay patterns in 2024, combining exploratory data analysis with machine learning to predict flight delays. The project demonstrates end-to-end data science workflow from data ingestion to model deployment.

### Key Objectives
- Predict flight delays (>15 min) using historical flight data
- Identify key factors driving delay patterns
- Provide actionable insights for airline operations
- Demonstrate production-ready ML pipeline

### Target Audience
- Airline operations teams
- Data scientists and ML engineers
- Aviation analytics professionals
- Portfolio reviewers

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install kagglehub[hf-datasets] pandas numpy pyarrow scikit-learn lightgbm xgboost catboost shap optuna optuna-integration[lightgbm] matplotlib seaborn plotly tqdm jinja2
```

### Running the Analysis
1. Clone this repository
2. Open `us-flight-delays-2024-analysis.ipynb` in Jupyter
3. Run all cells sequentially
4. Review generated artifacts in `artifacts/` and `figures/` directories

## 📈 Methodology

### Data Processing Pipeline
1. **Data Ingestion**: Download from Kaggle Hub
2. **Data Cleaning**: Handle missing values, standardize formats
3. **Feature Engineering**: Create temporal, route, and congestion features
4. **Data Splitting**: Stratified 60/20/20 train/validation/test split
5. **Preprocessing**: Imputation, scaling, one-hot encoding

### Machine Learning Pipeline
1. **Baseline Models**: Logistic Regression, Random Forest, LightGBM
2. **Hyperparameter Optimization**: Optuna with 20 trials
3. **Model Selection**: ROC-AUC and F1-score optimization
4. **Threshold Tuning**: F1-score maximization
5. **Explainability**: SHAP analysis for feature importance

### Evaluation Metrics
- **ROC-AUC**: Model discrimination ability
- **PR-AUC**: Precision-recall balance
- **F1-Macro**: Balanced performance across classes
- **Accuracy**: Overall correctness
- **Brier Score**: Probability calibration

