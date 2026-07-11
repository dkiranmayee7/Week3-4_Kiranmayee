# Sales Forecasting System

A Streamlit-based interactive dashboard for sales data analysis, forecasting, anomaly detection, and product demand segmentation.

## Features

- **Sales Overview** – KPIs, monthly trends, regional/category breakdowns, and interactive filters
- **Forecast Explorer** – 3-month Prophet forecasts, segment-level forecasts (by Category/Region), and model performance metrics (MAE, RMSE)
- **Anomaly Report** – Weekly anomaly detection using Isolation Forest and Z-Score methods with visual identification of holiday-driven spikes
- **Product Segments** – K-Means clustering (with PCA visualization) of sub-categories into demand segments: High Volume / Stable Demand, Growing Demand, Low Volume / High Volatility, and Declining Demand, with recommended stocking strategies

## Tech Stack

- **Frontend**: Streamlit
- **Forecasting**: Prophet (by Facebook)
- **Anomaly Detection**: Isolation Forest, Z-Score
- **Clustering**: K-Means, PCA
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, xgboost

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File Structure

```
SalesForecasting_Kiranmayee/
├── app.py              # Streamlit dashboard
├── analysis.ipynb      # Jupyter notebook with EDA
├── train.csv           # Sales transaction data
├── vgsales.csv         # Additional sales dataset
├── requirements.txt    # Python dependencies
├── summary.pdf         # Analysis summary report
└── charts/             # Generated visualizations
```

## Data

The system uses `train.csv` containing sales transactions with fields such as Order Date, Ship Date, Sales, Category, Sub-Category, Region, and Segment.
