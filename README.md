Hybrid Time-Series Volatility Forecasting Project

1. Overview

- Hybrid model combining ARIMA, GARCH, LSTM, and Transformer.

- Predicts daily stock market volatility.

- Follows DSCC 475 methodology.

- Fully reproducible using the provided Colab notebook.

2. How to Run the Notebook

- Open Google Colab and upload the notebook.

- Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


- Store datasets in:

/content/drive/MyDrive/TSA/


- Enable GPU: Runtime → Change Runtime Type → GPU.

- Run the notebooks from top to bottom.

3. Required Libraries

* pandas

* numpy

* statsmodels

* pmdarima

* arch

* tensorflow

* keras

* scikit-learn

* matplotlib

Install manually if needed:

pip install numpy pandas matplotlib statsmodels pmdarima arch tensorflow scikit-learn

4. Dataset Files

* AAPL_clean.csv

* TSLA_clean.csv

* SPY_clean.csv

Path:

/content/drive/MyDrive/TSA/


Columns:

* Close

* LogReturn

* Volatility_21

* Volatility_7

* Volume

Load example:

df = pd.read_csv(
    "/content/drive/MyDrive/TSA/AAPL_clean.csv",
    index_col=0, parse_dates=True
)

5. Analysis Workflow
   
Week 1 — Data Preparation

- Load and clean datasets.

- Generate log returns and rolling volatility.

- Run ADF, ACF, and PACF tests.

- Visualize price trends and volatility clustering.

Week 2 — ARIMA and GARCH

- Fit ARIMA on log returns.

- Fit GARCH(1,1) for volatility.

- Forecast returns and volatility.

- Compare predictions with realized volatility.

- Compute RMSE.

Week 3 — LSTM

- Normalize volatility series.

- Create 30-day input sequences.

- Train LSTM models.

- Evaluate using RMSE and directional accuracy.

- Plot predictions.

Week 4 — Transformer and Hybrid Ensemble

- Implement a compact Transformer encoder.

- Train models for each ticker.

- Align ARIMA, GARCH, LSTM, Transformer predictions.

- Build a hybrid ensemble using non-negative regression.

- Generate final evaluation tables and plots.

6. Outputs Generated

- Forecast plots for all models.

- RMSE comparison tables.

- Directional accuracy metrics.

- Ensemble weight tables.

vFinal ensemble forecast visualizations.

7. Notes

- Deep learning steps require GPU.

- Update file paths if different.

- Assumes cleaned CSVs exist.
