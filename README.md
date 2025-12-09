Hybrid Time-Series Volatility Forecasting Project
1. Overview

This project implements a hybrid forecasting framework combining classical econometric models (ARIMA, GARCH) with deep learning architectures (LSTM and Transformer) to predict daily stock market volatility.

The workflow follows the DSCC 475 methodology, including:

Data preprocessing

Exploratory analysis

Classical statistical modeling

Deep learning forecasting

Hybrid ensemble integration

All code is contained in the provided Jupyter/Colab notebook. Results are reproducible by following the steps below.

2. How to Run the Notebook

The notebook is designed to run in Google Colab.

Execution Steps

Open Google Colab and upload the project notebook.

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


Ensure datasets are stored at:

/content/drive/MyDrive/TSA/


Enable GPU:

Runtime → Change Runtime Type → Hardware Accelerator → GPU

Run the notebook sequentially from top to bottom.

3. Required Libraries

The notebook installs missing dependencies automatically.
Core libraries:

Time-Series and Econometrics

pandas

numpy

statsmodels

pmdarima

arch

Machine Learning / Deep Learning

tensorflow

keras

scikit-learn

Visualization

matplotlib

If manual installation is needed:

pip install numpy pandas matplotlib statsmodels pmdarima arch tensorflow scikit-learn

4. Dataset Files and Paths

Datasets used for modeling:

/content/drive/MyDrive/TSA/AAPL_clean.csv
/content/drive/MyDrive/TSA/TSLA_clean.csv
/content/drive/MyDrive/TSA/SPY_clean.csv


Each file contains:

Close — Adjusted closing price

LogReturn — Daily log returns

Volatility_21 — 21-day rolling standard deviation

Volatility_7 — 7-day rolling standard deviation

Volume — Trading volume

To load a dataset:

df = pd.read_csv(
    "/content/drive/MyDrive/TSA/AAPL_clean.csv",
    index_col=0, parse_dates=True
)

5. Steps to Reproduce the Results

The notebook sections mirror the analytical pipeline.

Week 1 — Data Preparation & Exploratory Analysis

Load and clean datasets

Generate engineered features

ADF, ACF, PACF diagnostics

Visualize trends and volatility clustering

Week 2 — Classical Models: ARIMA & GARCH

Fit ARIMA on log returns

Fit GARCH(1,1) for conditional volatility

Forecast returns and volatility

Compare predicted vs realized volatility

Compute RMSE for each ticker

Week 3 — Deep Learning: LSTM

Normalize volatility series

Construct 30-day input sequences

Train LSTM models for 1-step and multi-step forecasts

Evaluate using RMSE & directional accuracy

Generate comparison plots

Week 4 — Transformer & Hybrid Ensemble

Implement compact Transformer encoder

Train for each ticker

Align predictions across all models

Construct hybrid ensemble (non-negative regression)

Produce final evaluation tables & visualizations

6. Output & Reproducibility

The notebook automatically produces:

Forecast plots (ARIMA, GARCH, LSTM, Transformer)

RMSE comparison tables

Directional accuracy metrics

Ensemble weight tables

Final ensemble forecast visualizations

All results will reproduce consistently when executed in order.

7. Notes

Deep learning steps require GPU; statistical models run on CPU.

Update file paths if datasets are elsewhere.

Notebook assumes cleaned CSVs; raw data cleaning is documented separately.
