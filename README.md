README — Hybrid Time-Series Volatility Forecasting Project

1. Overview
This project implements a hybrid forecasting framework combining classical econometric models (ARIMA, GARCH) with deep learning architectures (LSTM and Transformer) to predict daily stock market volatility. The workflow follows the prescribed methodology in DSCC 475, including data preprocessing, exploratory analysis, statistical modeling, deep learning forecasting, and hybrid ensemble integration.
All code is contained in the provided Colab notebook, and results can be reproduced following the instructions below.

2. How to Run the Notebook
The analysis is designed to run in Google Colab.
Execution Steps:
	1	Open Google Colab and upload the project notebook.
	2	Mount Google Drive (required to access dataset files):

from google.colab import drive
drive.mount('/content/drive')

  3	Ensure that the dataset files are located in the directory:
/content/drive/MyDrive/TSA/

  4	Change runtime to GPU to enable deep learning model training:
Runtime → Change Runtime Type → Hardware Accelerator: GPU

  5	Run the notebook sequentially from top to bottom. Each section is self-contained and will reproduce all numerical results, plots, and tables.

3. Required Libraries
The notebook installs missing dependencies automatically. The analysis relies on the following core libraries:
Time-Series and Econometrics
	•	pandas
	•	numpy
	•	statsmodels
	•	pmdarima
	•	arch
Machine Learning / Deep Learning
	•	tensorflow
	•	keras
	•	scikit-learn
Visualization
	•	matplotlib
If manual installation is required:

pip install numpy pandas matplotlib statsmodels pmdarima arch tensorflow scikit-learn

4. Dataset Files and Paths
The cleaned datasets used for modeling are stored as:

/content/drive/MyDrive/TSA/AAPL_clean.csv
/content/drive/MyDrive/TSA/TSLA_clean.csv
/content/drive/MyDrive/TSA/SPY_clean.csv

Each file contains:
	•	Close — Adjusted closing price
	•	LogReturn — Daily log returns
	•	Volatility_21 — 21-day rolling standard deviation
	•	Volatility_7 — 7-day rolling standard deviation
	•	Volume — Trading volume
  
To load a dataset:

df = pd.read_csv("/content/drive/MyDrive/TSA/AAPL_clean.csv",
                 index_col=0, parse_dates=True)

5. Steps to Reproduce the Results
The notebook is organized into structured sections that mirror the analytical pipeline.

Week 1 — Data Preparation and Exploratory Analysis
	•	Load and clean datasets.
	•	Generate engineered features (log returns, rolling volatility).
	•	Perform ADF, ACF, and PACF diagnostics.
	•	Visualize price trends and volatility clustering.
	
Week 2 — Classical Models: ARIMA and GARCH
	•	Fit ARIMA models on log returns.
	•	Fit GARCH(1,1) models to capture conditional volatility.
	•	Forecast returns and volatility.
	•	Compare predicted and realized volatility.
	•	Compute RMSE for each ticker.
	
Week 3 — Deep Learning Models: LSTM
	•	Normalize volatility series.
	•	Construct 30-day input sequences.
	•	Train LSTM models for t+1 and multi-step forecasts.
	•	Evaluate predictions using RMSE and directional accuracy.
	•	Generate comparative plots.
	
Week 4 — Transformer Model and Hybrid Ensemble
	•	Implement a compact Transformer encoder for volatility forecasting.
	•	Train models for each ticker.
	•	Align predictions across ARIMA, GARCH, LSTM, and Transformer.
	•	Construct a hybrid weighted ensemble using non-negative linear regression.
	•	Produce final evaluation metrics and ensemble performance tables.

7. Output and Reproducibility
The notebook automatically produces:
	•	Forecast plots for ARIMA, GARCH, LSTM, and Transformer models
	•	Comparative RMSE tables
	•	Directional accuracy metrics
	•	Hybrid ensemble weight tables
	•	Ensemble forecast visualizations
All results are generated using deterministic steps and will reproduce consistently when executed in order.

8. Notes
	•	Deep learning sections require GPU acceleration; statistical models run on CPU.
	•	All file paths must be updated if datasets are stored in a different location.
	•	The notebook assumes cleaned CSV files are available; raw data cleaning is documented separately.
