# Stock Market Prediction and Forecasting Platform

This project integrates two major components: **S&P 500 Stock Data Forecasting** and a **Stock Market Prediction Chatbot and Streamlit Application**. The system enables users to perform predictive analytics, interact with a chatbot for stock-related queries, and visualize stock trends in an intuitive interface.

---

## 📂 Project Structure

### S&P 500 Stock Data Forecasting
- Fetches historical stock data for selected S&P 500 companies using the `yfinance` library.
- Implements the `AutoTS` library for time series forecasting with customizable models.
- Visualizes original and forecasted stock prices using `Plotly`.
- Outputs insights on forecasting model performance metrics.

### Stock Market Prediction Chatbot and Streamlit Application
- Provides a user-friendly interface for stock analysis via Streamlit.
- Integrates a chatbot with:
  - **Generative Responses**: Handles complex queries like "What are the key drivers of Tesla's stock price?"
  - **Retrieval-based Responses**: Provides quick answers and links to relevant stock information.
- Offers forecasting models (ARIMA, LSTM, TFT) for stock price predictions.
- Visualizes trends, seasonal patterns, and key stock metrics.

---

## ⚙️ Features

### S&P 500 Stock Data Forecasting
- **Data Collection**: Fetches historical stock data for tickers (e.g., `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`) over the last 10 years.
- **Customizable Forecasting Models**: Leverages `AutoTS` for time series forecasting with models like NVAR and NeuralForecast.
- **Weighted Metrics**: Configurable weights for `Open`, `Close`, `High`, and `Low` columns.
- **Visualization**: Plots original vs. forecasted values using `Plotly`.

### Stock Market Prediction Chatbot and Streamlit Application
- **Chatbot Integration**:
  - **Generative Chatbot**: Answers complex stock-related questions.
  - **Retrieval-based Chatbot**: Fetches specific financial articles and insights.
- **Interactive Dashboard**:
  - Users can explore historical stock data and analyze trends.
  - Provides adjustable parameters for trading patterns and moving averages.
- **Forecasting Models**:
  - Includes ARIMA, LSTM, and TFT models for stock market prediction.
  - Allows variable weighting for features like `Open`, `Close`, and `Volume`.

---

## 📋 Requirements

1. **Python 3.8+**
2. **Libraries**:
    - `yfinance==0.2.31`
    - `pandas==1.5.3`
    - `autots==0.5.3`
    - `plotly==5.15.0`
    - `streamlit==1.25.0`
    - `numpy==1.24.3`
    - `matplotlib==3.8.0`
    - `seaborn==0.12.2`
    - `scikit-learn==1.4.0`

Install dependencies using:
```bash
pip install -r requirements.txt
