# Stock Market Prediction Chatbot and Streamlit Application

This repository focuses on building a user-friendly interface and chatbot for stock market prediction using **Streamlit**. The system integrates predictive analytics and an interactive chatbot, making financial analysis accessible to users of all expertise levels.

---

## 📂 Project Structure

- **`app.py`**: The main entry point for the Streamlit application.
- **`models/chatbot.py`**: Handles chatbot logic for both generative and retrieval-based responses.
- **`models/predict_stock.py`**: Includes prediction models like ARIMA, LSTM, and Temporal Fusion Transformer.
- **`utils/data_fetching.py`**: Fetches stock data using the Yahoo Finance API.
- **`utils/visualization.py`**: Generates stock trend visualizations.
- **`utils/helper_functions.py`**: Contains utility functions for preprocessing, feature engineering, and error handling.
- **`assets/`**: Contains static files like `styles.css` and image assets.
- **`requirements.txt`**: Lists all dependencies.
- **`Procfile`**: For deployment to Heroku or other cloud platforms.

---

## ⚙️ Features

### Chatbot Integration
- **Generative Chatbot**: 
  - Uses a fine-tuned language model to handle vague or complex stock-related queries.
  - Examples:
    - *"What are the key drivers of Tesla's stock price?"*
    - *"Explain what moving averages mean for trading."*

- **Retrieval-based Chatbot**:
  - Provides direct links to financial articles and data-driven insights.
  - Examples:
    - *"Show me the latest news about Apple."*
    - *"What were Amazon's closing prices last week?"*

### Streamlit Application
- **Interactive Dashboard**:
  - Allows users to input stock tickers and explore historical data.
  - Users can adjust variables like moving averages, volume, and trading patterns.
- **Visualization**:
  - Displays trends, seasonal patterns, and stock forecasts.
  - Plots include candlestick charts, line charts for key indicators, and more.
- **Stock Market Forecasting**:
  - Implements ARIMA, LSTM, and TFT models for time series prediction.
  - Features adjustable weights for variables like Open, Close, Volume, etc.

---

## 📋 Requirements

- Python 3.8+
- Libraries:
  - `streamlit`
  - `yfinance`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies using:

```bash
pip install -r requirements.txt
