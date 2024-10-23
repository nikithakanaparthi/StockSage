# Import necessary libraries
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

# Function to fetch stock data
def fetch_sp500_data():
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'TSLA', 'COST', 'WMT']
    
    # Define the date range (last 10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    # Create an empty DataFrame to hold all stock data
    all_stock_data = pd.DataFrame()

    # Loop through the tickers and fetch their data
    for ticker in sp500_tickers:
        try:
            # Fetch the stock data for each ticker
            stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            stock_data['Ticker'] = ticker  # Add a column for the ticker
            all_stock_data = pd.concat([all_stock_data, stock_data])
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    return all_stock_data

# Streamlit app
def main():
    # App title
    st.title("Stock Sage")
    
    # Fetch the stock data
    all_stock_data = fetch_sp500_data()

    # List of available tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'TSLA', 'COST', 'WMT']

    # Add a dropdown to select a ticker
    selected_ticker = st.selectbox("Select a stock to view data:", tickers)

    # Filter data for the selected ticker
    ticker_data = all_stock_data[all_stock_data['Ticker'] == selected_ticker]

    # Display the selected stock's data
    st.subheader(f"Stock data for {selected_ticker}")
    st.dataframe(ticker_data)

    # Show summary statistics of the selected stock
    st.subheader(f"Summary statistics for {selected_ticker}")
    st.write(ticker_data.describe())

    # Line chart for the closing prices over time
    st.subheader(f"Closing Prices Over Time for {selected_ticker}")
    st.line_chart(ticker_data['Close'])

    # Dropdown for selecting the forecast horizon
    forecast_weeks = st.selectbox("Select how many weeks to forecast:", options=list(range(1, 53)), index=3)  # Default to 4 weeks

    st.subheader(f"Forecasting {forecast_weeks} weeks ahead for {selected_ticker}")
    st.write("This feature will be implemented soon.")

    # Model descriptions with expandable sections at the bottom of the page
    st.subheader("Forecasting Models")

    models_info = {
        "DynamicFactorMQ": """
        Overview: A statistical model used for multivariate time series forecasting. It's based on the Dynamic Factor Model (DFM) but extended with mixed frequencies (MQ) to handle datasets that have variables measured at different frequencies (e.g., daily stock prices, quarterly earnings reports).
        
        **Advantages:**
        - Mixed Frequencies: Handles variables with different frequencies (daily, weekly, monthly).
        - Latent Factor Modeling: Can capture common trends across multiple time series.
        - Good for Noise Reduction: Filters out noise and focuses on key underlying factors affecting stock movements.
        
        **Disadvantages:**
        - High Complexity: Complex to implement and requires careful tuning.
        - Slow with Large Data: Can be slow to estimate with a lot of variables.
        - May Not Capture Non-linearities: Struggles with complex non-linear dynamics.
        """,
        "PytorchForecasting": """
        Overview: A deep learning-based forecasting library built on PyTorch, capable of handling both univariate and multivariate time series using neural networks.
        
        **Advantages:**
        - Highly Flexible: Allows advanced neural architectures for forecasting.
        - Good at capturing non-linear relationships.
        - Handles complex feature extraction effectively.
        
        **Disadvantages:**
        - High Computational Cost: Requires significant computational power.
        - Overfitting Risk: Can easily overfit if not carefully tuned.
        - Black Box: Harder to interpret compared to traditional models.
        """,
        "MultivariateRegression": """
        Overview: This is a traditional regression model that extends linear regression to multiple variables, using the relationships between multiple time series to predict future values.
        
        **Advantages:**
        - Simplicity: Easy to understand and implement.
        - Fast: Computationally efficient.
        - Effective for linear relationships.
        
        **Disadvantages:**
        - Limited to Linear Relationships: Cannot capture non-linear dependencies.
        - Misses Complex Interactions: Fails to account for complex relationships.
        - Vulnerable to Overfitting: Can overfit if too complex.
        """,
        "NVAR (Nonlinear Vector Autoregression)": """
        Overview: NVAR is a variant of traditional vector autoregression (VAR) that focuses on capturing non-linear relationships.
        
        **Advantages:**
        - Non-linear Modeling: Captures non-linear dependencies between variables.
        - Multi-Asset: Can model multiple assets' prices simultaneously.
        - Good for Complex Markets: Works well in non-linear markets.
        
        **Disadvantages:**
        - Model Complexity: Complex to implement and interpret.
        - Data Hungry: Requires large amounts of data.
        - Risk of Overfitting: Prone to overfitting in volatile markets.
        """,
        "NeuralForecast": """
        Overview: A class of neural network models designed for time series forecasting, particularly suited for complex and non-linear patterns.
        
        **Advantages:**
        - Great for capturing non-linear relationships.
        - Scalability: Can handle multiple assets and indicators.
        - Long-term Dependencies: Captures long-term dependencies effectively.
        
        **Disadvantages:**
        - Requires Careful Tuning: Time-consuming hyperparameter tuning.
        - Computationally Intensive: Resource-heavy, requiring powerful hardware.
        - Risk of Overfitting: Can overfit, especially in noisy data.
        """,
        "MAR (Multivariate Adaptive Regression)": """
        Overview: A flexible regression technique that adapts to underlying data patterns, often used for complex relationships.
        
        **Advantages:**
        - Adapts to Data: Models complex, non-linear relationships.
        - Interpretable: More interpretable than deep learning models.
        - Good for Noisy Data: Handles noisy time series well.
        
        **Disadvantages:**
        - Limited Flexibility: May miss very complex dynamics.
        - Slow for Large Datasets: Can be computationally slow.
        - Not State-of-the-art: Lags behind advanced techniques in predictive power.
        """,
        "RollingRegression": """
        Overview: A regression model applied over a rolling window, recalculating parameters as new data comes in.
        
        **Advantages:**
        - Handles Changing Relationships: Useful when relationships change over time.
        - Simplicity: Simple to understand and implement.
        - Adapts Over Time: Adapts to new market trends.
        
        **Disadvantages:**
        - Limited by Window Size: Choice of window size is critical.
        - Fails with Non-linear Data: Struggles with non-linearities.
        - Short-term Focus: May miss long-term dynamics.
        """,
        "BallTreeMultivariateMotif": """
        Overview: A time-series motif discovery method using a BallTree structure to efficiently find similar patterns across multiple time series.
        
        **Advantages:**
        - Pattern Discovery: Can discover recurring patterns.
        - Efficient Search: Computationally efficient for large datasets.
        - Multivariate Focus: Handles multiple time series.
        
        **Disadvantages:**
        - Pattern Dependency: Assumes patterns repeat, which may not always hold.
        - Less Focus on Forecasting: More suited for motif discovery.
        - Not Suitable for Volatile Markets: Struggles in highly volatile markets.
        """
    }

    # Display model names with expandable descriptions
    for model_name, description in models_info.items():
        with st.expander(model_name):
            st.markdown(description)

# Run the app
if __name__ == '__main__':
    main()
