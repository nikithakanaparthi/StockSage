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

def validate_weights(weights):
    """Validate that weights sum to 10 and are all integers"""
    return sum(weights) == 10 and all(isinstance(w, int) for w in weights)

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Stock Sage")
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 48px; margin-top: -30px;'>
        Stock Sage
        </h1>
        """, 
        unsafe_allow_html=True
    )
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'TSLA', 'COST', 'WMT']
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    selected_ticker = st.selectbox("Select a stock to view data:", tickers, key='centered_dropdown')
    st.markdown("</div>", unsafe_allow_html=True)
    all_stock_data = fetch_sp500_data()
    col1, col2 = st.columns([1, 2])

    with col1:
        ticker_data = all_stock_data[all_stock_data['Ticker'] == selected_ticker]
        
        # Feature Weights Section
        st.subheader("Variable Weights: How to Choose Weights for Stock Market Variables")
        st.markdown("""
            In stock market predictions, different variables can be weighted based on their relevance to specific forecasting goals.
            
            **Open Price**: Reflects after-hours trading and overnight news, useful for intraday or short-term predictions.  
            **Close Price**: The most consolidated view of daily trading, often given higher weight in end-of-day predictions.  
            **Low Price**: Shows the session's support level, higher weight helps identify buy signals or dips.  
            **High Price**: Represents resistance levels, useful for recognizing overbought conditions.  
            **Volume**: Critical for trend analysis, volume spikes can indicate impending price changes.
        """)
        
        # Initialize session state for weights if not exists
        if 'weights' not in st.session_state:
            st.session_state.weights = {
                'Open Price': 2,
                'Close Price': 2,
                'Low Price': 2,
                'High Price': 2
            }

        # Create number inputs for weights
        total_weight = 0
        new_weights = {}
        
        for feature, default_weight in st.session_state.weights.items():
            weight = st.number_input(
                feature,
                min_value=0,
                max_value=10,
                value=default_weight,
                step=1,
                key=f"weight_{feature}"
            )
            new_weights[feature] = weight
            total_weight += weight
        
        # Display total weight and validation
        st.write(f"Total Weight: {total_weight}/10")
        if total_weight != 10:
            st.error("Total weight must equal 10.")
        else:
            st.success("Weights properly allocated!")
            st.session_state.weights = new_weights

    with col2:
        # Display the selected stock's data
        st.subheader(f"Stock data for {selected_ticker}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Price Chart", "Statistics"])
        
        with tab1:
            st.line_chart(ticker_data['Close'])
        
        with tab2:
            st.dataframe(ticker_data.describe())
        
        forecast_weeks = st.selectbox("Select how many weeks to forecast for:", list(range(1, 53)), index=3)

        st.subheader(f"Forecasting {forecast_weeks} weeks ahead for {selected_ticker}")
        st.markdown("""
            As we predict for further weeks ahead, the accuracy may go down""")
            
        if st.button("Generate Forecast"):
            st.info("Forecast generation will be implemented soon.")

    # Additional section with model descriptions in horizontal layout
    st.divider()
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

    model_col1, model_col2 = st.columns(2)

    # Split the models across the two columns
    for idx, (model_name, description) in enumerate(models_info.items()):
        if idx % 2 == 0:
            with model_col1:
                with st.expander(model_name):
                    st.markdown(description)
        else:
            with model_col2:
                with st.expander(model_name):
                    st.markdown(description)

if __name__ == "__main__":
    main()
