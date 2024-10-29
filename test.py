import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Set Streamlit page configuration as the first command
st.set_page_config(layout="wide")

# Function to fetch stock data from S&P 500
@st.cache_data
def get_sp500_data():
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'TSLA', 'COST', 'WMT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    all_stock_data = pd.DataFrame()

    for ticker in sp500_tickers:
        try:
            stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            stock_data['Ticker'] = ticker
            all_stock_data = pd.concat([all_stock_data, stock_data])
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    return all_stock_data

# Load forecast and original data CSV files
@st.cache_data
def load_data():
    original_data_df = pd.read_csv('/Users/jenvithmanduva/Downloads/single_stock_original_data (1).csv', parse_dates=['Date'], index_col='Date')
    forecast_df = pd.read_csv('/Users/jenvithmanduva/Downloads/forecast_results (1).csv', parse_dates=['Date'], index_col='Date')
    return original_data_df, forecast_df

# Function to plot forecast using Plotly in Streamlit
def plot_forecast_plotly(forecast_df, original_data_df):
    original_close = original_data_df['Close'].tail(100)
    forecast_df.index += pd.DateOffset(days=1)  # Adjust the dates for forecast data
    forecast_close = forecast_df['Close']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=original_close.index,
        y=original_close,
        mode='lines+markers',
        name='Original Close Prices',
        marker=dict(size=8, color='blue'),
        line=dict(width=2, color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_close.index,
        y=forecast_close,
        mode='lines+markers',
        name='Forecasted Close Prices',
        marker=dict(size=10, color='orange', symbol='x'),
        line=dict(width=2, dash='dash', color='orange')
    ))

    fig.update_layout(
        title='Original vs Forecasted Close Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend_title='Legend',
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=2, linecolor='black'),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit app setup
def main():
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
    all_stock_data = get_sp500_data()
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
        
        if 'weights' not in st.session_state:
            st.session_state.weights = {
                'Open Price': 2,
                'Close Price': 2,
                'Low Price': 2,
                'High Price': 2
            }

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
        
        st.write(f"Total Weight: {total_weight}/10")
        if total_weight != 10:
            st.error("Total weight must equal 10.")
        else:
            st.success("Weights properly allocated!")
            st.session_state.weights = new_weights

    with col2:
        st.subheader(f"Stock data for {selected_ticker}")
        
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
            original_data_df, forecast_df = load_data()
            plot_forecast_plotly(forecast_df, original_data_df)

    # Model descriptions
    st.divider()
    st.subheader("Forecasting Models")

    models_info = {
        "DynamicFactorMQ": """
        Overview: A statistical model used for multivariate time series forecasting. Handles variables at different frequencies and captures common trends.

        **Advantages:** Mixed frequencies, latent factor modeling, noise reduction.
        **Disadvantages:** High complexity, slow with large data, struggles with non-linear dynamics.
        """,
        "PytorchForecasting": """
        Overview: A deep learning-based forecasting library built on PyTorch, capturing non-linear relationships and feature extraction.

        **Advantages:** Flexible neural architectures, non-linear relationships, effective feature extraction.
        **Disadvantages:** Computational cost, overfitting risk, harder interpretability.
        """,
        "MultivariateRegression": """
        Overview: This is a traditional regression model that extends linear regression to multiple variables, using the relationships between multiple time series to predict future values.
        
        **Advantages:** Simplicity, fast, effective for linear relationships.
        **Disadvantages:** Limited to linear dependencies, misses complex interactions, overfitting risk.
        """,
        "NVAR (Nonlinear Vector Autoregression)": """
        Overview: NVAR captures non-linear relationships, extending VAR models to complex markets.
        
        **Advantages:** Non-linear modeling, multi-asset focus, suitable for complex markets.
        **Disadvantages:** Model complexity, data intensive, overfitting risk in volatile markets.
        """,
        "NeuralForecast": """
        Overview: A neural network model that handles complex and non-linear patterns in time series forecasting.

        **Advantages:** Excellent for non-linear relationships, scalable, handles long-term dependencies.
        **Disadvantages:** Requires careful tuning, computationally intensive, overfitting risk in noisy data.
        """,
        "MAR (Multivariate Adaptive Regression)": """
        Overview: An adaptive regression technique that captures complex relationships.

        **Advantages:** Flexible to data, interpretable, handles noise well.
        **Disadvantages:** Limited to moderate complexities, slower for large datasets.
        """,
        "RollingRegression": """
        Overview: A regression model with rolling window adjustments for parameter recalculations.

        **Advantages:** Adapts to changing relationships, simple, responds to market changes.
        **Disadvantages:** Sensitive to window size, struggles with non-linearities, short-term focus.
        """,
        "BallTreeMultivariateMotif": """
        Overview: Uses BallTree to efficiently search for recurring patterns across multiple time series.
        
        **Advantages:** Pattern discovery, efficient search for large datasets, multivariate support.
        **Disadvantages:** Pattern dependency assumptions, not ideal for high volatility.
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

