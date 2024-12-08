import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(layout="wide")


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


@st.cache_data
def load_data():
    original_data_df = pd.read_csv('single_stock_original_data.csv', parse_dates=['Date'], index_col='Date')
    forecast_df = pd.read_csv('forecast_results .csv', parse_dates=['Date'], index_col='Date')
    return original_data_df, forecast_df


def plot_forecast_plotly(forecast_df, original_data_df, forecast_weeks):
    original_close = original_data_df['Close'].tail(100)
    forecast_df.index += pd.DateOffset(days=1)
    forecast_weeks = int(forecast_weeks)
    forecast_df = forecast_df.head(forecast_weeks)
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

variable_info = """
**Variable Weights: How to Choose Weights for Stock Market Variables**

In stock market predictions, different variables can be weighted based on their relevance to specific forecasting goals.

- **Open Price**: Reflects after-hours trading and overnight news.
- **Close Price**: Consolidated view of daily trading.
- **Low Price**: Session's support level, identifying buy signals.
- **High Price**: Represents resistance levels.
"""


def main():
    st.title("Stock Sage")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'TSLA', 'COST', 'WMT']
    selected_ticker = st.selectbox("Select a stock to view data:", tickers)
    all_stock_data = get_sp500_data()
    ticker_data = all_stock_data[all_stock_data['Ticker'] == selected_ticker]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "Statistics", "Model Info", "Variable Info", "Metrics Info"])

    with tab1:
        st.subheader(f"Price Chart for {selected_ticker}")
        st.line_chart(ticker_data['Close'])

    with tab2:
        st.subheader(f"Statistics for {selected_ticker}")
        st.dataframe(ticker_data.describe())

    with tab3:
        st.subheader("Forecasting Models")
        for model_name, description in models_info.items():
            with st.expander(model_name):
                st.markdown(description)

    with tab4:
        st.subheader("Variable Information")
        st.markdown(variable_info)

    with tab5:
        st.subheader("Metrics Information")
        st.markdown("""
        ### sMAPE (Symmetric Mean Absolute Percentage Error)
        Measures the percentage error between forecast and actual values, normalized by their magnitude. 
        In stock market forecasting, it handles varying stock prices better than raw errors and balances over-prediction and under-prediction. 
        However, it is less effective for stocks with frequent zero or near-zero prices.

        **Use Case:** Comparing predictions across stocks with widely varying price levels, such as a mix of blue-chip stocks and smaller firms.

        ### MAE (Mean Absolute Error)
        Measures the average magnitude of absolute errors and does not penalize large errors as heavily as RMSE. 
        Suitable for analyzing the typical forecast error and doesn't emphasize extreme mispredictions.

        **Use Case:** Predicting relatively stable stocks or when extreme outliers aren't a primary concern.

        ### RMSE (Root Mean Squared Error)
        Measures the square root of the average squared errors, penalizing larger errors more heavily than MAE. 
        Highlights extreme forecast misses and focuses on accuracy for high-magnitude movements.

        **Use Case:** Ideal for volatile stocks where sharp movements significantly impact model performance.

        ### SPL (Scaled Pinball Loss)
        Quantile-based loss for upper/lower bound predictions. Penalizes forecasts outside confidence intervals and is important for probabilistic forecasts like VaR.

        **Use Case:** Creating robust predictions with upper/lower bounds, especially in options trading or hedging strategies.

        ### Containment
        Measures how often actual prices fall within the predicted confidence intervals. 
        Offers feedback on the reliability of the model's uncertainty estimation and penalizes overly narrow predictions.

        **Use Case:** Evaluating model reliability for regulatory reporting or human-readable evaluations.

        ### MADE (Mean Absolute Differential Error)
        Measures the magnitude of changes in forecasted vs. actual time series. 
        Focuses on capturing price movements rather than price levels and encourages replicating the stock’s trendiness.

        **Use Case:** Optimizing predictions for momentum-based strategies or swing trading.

        ### MAGE (Mean Absolute aGgregate Error)
        Measures aggregate forecast errors over grouped data. 
        Valuable for portfolios or indices to avoid systematic over/under-prediction.

        **Use Case:** Best for fund managers tracking portfolio performance or index-level predictions.

        ### MLE and iMLE
        MLE penalizes under-predictions, while iMLE penalizes over-predictions. 
        Helps control systematic bias, making them suitable for conservative or bullish forecasting scenarios.

        **Use Case:** Applications like pricing derivatives where under/overestimation risks differ.

        ### Contour
        Evaluates how well the forecast matches the directional movements of the actual data. 
        Ensures the forecast aligns with market trends, even if the magnitude isn't perfectly accurate.

        **Use Case:** High-level forecasts or when visual alignment with actual prices is important.
        """)

    st.divider()  # Separator for inputs below tabs

    # Forecasting Inputs
    st.subheader("Forecast Settings")

    # Multiselect for model selection
    selected_models = st.multiselect(
        "Select up to 4 forecasting models:",
        list(models_info.keys()),
        max_selections=4
    )

    # Manual weight inputs for variables (Open, Close, High, Low)
    st.subheader("Assign Weights to Variables (Sum ≤ 10)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        open_weight = st.number_input("Open Weight:", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="open")
    with col2:
        close_weight = st.number_input("Close Weight:", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="close")
    with col3:
        high_weight = st.number_input("High Weight:", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="high")
    with col4:
        low_weight = st.number_input("Low Weight:", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="low")

    total_variable_weight = open_weight + close_weight + high_weight + low_weight

    if total_variable_weight > 10:
        st.error("The sum of variable weights must not exceed 10. Please adjust the values.")
    else:
        # Metric weight inputs in three rows (3 metrics per row)
        st.subheader("Assign Weights to Metrics (Sum ≤ 10)")

        col5, col6, col7 = st.columns(3)
        with col5:
            smape_weight = st.number_input("sMAPE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                           key="smape")
        with col6:
            mae_weight = st.number_input("MAE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="mae")
        with col7:
            rmse_weight = st.number_input("RMSE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                          key="rmse")

        col8, col9, col10 = st.columns(3)
        with col8:
            spl_weight = st.number_input("SPL Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="spl")
        with col9:
            containment_weight = st.number_input("Containment Weight:", min_value=0.0, max_value=10.0, value=0.0,
                                                 step=0.1, key="containment")
        with col10:
            made_weight = st.number_input("MADE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                          key="made")

        col11, col12, col13 = st.columns(3)
        with col11:
            mage_weight = st.number_input("MAGE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                          key="mage")
        with col12:
            mle_weight = st.number_input("MLE Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="mle")
        with col13:
            contour_weight = st.number_input("Contour Weight:", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                             key="contour")

        # Calculate the total weight for metrics
        total_metric_weight = (smape_weight + mae_weight + rmse_weight + spl_weight + containment_weight +
                               made_weight + mage_weight + mle_weight + contour_weight)

        if total_metric_weight > 10:
            st.error("The sum of metric weights must not exceed 10. Please adjust the values.")
        else:
            forecast_weeks = st.number_input("Number of Weeks to Forecast:", min_value=1, max_value=52, value=4)

            if st.button("Generate Forecast"):
                st.write(f"Generating forecast using models: {', '.join(selected_models)}...")
                st.write(
                    f"Variable weights: Open={open_weight}, Close={close_weight}, High={high_weight}, Low={low_weight}")
                st.write(
                    f"Metric weights: sMAPE={smape_weight}, MAE={mae_weight}, RMSE={rmse_weight}, SPL={spl_weight}, Containment={containment_weight}, MADE={made_weight}, MAGE={mage_weight}, MLE={mle_weight}, Contour={contour_weight}")
                # Add your forecasting logic here
                original_data_df, forecast_df = load_data()
                plot_forecast_plotly(forecast_df, original_data_df, forecast_weeks)

    st.divider()
    st.subheader("Chatbot")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box
    user_input = st.text_input("Type your question here...", key="chat_input")
    if user_input:
        # Save user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate chatbot response (dummy example)
        response = f"You said: {user_input}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Clear input box
        st.session_state.chat_input = ""

if __name__ == "__main__":
    main()

