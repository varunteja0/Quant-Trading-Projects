import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ✅ Data Collection with FIX for 'Adj Close' / 'Close'
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if 'Adj Close' in df.columns:
        df = df[['Adj Close']].rename(columns={'Adj Close': 'Price'})
    elif 'Close' in df.columns:
        df = df[['Close']].rename(columns={'Close': 'Price'})
    else:
        raise ValueError("Price column not found in data")
    return df

# ✅ SMA Strategy
def sma_strategy(df, short=50, long=200):
    df['SMA_Short'] = df['Price'].rolling(short).mean()
    df['SMA_Long'] = df['Price'].rolling(long).mean()
    df['SMA_Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)
    return df

# ✅ RSI Strategy
def rsi_strategy(df, window=14):
    delta = df['Price'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    return df

# ✅ Bollinger Bands Strategy
def bollinger_strategy(df, window=20):
    df['MA20'] = df['Price'].rolling(window).mean()
    rolling_std = df['Price'].rolling(window).std()
    if isinstance(rolling_std, pd.DataFrame):
        rolling_std = rolling_std.iloc[:, 0]
    df['Upper'] = df['MA20'] + 2 * rolling_std
    df['Lower'] = df['MA20'] - 2 * rolling_std
    price_series = df['Price'].squeeze()
    df['BB_Signal'] = np.where(price_series < df['Lower'], 1,
                               np.where(price_series > df['Upper'], -1, 0))
    return df

# ✅ ARIMA Forecasting
def arima_forecast(df, steps=5):
    model = ARIMA(df['Price'].dropna(), order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    print("ARIMA Forecast:", forecast.values)
    return forecast

# ✅ LSTM Forecasting
def lstm_forecast(df):
    series = df['Price'].ffill().values.reshape(-1, 1)
    series = (series - np.min(series)) / (np.max(series) - np.min(series))
    X, y = [], []
    window = 60
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    prediction = model.predict(X[-1].reshape(1, window, 1))
    print("LSTM Next Price Prediction:", prediction[0][0])

# ✅ Execute Strategy with Stop-Loss / Take-Profit
def execute_trading_logic(df):
    df['Signal'] = df[['SMA_Signal', 'RSI_Signal', 'BB_Signal']].sum(axis=1)
    df['Signal'] = np.where(df['Signal'] > 0, 1, np.where(df['Signal'] < 0, -1, 0))
    df['Returns'] = df['Price'].pct_change()
    stop_loss, take_profit = -0.02, 0.03
    df['Strategy_Returns'] = 0.0  # Set float type
    position = 0
    for i in range(1, len(df)):
        if df['Signal'].iloc[i - 1] != 0:
            position = df['Signal'].iloc[i - 1]
            ret = df['Returns'].iloc[i] * position
            if ret < stop_loss or ret > take_profit:
                position = 0
            df.loc[df.index[i], 'Strategy_Returns'] = ret
    return df

# ✅ Risk Metrics
def risk_metrics(df):
    sharpe = np.mean(df['Strategy_Returns']) / np.std(df['Strategy_Returns']) * np.sqrt(252)
    drawdown = (df['Price'].cummax() - df['Price']).max()
    var_95 = np.percentile(df['Strategy_Returns'].dropna(), 5)
    return float(sharpe), float(drawdown), float(var_95)

# ✅ Portfolio Optimization
def optimize_portfolio(tickers):
    prices = yf.download(tickers, start='2022-01-01', end='2024-01-01', auto_adjust=False)['Adj Close']
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()
    return cleaned, ef.portfolio_performance(verbose=True)

# ✅ Plot Equity Curve
def plot_equity(df):
    df['Equity'] = (1 + df['Strategy_Returns']).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Equity'])
    plt.title('AlphaEdge PRO Equity Curve')
    plt.show()

if __name__ == "__main__":
    ticker = 'AAPL'
    data = fetch_data(ticker, '2022-01-01', '2024-01-01')
    data = sma_strategy(data)
    data = rsi_strategy(data)
    data = bollinger_strategy(data)

    # Forecasting
    arima_forecast(data)
    lstm_forecast(data)

    # Trading Execution
    data = execute_trading_logic(data)

    # Risk Metrics
    sharpe, drawdown, var_95 = risk_metrics(data)
    print(f"Sharpe: {sharpe:.2f}, Max Drawdown: {drawdown:.2f}, VaR(95%): {var_95:.4f}")

    # Equity Curve
    plot_equity(data)

    # Portfolio Optimization Example
    weights, perf = optimize_portfolio(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
    print("Optimized Portfolio Weights:", weights)
    print("Portfolio Performance:", perf)
