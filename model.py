import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import plotly.graph_objs as go
from datetime import date, timedelta

def compute_technical_features(df):
    # Compute rolling mean, median, and standard deviation
    df['mean_close'] = df['Close'].rolling(window=7).mean()
    df['median_close'] = df['Close'].rolling(window=7).median()
    df['std_close'] = df['Close'].rolling(window=7).std()
    df['range_close'] = df['Close'].rolling(window=7).apply(lambda x: x.max() - x.min(), raw=True)
    
    # Compute moving averages
    df['ma_7'] = df['Close'].rolling(window=7).mean()
    df['ma_14'] = df['Close'].rolling(window=14).mean()
    
    # Compute exponential moving averages
    df['ema_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['ema_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    
    # Compute Bollinger Bands
    df['upper_band'] = df['ma_7'] + 2 * df['std_close']
    df['lower_band'] = df['ma_7'] - 2 * df['std_close']
    
    # Compute Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Compute Momentum
    df['momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Compute MACD
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Compute Average True Range (ATR)
    df['high_low'] = df['High'] - df['Low']
    df['high_prev_close'] = abs(df['High'] - df['Close'].shift(1))
    df['low_prev_close'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Compute Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Compute Rate of Change (ROC)
    df['roc'] = df['Close'].pct_change(periods=12) * 100
    
    # Compute On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    return df

def prediction(stock, n_days):
    df = yf.download(stock, period='60d')
    df.reset_index(inplace=True)
    df = compute_technical_features(df)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    df['Day'] = df.index
    
    # Prepare features
    features = ['Day', 'mean_close', 'median_close', 'range_close', 'std_close', 'ma_7', 'ma_14',
                'ema_7', 'ema_14', 'upper_band', 'lower_band', 'RSI', 'momentum', 'macd',
                'signal_line', 'atr', 'vwap', 'roc', 'obv']
    
    # Prepare training data
    X = df[features]
    Y = df[['Close']]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    
    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Set up GridSearchCV for SVR
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )
    
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train_scaled, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)
    
    # Train the best model
    best_svr.fit(x_train_scaled, y_train)
    
    # Prepare future data
    last_day = df['Day'].max()
    future_days = [[last_day + i] for i in range(1, n_days + 1)]
    future_df = pd.DataFrame(future_days, columns=['Day'])
    
    # Fill missing features
    for feature in features[1:]:
        future_df[feature] = df[feature].iloc[-1]  # Use the last known value
    
    future_X = future_df[features]
    
    # Fill NaN values in future data
    if future_X.isnull().values.any():
        future_X.fillna(method='bfill', inplace=True)
    
    # Scale the future data
    future_X_scaled = scaler.transform(future_X)
    
    # Predict future prices
    predicted_prices = best_svr.predict(future_X_scaled)
    
    # Generate future dates
    dates = [date.today() + timedelta(days=i) for i in range(1, n_days + 1)]
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predicted_prices, mode='lines+markers', name='Predicted Prices'))
    fig.update_layout(
        title=f"Predicted Close Price for Next {n_days} Days",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    
    return fig
