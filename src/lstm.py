import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col='DC_POWER', sequence_length=96):
    """
    データの前処理を行う関数
    sequence_length=96 は 15分×96 = 24時間分のデータを使用
    """
    # 特徴量の選択 (例: 気象データと時間関連の特徴量)
    features = ['DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 
                'IRRADIATION', 'HUMIDITY']
    
    # 時間関連の特徴量を追加
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    features.extend(['hour', 'day_of_week'])
    
    # スケーリング
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    
    # シーケンスデータの作成
    X, y = [], []
    for i in range(len(scaled_df) - sequence_length):
        X.append(scaled_df.iloc[i:(i + sequence_length)].values)
        y.append(scaled_df.iloc[i + sequence_length][target_col])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape, dropout_rate=0.2):
    """
    LSTMモデルの構築
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=30, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_solar_power_prediction(df, target_col='DC_POWER', sequence_length=96, 
                               test_size=0.2, validation_split=0.2):
    """
    モデルの学習を実行する関数
    """
    # データの準備
    X, y, scaler = prepare_data(df, target_col, sequence_length)
    
    # トレーニングデータとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # モデルの構築
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    
    # モデルの学習
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=validation_split,
        verbose=1
    )
    
    return model, scaler, history

def predict_power(model, scaler, new_data, sequence_length=96):
    """
    新しいデータでの予測を行う関数
    """
    # データの前処理
    scaled_data = scaler.transform(new_data)
    X_pred = np.array([scaled_data[-sequence_length:]])
    
    # 予測
    pred_scaled = model.predict(X_pred)
    
    # スケールを戻す
    pred = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), 
        scaler.scale_.shape[0]-1))], axis=1)
    )[:, 0]
    
    return pred