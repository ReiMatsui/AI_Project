import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
import tensorflow as tf
from loguru import logger

def prepare_data(df, target_col='DC_POWER', sequence_length=96):
    """
    データの前処理を行う関数
    sequence_length=96 は 15分×96 = 24時間分のデータを使用
    """
    # DATE_TIMEの処理
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('DATE_TIME')
    
    # 重複インデックスの処理
    df = df[~df.index.duplicated(keep='first')]
    
    # データが時系列順になっているか確認し、ソート
    df = df.sort_index()
    
    # 特徴量の選択
    features = ['DC_POWER']
  
    # スケーリング
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    
    # シーケンスデータの作成
    X, y = [], []
    for i in range(len(scaled_df) - sequence_length):
        X.append(scaled_df.iloc[i:(i + sequence_length)].values)
        y.append(scaled_df.iloc[i + sequence_length][features.index(target_col)])
    
    logger.info(df)
    
    return np.array(X), np.array(y), scaler, df[features]

def create_lstm_model(input_shape, dropout_rate=0.2):
    """
    LSTMモデルの構築
    """
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=32, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', 
                 loss='mse',
                 metrics=['mae', 'mape'])
    return model

def train_solar_power_prediction(df, target_col='DC_POWER', sequence_length=96, 
                               test_size=0.2, validation_split=0.2):
    """
    モデルの学習を実行する関数
    """
    # データの準備
    X, y, scaler, processed_df = prepare_data(df, target_col, sequence_length)
    
    # トレーニングデータとテストデータの分割
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # モデルの構築
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Early Stopping の設定
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # モデルの学習
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # テストデータでの評価
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    test_pred = model.predict(X_test)
    
    evaluation = {
        'test_loss': test_loss,
        'test_predictions': test_pred.flatten(),
        'test_actual': y_test,
        'test_dates': processed_df.index[train_size + sequence_length:]
    }
    
    return model, scaler, history, evaluation

def predict_power(model, scaler, new_data, sequence_length=96):
    """
    新しいデータでの予測を行う関数
    """
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data.copy()
        if 'DATE_TIME' in new_data.columns:
            new_data['DATE_TIME'] = pd.to_datetime(new_data['DATE_TIME'], 
                                                 format='%d-%m-%Y %H:%M')
            new_data = new_data.set_index('DATE_TIME')
        
        # 時間特徴量の追加
        new_data['hour'] = new_data.index.hour
        new_data['day_of_week'] = new_data.index.dayofweek
        new_data['month'] = new_data.index.month
        new_data['day'] = new_data.index.day
        new_data['hour_sin'] = np.sin(2 * np.pi * new_data['hour']/24)
        new_data['hour_cos'] = np.cos(2 * np.pi * new_data['hour']/24)
    
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