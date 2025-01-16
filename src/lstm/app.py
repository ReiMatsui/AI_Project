import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lstm.lstm import train_solar_power_prediction, prepare_data
from src.process_data import merge_data_1
from loguru import logger

def plot_training_history(history):
    """学習履歴をプロットする関数"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Mean Absolute Error'))
    
    # Loss plot
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name="Training Loss"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name="Validation Loss"),
        row=1, col=1
    )
    
    # MAE plot
    fig.add_trace(
        go.Scatter(y=history.history['mae'], name="Training MAE"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_mae'], name="Validation MAE"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, width=800, title_text="Training History")
    return fig

def plot_predictions(y_true, y_pred, dates):
    """予測結果と実際の値をプロットする関数"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        mode='lines',
        name='Actual Values',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred,
        mode='lines',
        name='Predicted Values',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Solar Power Generation: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='DC Power',
        height=500
    )
    
    return fig

def main():
    st.title('太陽光発電量予測ダッシュボード')
    
    # サイドバーの設定
    st.sidebar.header('モデルパラメータ')
    sequence_length = st.sidebar.slider('シーケンス長（時間）', 4, 96, 96, 4)
    test_size = st.sidebar.slider('テストデータの割合', 0.1, 0.3, 0.2, 0.05)
    epochs = st.sidebar.slider('学習エポック数', 10, 100, 50, 10)
    
    # データのアップロード
    uploaded_file = st.file_uploader("genファイルをアップロード", type=['csv'])
    uploaded_file2 = st.file_uploader("senファイルをアップロード", type=['csv'])
    
    if (uploaded_file is not None) and (uploaded_file2 is not None):
        # データの読み込みと前処理
        df = pd.read_csv(uploaded_file)
        df2 = pd.read_csv(uploaded_file2)
        merged_df = merge_data_1(df, df2)
        
        st.subheader('データプレビュー')
        st.dataframe(merged_df.head())
        
        if st.button('モデルを学習'):
            with st.spinner('モデルを学習中...'):
                # モデルの学習
                model, scaler, history, evaluation = train_solar_power_prediction(
                    df, 
                    sequence_length=sequence_length,
                    test_size=test_size,
                    validation_split=0.2
                )
                
                # 学習履歴のプロット
                st.subheader('学習履歴')
                history_fig = plot_training_history(history)
                st.plotly_chart(history_fig)
                
                # テストデータでの予測
                X, y, _, features = prepare_data(df, sequence_length=sequence_length)
                train_size = int(len(X) * (1 - test_size))
                X_test = X[train_size:]
                y_test = y[train_size:]
                
                # 予測の実行
                y_pred = model.predict(X_test)
                
                # スケールを戻す
                y_pred = scaler.inverse_transform(
                    np.concatenate([y_pred, np.zeros((len(y_pred), 0))], axis=1)
                )[:, 0]
                
                y_test = scaler.inverse_transform(
                    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 0))], axis=1)
                )[:, 0]
                
                # 予測結果のプロット
                st.subheader('予測結果')
                logger.info(df.index)
                dates = df.index[train_size + sequence_length:train_size + sequence_length + len(y_test)]
                pred_fig = plot_predictions(y_test, y_pred.flatten(), dates)
                st.plotly_chart(pred_fig)
                
                # 評価指標の表示
                mae = np.mean(np.abs(y_test - y_pred.flatten()))
                mse = np.mean((y_test - y_pred.flatten())**2)
                rmse = np.sqrt(mse)
                
                st.subheader('評価指標')
                col1, col2, col3 = st.columns(3)
                col1.metric('MAE', f'{mae:.2f}')
                col2.metric('MSE', f'{mse:.2f}')
                col3.metric('RMSE', f'{rmse:.2f}')

if __name__ == '__main__':
    main()