from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
import itertools
from pydantic import BaseModel
import os
from datetime import datetime, timedelta
import sklearn
from models import SolarPowerData
from loguru import logger
import streamlit as st

def process_data(data):
    # データのフィルタリング
    logger.info(f"データ数: {data.shape[0]}")

    filtered_data = data[
        (data['DC_POWER'] > 0) & (data['IRRADIATION'] > 0)
    ]
    st.info(f"データ数（条件適用後）: {filtered_data.shape[0]}")
    
    # 直近30日分のデータを選択
    last_date = filtered_data['DATE_TIME'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = filtered_data[
        (filtered_data['DATE_TIME'] >= start_date) & 
        (filtered_data['DATE_TIME'] <= last_date)
    ]
    st.info(f"データ数（直近30日分）: {recent_data.shape[0]}")
    logger.info(recent_data[['DC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']])
    return recent_data
        
def fit_model(data):
    # 説明変数と目的変数
    X = data[['IRRADIATION', 'MODULE_TEMPERATURE']]
    y = data['DC_POWER']
    
    # データを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.info(f"学習データ数:{X_train.shape[0]}, 評価用データ数:{X_test.shape[0]}")

    # 最小二乗法でモデルを構築
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # モデルの評価
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    st.info(f"Initial Model - MSE: {mse}")
    st.info(f"Coefficients (w1, w2): {model.coef_}")
    st.info(f"Intercept (b): {model.intercept_}")
    
    return model, mse, y_test, y_pred

def exclude_outliers(data, model: LinearRegression, mse):
    # 説明変数と目的変数
    X = data[['IRRADIATION', 'MODULE_TEMPERATURE']]
    y = data['DC_POWER']
    
    # 外れ値の閾値を計算 (MSEの25%以上)
    threshold = mse * 1.25
    y_pred = model.predict(X)
    
    # 外れ値を除外
    residuals = (y - y_pred) ** 2

    inliers = residuals < threshold
    
    st.info(f"外れ値の数: {(~inliers).sum()}")
    st.info(f"外れ値を除外したデータ数: {inliers.sum()}")
    
    filtered_data = data[inliers]
    return filtered_data
    
def merge_data_1(gen_data, sen_data):
    # 日付フォーマットを統一
    gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
    sen_data['DATE_TIME'] = pd.to_datetime(sen_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # データの結合
    merged_data = pd.merge(
        gen_data, sen_data,
        on=['DATE_TIME', 'PLANT_ID'],
        how='inner'
    )
    st.info(f"元データの数 (結合後): {merged_data.shape[0]}")
    return merged_data

def merge_data_2(gen_data, sen_data):
    # 日付フォーマットを統一
    gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    sen_data['DATE_TIME'] = pd.to_datetime(sen_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # データの結合
    merged_data = pd.merge(
        gen_data, sen_data,
        on=['DATE_TIME', 'PLANT_ID'],
        how='inner'
    )
    logger.info(f"元データの数 (結合後): {merged_data.shape[0]}")
    return merged_data

def evaluate_model(model, merged_data):    
    # 条件を満たすデータをフィルタリング
    filtered_data = merged_data[
        (merged_data['DC_POWER'] > 0) & (merged_data['IRRADIATION'] > 0)
    ]
    logger.info(f"評価データ数（条件適用後）: {filtered_data.shape[0]}")

    # 直近30日分のデータを選択
    last_date = filtered_data['DATE_TIME'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = filtered_data[
        (filtered_data['DATE_TIME'] >= start_date) & 
        (filtered_data['DATE_TIME'] <= last_date)
    ]
    logger.info(f"評価用データ数（直近30日分）: {recent_data.shape[0]}")
    logger.info(recent_data[['DC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']])
    
    # 説明変数と目的変数
    X = recent_data[['IRRADIATION', 'MODULE_TEMPERATURE']]
    y = recent_data['DC_POWER']
    st.info(f"評価用データ数は{X.shape[0]}です")
    # モデルを使って予測
    y_pred = model.predict(X)

    # 評価指標の計算 (MSE)
    mse = mean_squared_error(y, y_pred)
    st.info(f"MSE:{mse}")
    logger.info(f"評価結果 (Plant 2): MSE = {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Predictions')
    plt.xlabel('Actual DC Power')
    plt.ylabel('Predicted DC Power')
    plt.title('Model Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()
    return y, y_pred