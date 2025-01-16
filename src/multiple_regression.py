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
import sklearn
from models import SolarPowerData
from loguru import logger
import streamlit as st

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
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    
    st.info(f"Initial Model - MSE: {mse}")
    st.info(f"R2:{r2}")
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
    outliers = residuals >= threshold
    
    st.info(f"外れ値の数: {(~inliers).sum()}")
    st.info(f"外れ値を除外したデータ数: {inliers.sum()}")
    
    filtered_data = data[inliers]
    outlier_data = data[outliers]
    
    return filtered_data, outlier_data
    
def evaluate_model(model, data):    
    # 説明変数と目的変数
    X = data[['IRRADIATION', 'MODULE_TEMPERATURE']]
    y = data['DC_POWER']
    st.info(f"評価用データ数は{X.shape[0]}です")
    # モデルを使って予測
    y_pred = model.predict(X)

    # 評価指標の計算 (MSE)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y_true=y, y_pred=y_pred)
    st.info(f"MSE:{mse}")
    st.info(f"R2:{r2}")
    
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