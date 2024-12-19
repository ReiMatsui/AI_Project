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

def perform_multiple_regression(data):
    # 必要なデータを抽出
    X = data[['IRRADIATION', 'MODULE_TEMPERATURE']]  # 説明変数
    y = data['DC_POWER']  # 目的変数

    # 欠損値の確認と処理
    if X.isnull().values.any() or y.isnull().values.any():
        X = X.dropna()
        y = y[X.index]  # Xと同じインデックスで目的変数をフィルタリング

    # データを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの構築と学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    # モデルの予測
    y_pred = model.predict(X_test)

    # 結果の評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R-squared: {r2}")

    # 回帰係数と切片
    logger.info(f"Coefficients: {model.coef_}")
    logger.info(f"Intercept: {model.intercept_}")

    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label='Ideal Fit')
    plt.title('Predicted vs Actual DC Power')
    plt.xlabel('Actual DC Power (kW)')
    plt.ylabel('Predicted DC Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.show()

def filter_and_fit_model(data):
    # データのフィルタリング
    logger.info(f"データ数: {data.shape[0]}")

    filtered_data = data[
        (data['DC_POWER'] > 0) & (data['IRRADIATION'] > 0)
    ]
    logger.info(f"データ数（条件適用後）: {filtered_data.shape[0]}")
    # 直近30日分のデータを選択
    last_date = filtered_data['DATE_TIME'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = filtered_data[
        (filtered_data['DATE_TIME'] >= start_date) & 
        (filtered_data['DATE_TIME'] <= last_date)
    ]
    logger.info(f"データ数（直近30日分）: {recent_data.shape[0]}")
    logger.info(recent_data[['DC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']])
    
    # 説明変数と目的変数
    X = recent_data[['IRRADIATION', 'MODULE_TEMPERATURE']]
    y = recent_data['DC_POWER']
    
    # 最小二乗法でモデルを構築
    model = LinearRegression()
    model.fit(X, y)
    
    # モデルの評価
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    logger.info(f"Initial Model - MSE: {mse}")
    logger.info(f"Coefficients (w1, w2): {model.coef_}")
    logger.info(f"Intercept (b): {model.intercept_}")
    
    return recent_data, X, y, y_pred, model, mse

def exclude_outliers_and_refit(recent_data, X, y, y_pred, model, mse):
    # 外れ値の閾値を計算 (MSEの25%以上)
    threshold = mse * 2
    
    # 外れ値を除外
    residuals = (y - y_pred) ** 2
    inliers = residuals < threshold
    
    logger.info(f"外れ値の数: {(~inliers).sum()}")
    logger.info(f"外れ値を除外したデータ数: {inliers.sum()}")
    
    filtered_X = X[inliers]
    filtered_y = y[inliers]
    
    # 新しいモデルを再構築
    new_model = LinearRegression()
    new_model.fit(filtered_X, filtered_y)
    new_y_pred = new_model.predict(filtered_X)
    new_mse = mean_squared_error(filtered_y, new_y_pred)
    
    logger.info(f"Refined Model - MSE: {new_mse}")
    logger.info(f"Coefficients (w1, w2): {new_model.coef_}")
    logger.info(f"Intercept (b): {new_model.intercept_}")
    
    # 結果をプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Initial Predictions')
    plt.scatter(filtered_y, new_y_pred, color='green', alpha=0.7, label='Refined Predictions')
    plt.xlabel('Actual DC Power')
    plt.ylabel('Predicted DC Power')
    plt.title('Model Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return new_model, new_mse
    
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
    logger.info(f"元データの数 (結合後): {merged_data.shape[0]}")
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

    # モデルを使って予測
    y_pred = model.predict(X) *10

    # 評価指標の計算 (MSE)
    mse = mean_squared_error(y, y_pred)
    logger.info(f"評価結果 (Plant 2): MSE = {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Predictions')
    plt.xlabel('Actual DC Power')
    plt.ylabel('Predicted DC Power')
    plt.title('Model Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()
    return mse

def main():
    #必要データを読み込み、データテーブルを作成　※ここは、各自データの保管先のパスに変更下さい。
    gen_1 = pd.read_csv('data/csv/Plant_1_Generation_Data.csv')
    gen_2 = pd.read_csv('data/csv/Plant_2_Generation_Data.csv')
    sen_1= pd.read_csv('data/csv/Plant_1_Weather_Sensor_Data.csv')
    sen_2= pd.read_csv('data/csv/Plant_2_Weather_Sensor_Data.csv')
    # データの前処理

    merged_data_1 = merge_data_1(gen_data=gen_1, sen_data=sen_1)
    merged_data_2 = merge_data_2(gen_data=gen_2, sen_data=sen_2)
    
    # モデル構築と外れ値除外
    recent_data, X, y, y_pred, model, mse = filter_and_fit_model(merged_data_2)
    refined_model, refined_mse = exclude_outliers_and_refit(recent_data, X, y, y_pred, model, mse)
    evaluate_model(refined_model, merged_data_1)
    return refined_model, refined_mse

if __name__ == "__main__":
    refined_model, refined_mse = main()

