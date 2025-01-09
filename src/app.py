import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from loguru import logger
from src.multiple_regression import process_data, merge_data_1, merge_data_2, fit_model, evaluate_model, exclude_outliers

def main():
    # メインコード
    st.title("Solar Power Data Analysis")

    # CSVの読み込み
    gen_1 = pd.read_csv('data/csv/Plant_1_Generation_Data.csv')
    sen_1 = pd.read_csv('data/csv/Plant_1_Weather_Sensor_Data.csv')
    
    gen_2 = pd.read_csv('data/csv/Plant_2_Generation_Data.csv')
    sen_2= pd.read_csv('data/csv/Plant_2_Weather_Sensor_Data.csv')

    # データの結合
    merged_data_1 = merge_data_1(gen_data=gen_1, sen_data=sen_1)
    merged_data_2 = merge_data_2(gen_data=gen_2, sen_data=sen_2)
    
    # 処理したデータ
    processed_data_1 = process_data(merged_data_1)
    
    # モデルの構築と外れ値除外
    model, mse, y_test, y_pred = fit_model(processed_data_1)
    
    st.info("学習結果グラフを以下に示します")
    # グラフ1: 初期モデル
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Initial Predictions')
    ax1.set_xlabel('Actual DC Power')
    ax1.set_ylabel('Predicted DC Power')
    ax1.set_title('Initial Model Predictions vs Actual')
    ax1.legend()
    ax1.grid(True)
    
    st.pyplot(fig1)
    
    filtered_data_1 = exclude_outliers(processed_data_1, model, mse)
    refined_model, refined_mse, filtered_y_test, new_y_pred = fit_model(filtered_data_1)

    st.info("外れ値を除去した学習結果グラフを以下に示します")
    # グラフ2: 外れ値除外後のモデル
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(filtered_y_test, new_y_pred, color='green', alpha=0.7, label='Refined Predictions')
    ax2.set_xlabel('Actual DC Power')
    ax2.set_ylabel('Predicted DC Power')
    ax2.set_title('Refined Model Predictions vs Actual')
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
    
    evaluate_y, evaluate_y_pred = evaluate_model(model, filtered_data_1)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(evaluate_y, evaluate_y_pred, color='green', alpha=0.7, label='Refined Predictions')
    ax3.set_xlabel('Actual DC Power')
    ax3.set_ylabel('Predicted DC Power')
    ax3.set_title('Refined Model Predictions vs Actual')
    ax3.legend()
    ax3.grid(True)
    st.info("二つ目のデータセットに適用したグラフを以下に示します。")
    st.pyplot(fig3)
    
if __name__=="__main__":
    main()