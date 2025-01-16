import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from loguru import logger
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from multiple_regression.multiple_regression import fit_model, evaluate_model, exclude_outliers
from src.process_data import process_data, merge_data_1, merge_data_2

def main():
    st.title("Solar Power Data Analysis")

    # CSVの読み込み
    gen_1 = pd.read_csv('data/csv/Plant_1_Generation_Data.csv')
    sen_1 = pd.read_csv('data/csv/Plant_1_Weather_Sensor_Data.csv')

    # データの結合
    merged_data_1 = merge_data_1(gen_data=gen_1, sen_data=sen_1)
    processed_data_1 = process_data(merged_data_1)

    # モデルの構築と外れ値除外
    model, mse, y_test, y_pred = fit_model(processed_data_1)

    st.info("外れ値を含めて学習した学習結果グラフを以下に示します")
    # 初期モデルのグラフ
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='Initial Predictions'
    ))
    fig1.update_layout(
        title="Initial Model Predictions vs Actual",
        xaxis_title="Actual DC Power",
        yaxis_title="Predicted DC Power",
        legend=dict(title="Legend"),
        template="plotly_white"
    )
    st.plotly_chart(fig1)

    # 外れ値を除去したデータ
    filtered_data_1, outlier_data_1 = exclude_outliers(processed_data_1, model, mse)
    
    
    refined_model, refined_mse, filtered_y_test, new_y_pred = fit_model(filtered_data_1)

    st.info("外れ値を除去して学習した学習結果グラフを以下に示します")
    # 外れ値除外後のモデルのグラフ
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=filtered_y_test,
        y=new_y_pred,
        mode='markers',
        marker=dict(color='green', opacity=0.7),
        name='Refined Predictions'
    ))
    fig2.update_layout(
        title="Refined Model Predictions vs Actual",
        xaxis_title="Actual DC Power",
        yaxis_title="Predicted DC Power",
        legend=dict(title="Legend"),
        template="plotly_white"
    )
    st.plotly_chart(fig2)

    # 3Dグラフ
    refined_evaluate_y, refined_evaluate_y_pred = evaluate_model(refined_model, processed_data_1)

    fig3d = go.Figure()

    # 予測値 (緑の三角形)
    fig3d.add_trace(go.Scatter3d(
        x=processed_data_1['IRRADIATION'],
        y=processed_data_1['MODULE_TEMPERATURE'],
        z=refined_evaluate_y_pred,
        mode='markers',
        marker=dict(
            size=2, 
            color='#00AA00',  # 青寄りの緑
            opacity=0.1,
            symbol='diamond'  # 三角形
        ),
        name='Predictions'
    ))

    # 実測値 (鮮やかな赤の円)
    fig3d.add_trace(go.Scatter3d(
        x=processed_data_1['IRRADIATION'],
        y=processed_data_1['MODULE_TEMPERATURE'],
        z=refined_evaluate_y,
        mode='markers',
        marker=dict(
            size=2, 
            color='#FF0000',  # 鮮やかな赤
            opacity=0.1,
            symbol='circle'  # 円
        ),
        name='Actual Values'
    ))

    # グラフのレイアウトを設定
    fig3d.update_layout(
        title="Refined Model Predictions vs Actual (3D)",
        scene=dict(
            xaxis_title="IRRADIATION",
            yaxis_title="MODULE TEMPERATURE",
            zaxis_title="Pred DC Power"
        ),
        legend=dict(title="Legend"),
        template="plotly_white"
    )

    # Streamlitで表示
    st.info("外れ値を除去して学習したモデル")
    st.plotly_chart(fig3d)
    
        
    # 初期モデルのグラフ
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=outlier_data_1['AMBIENT_TEMPERATURE'],
        y=outlier_data_1['IRRADIATION'],
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='Initial Predictions'
    ))
    fig4.update_layout(
        title="DC POWER vs AC POWER",
        xaxis_title="DC Power",
        yaxis_title="AC Power",
        legend=dict(title="Legend"),
        template="plotly_white"
    )
    st.plotly_chart(fig4)


    # 初期モデルのグラフ
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=filtered_data_1['AMBIENT_TEMPERATURE'],
        y=filtered_data_1['IRRADIATION'],
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='Initial Predictions'
    ))
    fig5.update_layout(
        title="DC POWER vs AC POWER",
        xaxis_title="DC Power",
        yaxis_title="AC Power",
        legend=dict(title="Legend"),
        template="plotly_white"
    )
    st.plotly_chart(fig5)

if __name__ == "__main__":
    main()
