import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
import streamlit as st


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
        