#必要なパッケージをインポート
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
import itertools
from pydantic import BaseModel
import os
import datetime
from models import SolarPowerData
from loguru import logger

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#必要データを読み込み、データテーブルを作成　※ここは、各自データの保管先のパスに変更下さい。
gen_1 = pd.read_csv('data/csv/Plant_1_Generation_Data.csv')
gen_2 = pd.read_csv('data/csv/Plant_2_Generation_Data.csv')
sen_1= pd.read_csv('data/csv/Plant_1_Weather_Sensor_Data.csv')
sen_2= pd.read_csv('data/csv/Plant_2_Weather_Sensor_Data.csv')

#PLANT_IDの削除(使用しないため)
# gen_1.drop('PLANT_ID', 1, inplace = True)
# sen_1.drop('PLANT_ID', 1, inplace = True)
# gen_2.drop('PLANT_ID', 1, inplace = True)
# sen_2.drop('PLANT_ID', 1, inplace = True)

def create_graph():
    #DATE_TIMEの日付を同一の形に変更（データにより、日付の表示形式が異なるため）
    gen_1['DATE_TIME']=pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')
    sen_1['DATE_TIME']=pd.to_datetime(sen_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    gen_2['DATE_TIME']=pd.to_datetime(gen_2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    sen_2['DATE_TIME']=pd.to_datetime(sen_2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')


    #34日間における照射量/発電量の推移測定のためのデータテーブル
    df_gen1 = gen_1.groupby('DATE_TIME').sum().reset_index()
    df_gen2 = gen_2.groupby('DATE_TIME').sum().reset_index()

    #グラフの準備
    fig,ax = plt.subplots(ncols=2,nrows=3,dpi=100,figsize=(20,20))
    # 1日の発電量の推移
    df_gen1.plot(x='DATE_TIME', y='DAILY_YIELD', ax=ax[0,0])
    df_gen2.plot(x='DATE_TIME', y='DAILY_YIELD', ax=ax[0,1])
    #照射量のグラフ
    sen_1.plot(x='DATE_TIME', y='IRRADIATION', ax=ax[1,0])
    sen_2.plot(x='DATE_TIME', y='IRRADIATION', ax=ax[1,1])
    #気温のグラフ(周辺温度とパネル上の温度)
    sen_1.plot(x='DATE_TIME', y='AMBIENT_TEMPERATURE', ax=ax[2,0])
    sen_1.plot(x='DATE_TIME', y='MODULE_TEMPERATURE', ax=ax[2,0])
    sen_2.plot(x='DATE_TIME', y='AMBIENT_TEMPERATURE', ax=ax[2,1])
    sen_2.plot(x='DATE_TIME', y='MODULE_TEMPERATURE', ax=ax[2,1])

    #グラフのタイトル追加
    ax[0,0].set_title('Daily yield by gen_1')
    ax[0,1].set_title('Daily yield by gen_2')
    ax[1,0].set_title('Daily irradiation by sen_1')
    ax[1,1].set_title('Daily irradiation by sen_1')
    ax[2,0].set_title('Daily temperature by sen_1')
    ax[2,1].set_title('Daily temperature by sen_1')

    plt.show()

def main():
    # 日付フォーマットの統一
    gen_1['DATE_TIME'] = pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')
    sen_1['DATE_TIME'] = pd.to_datetime(sen_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # データの結合
    merged_data = pd.merge(
        gen_1, sen_1,
        on=['DATE_TIME', 'PLANT_ID'],  # 結合キー
        how='inner'  # 共通するデータのみ結合
    )

    # 結果の確認
    logger.info(merged_data)

    # 結合データをモデルに変換
    solar_power_data_list = [
        SolarPowerData(
            date_time=row['DATE_TIME'].strftime('%Y-%m-%d %H:%M:%S'),
            plant_id=row['PLANT_ID'],
            source_key_power=row['SOURCE_KEY_x'],
            source_key_sensor=row['SOURCE_KEY_y'],
            dc_power=row['DC_POWER'],
            ac_power=row['AC_POWER'],
            daily_yield=row['DAILY_YIELD'],
            total_yield=row['TOTAL_YIELD'],
            ambient_temperature=row['AMBIENT_TEMPERATURE'],
            module_temperature=row['MODULE_TEMPERATURE'],
            irradiation=row['IRRADIATION'],
        )
        for _, row in merged_data.iterrows()
    ]

    # 結果の確認
    for data in solar_power_data_list:
        logger.info(data)
        
    # SolarPowerDataのリストを使ってデータを抽出
    irradiation_values = [data.irradiation for data in solar_power_data_list if data.irradiation is not None]
    dc_power_values = [data.dc_power for data in solar_power_data_list if data.dc_power is not None]

    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(irradiation_values, dc_power_values, color='blue', alpha=0.7, label='DC Power vs Irradiation')
    plt.title('Relationship Between Irradiation and DC Power')
    plt.xlabel('Irradiation (W/m²)')
    plt.ylabel('DC Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    main()