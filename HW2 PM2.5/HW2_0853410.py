import pandas as pd
import numpy as np

# load xls file
df_pm25 = pd.read_excel('./107年新竹站_20190315.xls')

# 把中文的column重新命名
df_pm25.rename(columns={'日期': 'date', '測站': 'station', '測項': 'test_item'}, inplace=True)

# 1-a. 取出10.11.12月資料
df_pm25['date'] = pd.to_datetime(df_pm25['date']).dt.normalize()  # 只顯示日期, 但不改變dtype
df_pm25 = df_pm25.loc[(df_pm25['date'] >= '2018-10-1') & (df_pm25['date'] <= '2018-12-31')].reset_index(drop=True)

# 1-b. check 缺失值以及無效值
# 找出含 #, *, x 的數值 (A 係指因儀器疑似故障警報所產生的無效值)
df_pm25.iloc[:, 3:] = df_pm25.iloc[:, 3:].replace(regex=r'[-]?(\d+(\.\d+)?)[*#xA]', value=np.nan)
# 1-c. NR表示無降雨，以0取代 (為方便補異常值，故先轉)
df_pm25.replace('NR', 0, inplace=True)
# 1-e.  製作時序資料 (為方便補異常值，故先做)
df_times = pd.DataFrame(np.hstack(np.reshape(df_pm25.iloc[:, 3:].values, (-1, 18, len(df_pm25.iloc[:, 3:].columns)))))

# 以前後一小時平均值取代 (axis=1 取左右)
df_times_T = (df_times.astype(float).ffill(axis=1) + df_times.astype(float).bfill(axis=1)) / 2

print(df_times_T.isna().sum())

# 1-d. 取10, 11月當training set, 12月當testing set
# df_train = df_pm25.loc[(df_pm25['date'] >= '2018-10-1') & (df_pm25['date'] <= '2018-11-30')].reset_index(drop=True)

# train共61天
df_train = df_times_T.iloc[:, :24*61]
# test為剩下的31天
df_test = df_times_T.iloc[:, 24*61:]

# 2-a. 每天有18個測項

