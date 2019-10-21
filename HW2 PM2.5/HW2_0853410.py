import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
# train共61天
df_train = df_times_T.iloc[:, :24*61]
# test為剩下的31天
df_test = df_times_T.iloc[:, 24*61:]


# 2-a. 取6小時為一單位切割
def data_generator(df, style):
    """
    產生每六小時的時序資料
    """
    target_matrix = np.zeros(df.shape[1]-6)
    # 2-b. 只要PM2.5
    if style == 'PM2.5':
        df_matrix = df.iloc[9, :].T
        feature_matrix = np.zeros((df.shape[1] - 6, 6))
        for index in range(df.shape[1]-6):
            feature_matrix[index, :] = df_matrix.values[index:index+6]
            target_matrix[index] = df_matrix.values[index+6]
    # 2-b. 18個屬性全要
    else:
        df_matrix = df.T
        feature_matrix = np.zeros((df.shape[1] - 6, 18*6))
        for index in range(df.shape[1]-6):
            feature_matrix[index, :] = df_matrix.iloc[index:index+6, :].values.flatten()
            target_matrix[index] = df.iloc[9, :].T.values[index + 6]

    return feature_matrix, target_matrix


def build_models(X, y, testX, testy):
    mae = []
    # 2-c. 建模linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(testX)
    # 2-d. 用MAE評估loss
    mae.append(mean_absolute_error(testy, y_pred))

    # 2-c. 建模random forest
    rf = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=4).fit(X, y)
    y_pred = rf.predict(testX)
    # 2-d. 用MAE評估loss
    mae.append(mean_absolute_error(testy, y_pred))

    return mae


# PM2.5
df_train_ohe, df_train_label = data_generator(df_train, 'PM2.5')
df_test_ohe, df_test_label = data_generator(df_test, 'PM2.5')

# 兩種模型的準確度
print(build_models(df_train_ohe, df_train_label, df_test_ohe, df_test_label))

# 18種屬性
df_train_ohe, df_train_label = data_generator(df_train, '18 attributes')
df_test_ohe, df_test_label = data_generator(df_test, '18 attributes')

print(build_models(df_train_ohe, df_train_label, df_test_ohe, df_test_label))
