import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')


#Визуализация ряда
def plot_series(DataFrame, a=1, x=15, y=5, color='blue', title='Time series'): 
    plt.figure(figsize=(x, y)) 
    if a == 1:
        plt.plot(DataFrame['Value'], color=color, label='Date', linewidth=2, markersize=15)  
    else:
        plt.plot(DataFrame['Value_pred'], color='red', label='Value_pred', linewidth=2, markersize=15)  
        plt.plot(DataFrame['Value'], color='blue', label='Value', linewidth=2, markersize=15)  
        
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

#Скользящие статистики
def rolling_mean(df, n):
    if (n == 2):
         print("Error::n")
    stride = int(n / 2) if (n % 2) else int(n / 2) - 1
    delete = 2 * stride if (n % 2) else stride + 2
    Values = df.reset_index().copy()
    for i in range(df["Value"].size - delete):
        Values["Value"][i + stride] = df.reset_index()["Value"][i:n + i].sum() / n
    Values = Values.drop(range(Values["Value"].size - 1, Values["Value"].size - delete, -1))
    Values = Values.drop(range(stride))
    Values = Values.set_index("Date")
    return Values
def centered_rolling_mean(df):
    Values = df.copy()
    Values = Values.reset_index()
    for i in range(df["Value"].size):
        Values["Value"][i + 1] = df["Value"][i:2 + i].sum() / 2
    Values = Values.reset_index()
    Values = Values.drop(0)
    Values = Values.set_index("Date")
    del Values["index"]
    return Values
def rolling_std(df, n):
    stride = int(n / 2) if (n % 2) else int(n / 2) - 1
    Values = df.reset_index().copy()
    Mean_Values = (rolling_mean(df, n)).reset_index().copy()
    stds = (rolling_mean(df, n)).reset_index().copy()
    for i in range(0, Mean_Values["Value"].size):
        for j in range(i, i + n):
            x = (Values["Value"][j] - Mean_Values["Value"][i]) ** 2
        stds["Value"][i] = (x/(n-1)) ** 0.5
    return rolling_mean(stds, 5)

def seasonal_component(df, n, model='additive'):
    Values = df.reset_index().copy()
    Values = centered_rolling_mean(rolling_mean(df,n))
    if (model == 'additive'):
        Values["Value"] = df["Value"][int(n / 2):df["Value"].size - int(n / 2)] - Values["Value"]
    else:
        Values["Value"] = df["Value"][int(n / 2):df["Value"].size - int(n / 2)] / Values["Value"]
    #средняя оценка сезонной компоненты    
    seas_com = np.zeros(n) 
    period = int(Values["Value"].size / n) 
    Values = Values.reset_index()
    for i in range(0, n):
        seas_com[i] = Values["Value"][i:Values["Value"].size:n].sum() / period
    #циклический сдвиг массива
    from collections import deque
    d = deque(seas_com)
    d.rotate(period)
    seas_com = np.array(list(d))
    Values = Values.set_index("Date")
    correction_factor = seas_com.sum() / n if model == 'additive' else n / seas_com.sum()
    adjusted_SC = seas_com - correction_factor if model == 'additive' else seas_com * correction_factor
    return adjusted_SC,Values

#Метод наименьших квадратов для нахождения коэффициентов T
def least_square_method(df, adjusted_SC, n):
    Values = df.reset_index().copy()
    for i in range(0, Values["Value"].size, n):
        for j in range(n):
            Values["Value"][j + i] -= adjusted_SC[j] 
    t = pd.DataFrame({ 
                    "Date": np.array(range(1,Values["Value"].size + 1)),
                    "Value":np.array(range(1,Values["Value"].size + 1))
    })
    t.set_index("Date") 
    n = Values["Value"].size
    const = n * (t["Value"] * t["Value"]).sum() - t["Value"].sum() * t["Value"].sum()
    a_1 = (n * ((Values["Value"] * t["Value"]).sum()) - Values["Value"].sum() * t["Value"].sum()) / const
    a_0 = (Values["Value"].sum() - a_1 * t["Value"].sum()) / n 
    return a_1, a_0

def get_trend(df, n):
    #Тренд
    #T=a_0+a_1*t
    SC, trend = seasonal_component(df, n)
    a_1, a_0 = least_square_method(df, SC, n)
    Values = df.copy()
    t = pd.DataFrame({ 
                        "Date": np.array(range(1,Values["Value"].size + 1)),
                        "Value":np.array(range(1,Values["Value"].size + 1))
    })
    t.reset_index()
    for i in range(Values["Value"].size):
        Values["Value"][i] = a_0 + a_1 * t["Value"][i]
    return Values

def seasonality(df, n, model='additive'):
    mod = df["Value"].size % n
    SC, value = seasonal_component(df, n, model)
    season = pd.Series(np.zeros(df["Value"].size))
    for i in range(0, df["Value"].size - mod, n):
        season[i:i + n] = SC[0:n]
    if (mod):
        season[df["Value"].size - mod: df["Value"].size] = SC[:mod]
    season_df = df.reset_index().copy()
    season_df["Value"] = season[:]
    return season_df

def seasonal_decompose(df, model='additive', trend='roll'):
    if (model == 'additive'):
        observed = df.copy()
        trend = rolling_mean(df, 12) if (trend == 'roll') else get_trend(df, 12) 
        seasonal = seasonality(df, 12).set_index("Date")
        resid = (df - (trend + seasonal)).dropna()
    if (model == 'multiplicative'):
        observed = df.copy()
        trend = rolling_mean(df, 12) if (trend == 'roll') else get_trend(df, 12) 
        seasonal = seasonality(df, 12, model='multiplicative').set_index("Date")
        resid = (df / (trend * seasonal)).dropna()
    return observed, trend, seasonal, resid