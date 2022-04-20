import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mport sys
import datetime
from tqdm import tqdm
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import math

# 分足データの取得
def make_data(code, day_length, minite_type):
    my_share = share.Share(code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                            day_length,
                                            share.FREQUENCY_TYPE_MINUTE,
                                            minite_type)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    data = symbol_data['timestamp']

    high = symbol_data['high']
    low = symbol_data['low']
    open = symbol_data['open']
    close = symbol_data['close']
    volume = symbol_data['volume']

    new_data = [datetime.datetime.utcfromtimestamp(int(data[i]/1000)) + datetime.timedelta(hours=9) for i in range(len(data))]
    all_data = np.array([new_data, high, low, open, close, volume]).T
    all_data = pd.DataFrame(all_data)

    all_data.columns = ['TimeStamp', 'High', 'Low', 'Open', 'Close', 'Volume']
    # all_data.to_csv(f'./stock_data/{code}.csv', mode='w', header=True, index=None)

    return all_data

# ドルバーの作成
def make_dollar_bar(df, threshold):
    df_dropped = df.dropna().reset_index()
    print('欠損値削除後', df_dropped.shape)
    data_length = df_dropped.shape[0]
    data_list  = []
    df_dropped_list = df_dropped.to_dict('record')
    price = 0
    for i in range(data_length):
        close_data = df_dropped_list[i]['Close']
        volume_data = df_dropped_list[i]['Volume']
        price += close_data * volume_data
        if price > threshold:
            data_list.append(df_dropped_list[i])
            price = 0
    dollar_df = pd.DataFrame(data_list)
    dollar_df.index = dollar_df['Date']
    return dollar_df

# ボリンジャーバンド
def bbands(close_prices, window, no_of_stdev):
    # rolling_mean = close_prices.rolling(window=window).mean()
    # rolling_std = close_prices.rolling(window=window).std()
    rolling_mean = close_prices.ewm(span=window).mean()
    rolling_std = close_prices.ewm(span=window).std()

    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band

# 日次ボラティリティの推定
def get_daily_vol(close, span0=100):
    
    # 日次ボラティリティ、close(株価系列)に従いインデックスを再作成
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    
    # 日次リターン
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    
    return df0

# 対称CUSUMフィルタ
def get_t_events(raw_price, threshold):
    print('Applying Symmetric CUSUM filter.')

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(raw_price).diff().dropna()

    # Get event time stamps for the entire series
    for i in tqdm(diff.index[1:]):
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    event_timestamps = pd.DatetimeIndex(t_events)
    return event_timestamps

# トリプルバリア法
def triple_barrier(df, take_profit=0.02, stop_loss=0.01, max_holding_period=5):
    prices = df["Close"].values
    label = [0]*len(prices)

    for n in range(len(prices)-max_holding_period):
        outcome = 0
        for m in range(1, max_holding_period):
            if prices[n+m]/prices[n] > 1+take_profit:
                outcome = take_profit
                break
            if prices[n+m]/prices[n] < 1-stop_loss:
                outcome = -stop_loss
                break
        if outcome == 0:
            outcome = prices[n+m]/prices[n]-1
        label[n] = outcome
    labels = pd.DataFrame(label, index=df.index, columns=["bin"])

    labels = labels.replace(0.02, 1)
    labels = labels.replace(0.00, 0)
    labels = labels.replace(-0.01, -1)
    
    return labels