import datetime
import pandas as pd
import sys
import numpy as np
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import math

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

def get_dollar_bars(time_bars, dollar_threshold, minites): #function credit to Max Bodoia

    # ドルバーのからのリストを作成
    dollar_bars = []

    # 出来高を初期化
    running_volume = 0

    # 高値と安値を初期化
    running_high, running_low = 0, math.inf

    for i in range(len(time_bars)):

        # 時間、始値、高値、安値、終値、出来高を取得
        next_timestamp, next_high, next_low, next_open, next_close, next_volume = [time_bars[i][k] for k in ['TimeStamp', 'High', 'Low', 'Open', 'Close', 'Volume']]

        # 始値と終値の平均価格
        midpoint_price = ((next_open)+(next_close))/2

        # 出来高と平均価格を用いて、バーのおおよそのドル建て出来高を取得
        dollar_volume = next_volume * midpoint_price

        # 高値と安値を更新
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # 次のバーのドル建て出来高で閾値を超えれば
        if dollar_volume + running_volume >= dollar_threshold:

            # ドルバーのタイムスタンプを、バーが閉じたタイムスタンプに設定する（つまり、ドルバーに含まれる最後の分単位のバーのタイムスタンプから1分後）
            bar_timestamp = next_timestamp + datetime.timedelta(minutes=minites)
        
            # 新しいドルバーを、タイムスタンプ、実行中の高値/安値、次の終値とともにドルバーのリストに追加
            dollar_bars += [{'timestamp': bar_timestamp, 'open': next_open, 'high': running_high, 'low': running_low, 'close': next_close}]

            # 出来高をリセット
            running_volume = 0

            # 高値と安値をリセット
            running_high, running_low = 0, math.inf

        # それ以外の場合は、実行中の出来高を増加する
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars
