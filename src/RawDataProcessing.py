import numpy as np
import pandas as pd
from datetime import datetime
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 1000)


tencent_raw_df = pd.read_excel('raw_data.xlsx', sheet_name='700 HK Option', skiprows=1)
hsbc_raw_df = pd.read_excel('raw_data.xlsx', sheet_name='5 HK Option', skiprows=1)
cm_raw_df = pd.read_excel('raw_data.xlsx', sheet_name="941 HK Option", skiprows=1)


def parse_option_chain(stock_code):
    raw_df = pd.read_excel('raw_data.xlsx', sheet_name=f'{stock_code} Option', skiprows=1)
    raw_df = raw_df[['Strike', 'Last', 'IVM', 'Last.1', 'IVM.1']]
    raw_df.columns = ['strike', 'call_price', 'call_implied_vol_bbg', 'put_price', 'put_implied_vol_bbg']
    raw_df['stock_code'] = stock_code
    split_df = np.split(raw_df, raw_df[raw_df.isnull().any(axis=1)].index)[1:]
    total_df = pd.DataFrame()
    for df in split_df:
        info = df.iloc[0, 0]
        expire_date, time_to_maturity_in_days = info.split(';')[0].split(' ')
        df['expire_date'] = datetime.strptime(expire_date, '%d-%b-%y')
        df['time_to_maturity'] = int(time_to_maturity_in_days.strip('(').strip(')').strip('d'))
        df = df.iloc[1:, :]
        total_df = pd.concat([total_df, df])
    return total_df


if __name__ == '__main__':
    total_df = pd.DataFrame()
    for code in ['700 HK', '5 HK', '941 HK']:
        df = parse_option_chain(code)
        total_df = pd.concat([total_df, df])
    total_df.to_excel('option_chains.xlsx')
