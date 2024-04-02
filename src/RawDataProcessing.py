import numpy as np
import pandas as pd
from datetime import datetime

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", 1000)


tencent_raw_df = pd.read_excel("data/raw_data.xlsx", sheet_name="700 HK Option", skiprows=1)
hsbc_raw_df = pd.read_excel("data/raw_data.xlsx", sheet_name="5 HK Option", skiprows=1)
cm_raw_df = pd.read_excel("data/raw_data.xlsx", sheet_name="941 HK Option", skiprows=1)


def parse_option_chain(stock_code, spot_price):
    raw_df = pd.read_excel("data/raw_data.xlsx", sheet_name=f"{stock_code} Option", skiprows=1)
    raw_df = raw_df[["Strike", "Last", "IVM", "Last.1", "IVM.1"]]
    raw_df.columns = ["strike", "call_price", "call_implied_vol_bbg", "put_price", "put_implied_vol_bbg"]
    raw_df["stock_code"] = stock_code
    split_df = np.split(raw_df, raw_df[raw_df.isnull().any(axis=1)].index)[1:]
    total_df = pd.DataFrame()
    for df in split_df:
        info = df.iloc[0, 0]
        expire_date, _ = info.split(";")[0].split(" ")
        df["expire_date"] = datetime.strptime(expire_date, "%d-%b-%y").date()
        df = df.iloc[1:, :]
        total_df = pd.concat([total_df, df])
    today = datetime.strptime("2023-11-27", "%Y-%m-%d")
    total_df["biz_days_to_maturity"] = np.busday_count(today.date(), total_df["expire_date"].tolist())
    total_df["spot_price"] = spot_price
    return total_df


if __name__ == "__main__":
    total_df = pd.DataFrame()
    for pair in [("700 HK", 321.2), ("5 HK", 59.45), ("941 HK", 63.35)]:
        df = parse_option_chain(pair[0], pair[1])
        total_df = pd.concat([total_df, df])
    total_df.to_excel("data/option_chains.xlsx", index=False)
