import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize
from functools import partial
from typing import Literal
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", 1000)


def cubic_splines_of_option_price_with_respect_to_maturity(stock_code: Literal["700 HK", "5 HK", "941 HK"], strike: float) -> (CubicSpline, CubicSpline):
    df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{stock_code}' "
                                                        f"and strike == {strike}")
    maturity = df['biz_days_to_maturity'] / 252
    cs_call = CubicSpline(maturity, df['call_price'])
    cs_put = CubicSpline(maturity, df['put_price'])
    return cs_call, cs_put


def cubic_splines_of_option_price_with_respect_to_strike(stock_code: Literal["700 HK", "5 HK", "941 HK"], expire_date: datetime) -> (CubicSpline, CubicSpline):
    df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{stock_code}' "
                                                        f"and expire_date == '{expire_date.strftime('%Y-%m-%d')}'")
    strike = df['strike']
    cs_call = CubicSpline(strike, df['call_price'])
    cs_put = CubicSpline(strike, df['put_price'])
    return cs_call, cs_put


def local_vol_calc(stock_code: Literal["700 HK", "5 HK", "941 HK"], strike: float, expire_date: datetime) -> (float, float, float):
    df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{stock_code}' "
                                                        f"and strike == {strike} "
                                                        f"and expire_date == '{expire_date.strftime('%Y-%m-%d')}'")
    maturity = df['biz_days_to_maturity'].iloc[0] / 252
    moneyness = strike / df['spot_price'].iloc[0]
    i = 0 if moneyness > 1 else 1
    dc_dt = derivative(cubic_splines_of_option_price_with_respect_to_maturity(stock_code, strike)[i], maturity, dx=1e-3*maturity, n=1)
    d2c_dk2 = derivative(cubic_splines_of_option_price_with_respect_to_strike(stock_code, expire_date)[i], strike, dx=0.05*strike, n=2)
    local_vol = 2 * dc_dt / strike**2 / d2c_dk2
    return local_vol


def fitting_local_vol_curve_along_moneyness(stock_code: Literal["700 HK", "5 HK", "941 HK"], expire_date: datetime):
    df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{stock_code}' "
                                                        f"and expire_date == '{expire_date.strftime('%Y-%m-%d')}'")
    maturity = df['biz_days_to_maturity'].iloc[0] / 252
    df['moneyness'] = df['strike'] / df['spot_price']
    df = df.query("0.8 < moneyness < 1.2")
    df['dupire_local_vol'] = df.apply(lambda row: local_vol_calc(stock_code=stock_code, strike=row.strike, expire_date=row.expire_date), axis=1)

    def fitting_function(parameters, x):
        sig2_0, delta, kappa, gamma = parameters
        return sig2_0 + delta * np.tanh(kappa * x) / kappa + gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2

    def calc_rss(parameters):
        return sum((df['dupire_local_vol'] - fitting_function(parameters, np.log(df['moneyness']))) ** 2)

    parameters = minimize(calc_rss, np.array([0.01, 0.5, 0.5, 0.5])).x

    def func(moneyness):
        return partial(fitting_function, parameters=parameters)(x=np.log(moneyness))

    return func


def local_vol_surface(stock_code, moneyness, maturity):
    df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{stock_code}' ")

    t = df[['expire_date', 'biz_days_to_maturity']].drop_duplicates()

    x = []
    y = []
    for idx, row in t[1:].iterrows():
        expire_date = row.expire_date
        time_to_maturity = row.biz_days_to_maturity / 252
        local_vol_of_moneyness = fitting_local_vol_curve_along_moneyness(stock_code, expire_date)(moneyness)
        x.append(time_to_maturity)
        y.append(local_vol_of_moneyness)
    cs = CubicSpline(x, y)

    local_vol = cs(maturity)
    return local_vol


if __name__ == '__main__':
    vol_surface = local_vol_surface(stock_code='700 HK', moneyness=np.arange(0.8, 1.2, 0.001), maturity=np.arange(0, 0.5, 1/252))
    print(vol_surface)
