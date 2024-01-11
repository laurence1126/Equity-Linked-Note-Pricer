import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime
from matplotlib import pyplot as plt
from math import exp
from scipy.misc import derivative
from functools import partial
from typing import Literal
import warnings
from Curves import dividend_yield_curve, forward_rate_curve, yield_curve_interpolate
from scipy.optimize import curve_fit


warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", 1000)


class DupireLocalVolSurface:
    def __init__(self, stock_code: Literal["700 HK", "5 HK", "941 HK"]):
        self.stock_code = stock_code
        self.r = forward_rate_curve(yield_curve_interpolate())
        self.q = dividend_yield_curve(stock_code)
        self.total_df = self.total_df_func()
        self.total_df_with_local_vol_func()
        self.vol_surface = self.local_vol_surface(moneyness=np.arange(0.7, 1.3, 0.001), maturity=np.arange(0, 0.5, 1/252))

    def total_df_func(self):
        df = pd.read_excel('option_chains.xlsx').query(f"stock_code == '{self.stock_code}' ")
        df['time_to_maturity'] = df['biz_days_to_maturity'] / 252
        df['moneyness'] = df['strike'] / df['spot_price']
        df['growth_factor'] = df.apply(lambda row:  exp(sum(self.r[:row.biz_days_to_maturity] - self.q[:row.biz_days_to_maturity]) / 252), axis=1)
        df['forward_moneyness'] = df.apply(lambda row: row.strike / (row.growth_factor * row.spot_price), axis=1)
        return df

    def total_df_with_local_vol_func(self):
        self.total_df['dupire_local_vol'] = self.total_df.apply(lambda row: self.local_vol_calc(strike=row.strike, expire_date=row.expire_date), axis=1)
        self.total_df['dupire_local_vol'] = self.total_df['dupire_local_vol'].apply(lambda x: np.nan if x <= 0 or x > 0.25 else x)

    def cubic_splines_of_option_price_with_respect_to_maturity(self, strike: float) -> (CubicSpline, CubicSpline):
        df = self.total_df.query(f"strike == {strike} ").copy()
        cs_call = CubicSpline(df['time_to_maturity'], df['call_price'])
        cs_put = CubicSpline(df['time_to_maturity'], df['put_price'])
        return cs_call, cs_put

    def cubic_splines_of_option_price_with_respect_to_strike(self, expire_date: datetime) -> (CubicSpline, CubicSpline):
        df = self.total_df.query(f"expire_date == '{expire_date.strftime('%Y-%m-%d')}'").copy()
        cs_call = CubicSpline(df['strike'], df['call_price'])
        cs_put = CubicSpline(df['strike'], df['put_price'])
        return cs_call, cs_put

    def local_vol_calc(self, strike: float, expire_date: datetime) -> float:
        df = self.total_df.query(f"strike == {strike} and "
                                 f"expire_date == '{expire_date.strftime('%Y-%m-%d')}'").copy()
        time_to_maturity = df['time_to_maturity'].iloc[0]
        moneyness = df['moneyness'].iloc[0]
        i = 0 if moneyness > 1 else 1
        dc_dt = derivative(self.cubic_splines_of_option_price_with_respect_to_maturity(strike)[i], time_to_maturity,
                           dx=0.01 * time_to_maturity, n=1)
        d2c_dk2 = derivative(self.cubic_splines_of_option_price_with_respect_to_strike(expire_date)[i], strike,
                             dx=0.05 * strike, n=2)
        local_vol = 2 * dc_dt / strike ** 2 / d2c_dk2
        return local_vol

    def local_vol_curve_along_moneyness_axis(self, expire_date: datetime):
        df = self.total_df.query(f"expire_date == '{expire_date.strftime('%Y-%m-%d')}' and dupire_local_vol > 0 ").copy().reset_index(drop=True)
        atm_vol = df.iloc[(df['moneyness'] - 1).abs().idxmin()]['dupire_local_vol']

        def regression_func(x, delta, kappa, half_gamma):
            return atm_vol + delta * np.tanh(kappa * np.log(x)) / kappa + half_gamma * (np.tanh(kappa * np.log(x)) / kappa) ** 2

        x = np.array(df['forward_moneyness'])
        y = np.array(df['dupire_local_vol'])

        lower_bounds = [0, 0, 0]
        upper_bounds = [np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)

        initial_guess = [0.5, 0.5, 0.5]  # Initial guess for the parameters
        optimized_params, _ = curve_fit(regression_func, x, y, p0=initial_guess, bounds=bounds, maxfev=10000)

        delta_opt = optimized_params[0]
        kappa_opt = optimized_params[1]
        half_gamma_opt = optimized_params[2]

        def func(forward_moneyness):
            return partial(regression_func, delta=delta_opt, kappa=kappa_opt, half_gamma=half_gamma_opt)(x=forward_moneyness)

        return func

    def local_vol_surface(self, moneyness, maturity):
        df = self.total_df[['expire_date', 'time_to_maturity', 'growth_factor']].drop_duplicates()

        x = []
        y = []
        for idx, row in df.iterrows():
            expire_date = row.expire_date
            time_to_maturity = row.time_to_maturity
            forward_moneyness = moneyness / row.growth_factor
            local_vol = self.local_vol_curve_along_moneyness_axis(expire_date)(forward_moneyness)
            x.append(time_to_maturity)
            y.append(local_vol)
        cs = CubicSpline(x, y)
        local_vols = cs(maturity)
        return local_vols


if __name__ == '__main__':
    vol_surface = DupireLocalVolSurface(stock_code='700 HK').vol_surface
    vol_surface = DupireLocalVolSurface(stock_code='5 HK').vol_surface
    vol_surface = DupireLocalVolSurface(stock_code='941 HK').vol_surface

