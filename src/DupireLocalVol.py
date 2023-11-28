import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime
from matplotlib import pyplot as plt
from math import exp
from scipy.misc import derivative
from scipy.optimize import minimize
from functools import partial
from typing import Literal
import warnings
from VolSurface import dividend_yield_curve, forward_rate_curve, yield_curve_interpolate


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
        self.vol_surface = self.local_vol_surface(moneyness=np.arange(0.85, 1.15, 0.001), maturity=np.arange(0, 0.5, 1/252))

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        X = np.arange(0.85, 1.15, 0.001)
        Y = np.arange(0, 0.5, 1/252)
        X, Y = np.meshgrid(X, Y)
        Z = self.vol_surface
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface Plot')

        # Display the plot
        plt.show()

    def total_df_func(self):
        df = pd.read_excel('data/option_chains.xlsx').query(f"stock_code == '{self.stock_code}' ")
        df['time_to_maturity'] = df['biz_days_to_maturity'] / 252
        df['moneyness'] = df['strike'] / df['spot_price']
        df['growth_factor'] = df.apply(lambda row:  exp(sum(self.r[:row.biz_days_to_maturity] - self.q[:row.biz_days_to_maturity]) / 252), axis=1)
        df['forward_moneyness'] = df.apply(lambda row: row.strike / (row.growth_factor * row.spot_price), axis=1)
        return df

    def cubic_splines_of_option_price_with_respect_to_maturity(self, strike: float) -> (CubicSpline, CubicSpline):
        df = self.total_df.query(f"strike == {strike}").copy()
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
                           dx=1e-3 * time_to_maturity, n=1)
        d2c_dk2 = derivative(self.cubic_splines_of_option_price_with_respect_to_strike(expire_date)[i], strike,
                             dx=0.05 * strike, n=2)
        local_vol = 2 * dc_dt / strike ** 2 / d2c_dk2
        return local_vol

    def local_vol_curve_along_moneyness_axis(self, expire_date: datetime):
        df = self.total_df.query(f"expire_date == '{expire_date.strftime('%Y-%m-%d')}'").copy()
        df = df.query("0.8 < moneyness < 1.2")
        df['dupire_local_vol'] = df.apply(lambda row: self.local_vol_calc(strike=row.strike, expire_date=row.expire_date), axis=1)

        def fitting_function(parameters, x):
            sig2_0, delta, kappa, gamma = parameters
            return sig2_0 + delta * np.tanh(kappa * x) / kappa + gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2

        def calc_rss(parameters):
            return sum((df['dupire_local_vol'] - fitting_function(parameters, np.log(df['forward_moneyness']))) ** 2)

        parameters = minimize(calc_rss, np.array([0.01, 0.5, 0.5, 0.5])).x

        def func(forward_moneyness):
            return partial(fitting_function, parameters=parameters)(x=np.log(forward_moneyness))

        return func

    def local_vol_surface(self, moneyness, maturity):
        df = self.total_df[['expire_date', 'time_to_maturity', 'growth_factor']].drop_duplicates()

        x = []
        y = []
        for idx, row in df[1:].iterrows():
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

