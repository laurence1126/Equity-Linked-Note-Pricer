import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from math import log, sqrt, exp, pi
from scipy.stats import norm
from functools import partial
from scipy.misc import derivative


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", 1000)


def bs_formula_pricer(isCall, F, y, w):
    # calc d1 and d2
    d1 = -y / sqrt(w) + sqrt(w) / 2
    d2 = d1 - sqrt(w)

    # calc option price
    callPrice = F * (norm.cdf(d1) - exp(y) * norm.cdf(d2))
    putPrice = F * (exp(y) * norm.cdf(-d2) - norm.cdf(-d1))

    # return option price
    if isCall:
        return callPrice
    else:
        return putPrice


def calc_implied_total_vol(price, isCall, F, y):
    # f(sig)
    def func(w):
        vol_partial_bs_formula = partial(bs_formula_pricer, isCall=isCall, F=F, y=y)
        return vol_partial_bs_formula(w=w) - price

    return newton(func, 0.005, tol=1e-15)


def yield_curve_interpolate():
    basic_curve = pd.read_excel("raw_data.xlsx", sheet_name="HIBOR", header=0, index_col=0)
    basic_curve = basic_curve.iloc[0, :] / 100
    basic_time = [1 / 252, 5 / 252, 15 / 252, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1]
    cs = CubicSpline(basic_time, basic_curve)
    ts = np.arange(0, 1, 1 / 252)
    return np.array(cs(ts))


def forward_rate_curve(yield_curve):
    dt = 1 / 252
    ts = np.arange(0, 1, 1 / 252)
    forward_rate_curve = np.zeros(len(ts))
    for i in range(len(ts) - 1):
        forward_rate_curve[i] = (yield_curve[i + 1] * ts[i + 1] - yield_curve[i] * ts[i]) / dt
    return np.array(forward_rate_curve)


def dividend_yield_curve(s0, day, dividend):
    discount_rate = yield_curve_interpolate()[day]
    dividend_yield = np.zeros(252)
    dividend_yield[day] = 252 * np.log(1 - dividend * np.exp(-discount_rate * day / 252) / s0)
    return np.array(dividend_yield)


def local_col_transform():
    pass


if __name__ == "__main__":
    s0 = 321.2
    K = 275
    T = 46 / 252
    price = 50.18
    d = 2.8
    sig = 0.3061

    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(s0, int(T * 252), d)

    r = fc[: int(T * 252)]
    q = dc[: int(T * 252)]
    F = s0 * exp(sum(r - q) / 252)
    y = log(K / F)
    w = calc_implied_total_vol(price, True, F, y)
    implied_vol = w / T
    print(w, y)

    def partial_calc_implied_total_vol(y):
        func = partial(calc_implied_total_vol, price=price, isCall=True, F=F)
        return func(y=y)


    delta_y = 1e-20*y
    dw_dy = (partial_calc_implied_total_vol(y=y+delta_y) - partial_calc_implied_total_vol(y=y-delta_y))/(2*delta_y)
    d2w_dy2 = (partial_calc_implied_total_vol(y=y+delta_y) - 2*partial_calc_implied_total_vol(y=y) + partial_calc_implied_total_vol(y=y-delta_y))/(delta_y**2)
    print(dw_dy, d2w_dy2)
    local_vol = implied_vol / (1 - y/w * dw_dy + 1/4 * (-1/4 - 1/w + y**2/w**2) * dw_dy**2 + 1/2 * d2w_dy2)

    print(local_vol)
