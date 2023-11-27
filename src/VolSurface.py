import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import newton, minimize
from scipy.interpolate import CubicSpline
from scipy.misc import derivative
from functools import partial
from matplotlib import pyplot as plt
from typing import Literal


def bs_formula_pricer(isCall: bool, F: float, y: float, w: float) -> float:
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


def yield_curve_interpolate() -> np.array:
    basic_curve = pd.read_excel("data/raw_data.xlsx", sheet_name="HIBOR", header=0, index_col=0)
    basic_curve = basic_curve.iloc[0, :] / 100
    basic_time = [1 / 252, 5 / 252, 15 / 252, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1]
    cs = CubicSpline(basic_time, basic_curve)
    ts = np.arange(0, 1, 1 / 252)
    return np.array(cs(ts))


def forward_rate_curve(yield_curve: np.array) -> np.array:
    dt = 1 / 252
    ts = np.arange(0, 1, 1 / 252)
    forward_rate_curve = np.zeros(len(ts))
    for i in range(len(ts) - 1):
        forward_rate_curve[i] = (yield_curve[i + 1] * ts[i + 1] - yield_curve[i] * ts[i]) / dt
    return np.array(forward_rate_curve)


def dividend_yield_curve(S0: float, day: int, dividend: float) -> np.array:
    discount_rate = yield_curve_interpolate()[day]
    dividend_yield = np.zeros(252)
    dividend_yield[day] = 252 * np.log(1 - dividend * np.exp(-discount_rate * day / 252) / S0)
    return np.array(dividend_yield)


def calc_implied_total_vol(price: float, isCall: bool, F: float, y: float) -> float:
    # f(sig)
    def func(w):
        vol_partial_bs_formula = partial(bs_formula_pricer, isCall=isCall, F=F, y=y)
        return vol_partial_bs_formula(w=w) - price

    return newton(func, 0.01, tol=1e-10)


def gen_implied_vol_curve(stock_code: Literal["700 HK", "5 HK", "941 HK"], day: int):
    option_chains = pd.read_excel("data/option_chains.xlsx", index_col=False)
    option_chains = option_chains[(option_chains["stock_code"] == stock_code) & (option_chains["biz_days_to_maturity"] == day)]

    div = 0  #! Modify
    S0 = option_chains["spot_price"].iloc[0]
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(S0, day, div)
    r = fc[:day]
    q = dc[:day]

    data = []
    for _, row in option_chains.iterrows():
        F = S0 * exp(sum(r - q) / 252)
        y = log(row["strike"] / F)
        if y >= 0:
            isCall = True
            price = row["call_price"]
        else:
            isCall = False
            price = row["put_price"]
        if price != 0:
            w = calc_implied_total_vol(price, isCall, F, y)
            data.append((y, w / (day / 252)))
    data = np.array(data)

    def fitting_function(parameters, x):
        sig2_0, delta, kappa, gamma = parameters
        return sig2_0 + delta * np.tanh(x) / kappa + gamma / 2 * (np.tanh(x) / kappa) ** 2

    def calc_rss(parameters):
        return sum((data[:, 1] - fitting_function(parameters, data[:, 0])) ** 2)

    parameters = minimize(calc_rss, np.array([0.01, 0.5, 0.5, 0.5])).x
    moneyness = np.arange(-0.4, 0.4, 1e-5)
    fitted = fitting_function(parameters, moneyness)
    plt.plot(data[:, 0], data[:, 1], "o", label="Original Data")
    plt.plot(moneyness, fitted, label="Fitted Curve")
    plt.show()


def local_vol_transform():
    pass


if __name__ == "__main__":
    gen_implied_vol_curve("700 HK", 23)
    # dw_dy = (partial_calc_implied_total_vol(y=y + delta_y) - partial_calc_implied_total_vol(y=y - delta_y)) / (2 * delta_y)
    # d2w_dy2 = (
    #     partial_calc_implied_total_vol(y=y + delta_y) - 2 * partial_calc_implied_total_vol(y=y) + partial_calc_implied_total_vol(y=y - delta_y)
    # ) / (delta_y**2)
    # print(dw_dy, d2w_dy2)
    # local_vol = implied_vol / (1 - y / w * dw_dy + 1 / 4 * (-1 / 4 - 1 / w + y**2 / w**2) * dw_dy**2 + 1 / 2 * d2w_dy2)

    # print(local_vol)
