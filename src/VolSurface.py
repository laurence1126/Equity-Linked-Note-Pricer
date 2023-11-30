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
import warnings

warnings.filterwarnings("ignore")


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


def dividend_yield_curve(stock_code: Literal["700 HK", "5 HK", "941 HK"]) -> np.array:
    match stock_code:
        case "700 HK":
            S0 = 321.2
            dividend = 2.256
            day = 73
        case "5 HK":
            S0 = 59.45
            dividend = 0.318
            day = 104
        case "941 HK":
            S0 = 63.35
            dividend = 2.53
            day = 151
    discount_rate = yield_curve_interpolate()[day]
    dividend_yield = np.zeros(252)
    dividend_yield[day] = 252 * np.log(1 - dividend * np.exp(-discount_rate * day / 252) / S0)
    return np.array(dividend_yield)


def calc_implied_total_vol(price: float, isCall: bool, F: float, y: float) -> float:
    def func(w):
        vol_partial_bs_formula = partial(bs_formula_pricer, isCall=isCall, F=F, y=y)
        return vol_partial_bs_formula(w=w) - price

    initial_guess = 0.01
    return newton(func, initial_guess, tol=1e-5)


def calc_implied_vol_curve(stock_code: Literal["700 HK", "5 HK", "941 HK"], day: int, log_moneyness: float | np.ndarray):
    # Read option data from excel
    option_chains = pd.read_excel("data/option_chains.xlsx", index_col=False)
    option_chains = option_chains[(option_chains["stock_code"] == stock_code) & (option_chains["biz_days_to_maturity"] == day)]

    # Calculate basic parameters
    S0 = option_chains["spot_price"].iloc[0]
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(stock_code)
    r = fc[:day]
    q = dc[:day]

    #  Store log moneyness-implied vol pairs in data
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

    # Fitting contentious implied vol curve
    def fitting_function(parameters, x):
        sig2_0, delta, kappa, gamma = parameters
        return sig2_0 + delta * np.tanh(kappa * x) / kappa + gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2

    def calc_rss(parameters):
        return sum((data[:, 1] - fitting_function(parameters, data[:, 0])) ** 2)

    parameters = minimize(calc_rss, np.array([0.01, 0.5, 0.5, 0.5])).x
    return fitting_function(parameters, log_moneyness)


def calc_forward_implied_vol_surface(
    stock_code: Literal["700 HK", "5 HK", "941 HK"], log_moneyness: float | np.ndarray, T: float | np.ndarray
) -> float | np.ndarray:
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(stock_code)
    day_list = [23, 46, 67, 87, 153]
    implied_vol_curve_list = []
    for day in day_list:
        r = fc[:day]
        q = dc[:day]
        log_forward_moneyness = log_moneyness - sum(r - q) / 252
        implied_vol_curve_list.append(calc_implied_vol_curve(stock_code, day, log_forward_moneyness))
    cs = CubicSpline(np.array(day_list) / 252, implied_vol_curve_list)
    return cs(T)


def gen_implied_vol_surface(stock_code: Literal["700 HK", "5 HK", "941 HK"]):
    moneyness = np.arange(0.85, 1.15, 1e-3)
    log_moneyness = np.log(moneyness)
    T = np.arange(0, 0.5, 1 / 252)
    return calc_forward_implied_vol_surface(stock_code, log_moneyness, T)


def local_vol_transform(stock_code: Literal["700 HK", "5 HK", "941 HK"], log_forward_moneyness: float, T: float):
    def partial_y(log_forward_moneyness: float):
        return partial(calc_forward_implied_vol_surface, stock_code=stock_code, T=T)(log_forward_moneyness=log_forward_moneyness) * T

    def partial_t(T: float):
        return partial(calc_forward_implied_vol_surface, stock_code=stock_code, log_forward_moneyness=log_forward_moneyness)(T=T) * T

    dw_dy = derivative(partial_y, log_forward_moneyness, dx=1e-2)
    d2w_dy2 = derivative(partial_y, log_forward_moneyness, dx=1e-2, n=2)
    dw_dt = derivative(partial_t, T, dx=1 / 252)
    w = calc_forward_implied_vol_surface(stock_code, log_forward_moneyness, T) * T
    local_vol = dw_dt / (
        1 - log_forward_moneyness / w * dw_dy + 1 / 4 * (-1 / 4 - 1 / w + log_forward_moneyness**2 / w**2) * dw_dy**2 + 1 / 2 * d2w_dy2
    )
    return local_vol


def calc_local_vol_surface(stock_code: Literal["700 HK", "5 HK", "941 HK"], log_moneyness, T) -> float:
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(stock_code)
    r = fc[: int(T * 252)]
    q = dc[: int(T * 252)]
    log_forward_moneyness = log_moneyness - sum(r - q) / 252
    return local_vol_transform(stock_code, log_forward_moneyness, T)


if __name__ == "__main__":
    moneyness = np.arange(0.85, 1.15, 1e-3)
    T = np.arange(0, 0.5, 1 / 252)
    # for t in T:
    #     res = calc_forward_implied_vol_surface("700 HK", 0, T)
    #     trams = local_vol_transform("700 HK", 0, T)
    #     print(res, "->", trams)
    # print(calc_forward_implied_vol_surface("700 HK", log_moneyness, 125 / 252))
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")
    x, y = np.meshgrid(moneyness, T)
    ax.plot_surface(x, y, gen_implied_vol_surface("700 HK"), cmap="viridis", edgecolor="none")
    ax.set_title("Implied Vol Surface")
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Time to Maturity")
    ax.set_zlabel("Implied Volatility")
    plt.show()
    # for t in [23, 46, 67, 87, 153]:
    #     plt.plot(log_moneyness, calc_implied_vol_curve("700 HK", t, log_moneyness))
    # plt.legend(["23", "46", "67", "87", "153"])
    # plt.show()
