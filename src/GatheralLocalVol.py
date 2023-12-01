import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton, minimize
from scipy.interpolate import CubicSpline
from scipy.misc import derivative
from functools import partial
from matplotlib import pyplot as plt
from typing import Literal
import warnings
from Curves import dividend_yield_curve, forward_rate_curve, yield_curve_interpolate

warnings.filterwarnings("ignore")


def bs_formula_pricer(isCall: bool, F: float, y: float, w: float) -> float:
    # calc d1 and d2
    d1 = -y / np.sqrt(w) + np.sqrt(w) / 2
    d2 = d1 - np.sqrt(w)
    # calc option price
    callPrice = F * (norm.cdf(d1) - np.exp(y) * norm.cdf(d2))
    putPrice = F * (np.exp(y) * norm.cdf(-d2) - norm.cdf(-d1))
    # return option price
    if isCall:
        return callPrice
    else:
        return putPrice


def calc_implied_total_vol(price: float, isCall: bool, F: float, y: float) -> float:
    def func(w):
        vol_partial_bs_formula = partial(bs_formula_pricer, isCall=isCall, F=F, y=y)
        return vol_partial_bs_formula(w=w) - price

    initial_guess = 0.01
    return newton(func, initial_guess, tol=1e-5)


def calc_implied_vol_curve(stock_code: Literal["700 HK", "5 HK", "941 HK"], day: int, log_forward_moneyness: float | np.ndarray):
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
        F = S0 * np.exp(sum(r - q) / 252)
        y = np.log(row["strike"] / F)
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
    return fitting_function(parameters, log_forward_moneyness)


def calc_forward_implied_vol_surface(
    stock_code: Literal["700 HK", "5 HK", "941 HK"], moneyness: float | np.ndarray, T: float | np.ndarray, log: bool = False
) -> float | np.ndarray:
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(stock_code)
    day_list = [23, 46, 67, 87, 153]
    implied_vol_curve_list = []
    for day in day_list:
        r = fc[:day]
        q = dc[:day]
        if not log:
            log_forward_moneyness = np.log(moneyness) - sum(r - q) / 252
        else:
            log_forward_moneyness = moneyness - sum(r - q) / 252
        implied_vol_curve_list.append(calc_implied_vol_curve(stock_code, day, log_forward_moneyness))
    cs = CubicSpline(np.array(day_list) / 252, implied_vol_curve_list)
    return cs(T)


def gen_implied_vol_surface(stock_code: Literal["700 HK", "5 HK", "941 HK"]):
    moneyness = np.arange(0.85, 1.15, 1e-3)
    T = np.arange(0, 0.5, 1 / 252)
    return calc_forward_implied_vol_surface(stock_code, moneyness, T)


def local_vol_transform(stock_code: Literal["700 HK", "5 HK", "941 HK"], moneyness: float | np.ndarray, T: float | np.ndarray):
    yc = yield_curve_interpolate()
    fc = forward_rate_curve(yc)
    dc = dividend_yield_curve(stock_code)
    log_moneyness = np.log(moneyness)

    def partial_y(log_moneyness: float):
        return np.dot(np.diag(T), partial(calc_forward_implied_vol_surface, stock_code=stock_code, T=T, log=True)(moneyness=log_moneyness))

    def partial_t(T: float):
        return np.dot(np.diag(T), partial(calc_forward_implied_vol_surface, stock_code=stock_code, moneyness=log_moneyness, log=True)(T=T))

    dw_dy = derivative(partial_y, log_moneyness, dx=1e-5)
    d2w_dy2 = derivative(partial_y, log_moneyness, dx=1e-5, n=2)
    dw_dt = derivative(partial_t, T, dx=1 / 252)
    w = np.dot(np.diag(T), calc_forward_implied_vol_surface(stock_code, log_moneyness, T, log=True))
    log_forward_moneyness = np.array([np.log(moneyness) - sum(fc[: int(t * 252)] - dc[: int(t * 252)]) / 252 for t in T])
    local_vol = dw_dt / (
        1 - log_forward_moneyness / w * dw_dy + 1 / 4 * (-1 / 4 - 1 / w + log_forward_moneyness**2 / w**2) * dw_dy**2 + 1 / 2 * d2w_dy2
    )
    return local_vol


def gen_local_vol_surface(stock_code: Literal["700 HK", "5 HK", "941 HK"]):
    moneyness = np.arange(0.85, 1.15, 1e-3)
    T = np.arange(0, 0.5, 1 / 252)
    return local_vol_transform(stock_code, moneyness, T)


if __name__ == "__main__":
    moneyness = np.arange(0.85, 1.15, 1e-3)
    T = np.arange(0, 0.5, 1 / 252)
    fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection="3d"), figsize=(15, 5))
    for i, stock_code in enumerate(["700 HK", "5 HK", "941 HK"]):
        x, y = np.meshgrid(moneyness, T)
        axs[i].plot_surface(x, y, gen_local_vol_surface(stock_code), cmap="plasma", edgecolor="none")
        axs[i].plot_surface(x, y, gen_implied_vol_surface(stock_code), cmap="viridis", edgecolor="none")
        axs[i].set_title(stock_code)
        axs[i].set_xlabel("Moneyness")
        axs[i].set_ylabel("Time to Maturity")
        axs[i].set_zlabel("Volatility")
    fig.suptitle(f"Local v.s. Implied Vol Surface")
    fig.legend(["Local Vol", "Implied Vol"])
    plt.subplots_adjust(left=0, right=1)
    plt.show()
    plt.savefig("img/local_vs_implied_combined.png")
