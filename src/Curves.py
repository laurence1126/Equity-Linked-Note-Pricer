import numpy as np
from scipy.interpolate import CubicSpline
from typing import Literal


def yield_curve_interpolate() -> np.array:
    basic_curve = [i / 100 for i in [5.07048, 5.28202, 5.28649, 5.37941, 5.58012, 5.64167, 5.61137, 5.58054]]
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
    dividend_yield[day] = -252 * np.log(1 - dividend * np.exp(-discount_rate * day / 252) / S0)
    return np.array(dividend_yield)
