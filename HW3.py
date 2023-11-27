import numpy as np
from scipy.stats import norm
import statistics
from typing import Literal

COL, ROW = 126 * 7 * 60, 5000


# generate random number and calculate stock matrix (minutely for 5000 paths)
def gen_stock_matrix(col: int = COL, row: int = ROW):
    S_0, r, sigma, dt = 100, 0.02, 0.3, 1 / (COL * 2)
    rnd = np.random.normal(size=(row, col))
    stock = np.zeros(shape=(row, col))  # simulating stock price
    stock_anti = np.zeros(shape=(row, col))  # simulating stock price using antithetic method
    for i in range(row):
        for j in range(col):
            if j == 0:
                print(f"Calculating stock path {i+1}")
                stock[i, j] = S_0
                stock_anti[i, j] = S_0
            else:
                stock[i, j] = stock[i, j - 1] * np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rnd[i, j - 1])
                stock_anti[i, j] = stock_anti[i, j - 1] * np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * (-rnd[i, j - 1]))
    np.save("data/stock.npy", stock)
    np.save("data/stock_anti.npy", stock_anti)


# Result validation
stock = np.load("data/stock.npy")
stock_anti = np.load("data/stock_anti.npy")
with open("console/validation.txt", "w") as f:
    print("> |$\hat{E}(S_T)$|:", np.mean(stock[:, -1]), file=f)
    print("> |$\hat{E}(S_{T(anti)})$|:", np.mean(stock_anti[:, -1]), file=f)
    print("> |$\hat{E}(S_{T(all)})$|:", (np.mean(stock[:, -1]) + np.mean(stock_anti[:, -1])) / 2, file=f)
    print("> True value:", 100 * np.exp(0.02 * 0.5), file=f)


# Case 1
def case_1(method: Literal["simple", "antithetic", "control"], rebate: int = 0, col: int = COL, row: int = ROW):
    S_0, r, sigma, T, K, barrier = 100, 0.02, 0.3, 0.5, 105, 110
    stock = np.load("data/stock.npy")
    stock_anti = np.load("data/stock_anti.npy")
    match method:
        case "simple":
            payoff = np.zeros(row)
            for i in range(row):
                payoff[i] = max(stock[i, -1] - K, 0) if max(stock[i, int(col / 2) :]) < barrier else rebate
            return np.mean(payoff) * np.exp(-r * T)
        case "antithetic":
            payoff = np.zeros(row)
            for i in range(row):
                payoff_1 = max(stock[i, -1] - K, 0) if max(stock[i, int(col / 2) :]) < barrier else rebate
                payoff_2 = max(stock_anti[i, -1] - K, 0) if max(stock_anti[i, int(col / 2) :]) < barrier else rebate
                payoff[i] = (payoff_1 + payoff_2) / 2
            return np.mean(payoff) * np.exp(-r * T)
        case "control":
            # use vanilla call option as control variate
            d1 = (np.log(S_0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            real_price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            payoff_barrier = np.zeros(row)
            payoff_vanilla = np.zeros(row)
            for i in range(row):
                payoff_barrier[i] = max(stock[i, -1] - K, 0) if max(stock[i, int(col / 2) :]) < barrier else rebate
                payoff_vanilla[i] = max(stock[i, -1] - K, 0)
            # using linear regression to solve $\beta$, which is the slope
            a, _ = statistics.linear_regression(payoff_vanilla, payoff_barrier)
            return (np.mean(payoff_barrier) + a * (real_price - np.mean(payoff_vanilla))) * np.exp(-r * T)


# Case 2
def case_2(method: Literal["simple", "antithetic", "control"], rebate: int = 0, col: int = COL, row: int = ROW):
    S_0, r, sigma, T, K, barrier_1, barrier_2 = 100, 0.02, 0.3, 0.5, 105, 108, 110
    stock = np.load("data/stock.npy")
    stock_anti = np.load("data/stock_anti.npy")
    match method:
        case "simple":
            payoff = np.zeros(row)
            for i in range(row):
                payoff[i] = (
                    max(stock[i, -1] - K, 0) if max(stock[i, : int(col / 2)]) < barrier_1 and max(stock[i, int(col / 2) :]) < barrier_2 else rebate
                )
            return np.mean(payoff) * np.exp(-r * T)
        case "antithetic":
            payoff = np.zeros(row)
            for i in range(row):
                payoff_1 = (
                    max(stock[i, -1] - K, 0) if max(stock[i, : int(col / 2)]) < barrier_1 and max(stock[i, int(col / 2) :]) < barrier_2 else rebate
                )
                payoff_2 = (
                    max(stock_anti[i, -1] - K, 0)
                    if max(stock_anti[i, : int(col / 2)]) < barrier_1 and max(stock_anti[i, int(col / 2) :]) < barrier_2
                    else rebate
                )
                payoff[i] = (payoff_1 + payoff_2) / 2
            return np.mean(payoff) * np.exp(-r * T)
        case "control":
            # use vanilla call option as control variate
            d1 = (np.log(S_0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            real_price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            payoff_barrier = np.zeros(row)
            payoff_vanilla = np.zeros(row)
            for i in range(row):
                payoff_barrier[i] = (
                    max(stock[i, -1] - K, 0) if max(stock[i, : int(col / 2)]) < barrier_1 and max(stock[i, int(col / 2) :]) < barrier_2 else rebate
                )
                payoff_vanilla[i] = max(stock[i, -1] - K, 0)
            # using linear regression to solve $\beta$, which is the slope
            a, _ = statistics.linear_regression(payoff_vanilla, payoff_barrier)
            return (np.mean(payoff_barrier) + a * (real_price - np.mean(payoff_vanilla))) * np.exp(-r * T)


# Case 3
def check_valid(stock_curve: np.ndarray, barrier_curve: np.array):
    if len(stock_curve) != len(barrier_curve):
        raise ValueError("The length of stock curve and barrier curve should be the same.")
    for i in range(len(stock_curve)):
        if stock_curve[i] >= barrier_curve[i]:
            return False
    return True


def case_3(method: Literal["simple", "antithetic", "control"], rebate: int = 0, col: int = COL, row: int = ROW):
    S_0, r, sigma, T, K = 100, 0.02, 0.3, 0.5, 105
    start, end = 105, 115
    barrier_curve = np.linspace(start, end, col, False)
    stock = np.load("data/stock.npy")
    stock_anti = np.load("data/stock_anti.npy")
    match method:
        case "simple":
            payoff = np.zeros(row)
            for i in range(row):
                payoff[i] = max(stock[i, -1] - K, 0) if check_valid(stock[i], barrier_curve) else rebate
            return np.mean(payoff) * np.exp(-r * T)
        case "antithetic":
            payoff = np.zeros(row)
            for i in range(row):
                payoff_1 = max(stock[i, -1] - K, 0) if check_valid(stock[i], barrier_curve) else rebate
                payoff_2 = max(stock_anti[i, -1] - K, 0) if check_valid(stock_anti[i], barrier_curve) else rebate
                payoff[i] = (payoff_1 + payoff_2) / 2
            return np.mean(payoff) * np.exp(-r * T)
        case "control":
            # use vanilla call option as control variate
            d1 = (np.log(S_0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            real_price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            payoff_barrier = np.zeros(row)
            payoff_vanilla = np.zeros(row)
            for i in range(row):
                payoff_barrier[i] = max(stock[i, -1] - K, 0) if check_valid(stock[i], barrier_curve) else rebate
                payoff_vanilla[i] = max(stock[i, -1] - K, 0)
            # using linear regression to solve $\beta$, which is the slope
            a, _ = statistics.linear_regression(payoff_vanilla, payoff_barrier)
            return (np.mean(payoff_barrier) + a * (real_price - np.mean(payoff_vanilla))) * np.exp(-r * T)


# Case 4
def case_4(
    freq: Literal["monthly", "weekly", "daily"],
    method: Literal["simple", "antithetic", "control"],
    rebate: int = 0,
    col: int = COL,
    row: int = ROW,
):
    match freq:
        case "monthly":
            slicer = 21 * 7 * 60
        case "weekly":
            slicer = 5 * 7 * 60
        case "daily":
            slicer = 7 * 60

    S_0, r, sigma, T, K, barrier = 100, 0.02, 0.3, 0.5, 105, 110
    stock = np.load("data/stock.npy")
    stock_anti = np.load("data/stock_anti.npy")
    match method:
        case "simple":
            payoff = np.zeros(row)
            for i in range(row):
                payoff[i] = (
                    max(stock[i, -1] - K, 0)
                    if check_valid(np.append(stock[i][::slicer], stock[i, -1]), [barrier for _ in range(len(stock[i][::slicer]) + 1)])
                    else rebate
                )
            return np.mean(payoff) * np.exp(-r * T)
        case "antithetic":
            payoff = np.zeros(row)
            for i in range(row):
                payoff_1 = (
                    max(stock[i, -1] - K, 0)
                    if check_valid(np.append(stock[i][::slicer], stock[i, -1]), [barrier for _ in range(len(stock[i][::slicer]) + 1)])
                    else rebate
                )
                payoff_2 = (
                    max(stock_anti[i, -1] - K, 0)
                    if check_valid(np.append(stock_anti[i][::slicer], stock_anti[i, -1]), [barrier for _ in range(len(stock_anti[i][::slicer]) + 1)])
                    else rebate
                )
                payoff[i] = (payoff_1 + payoff_2) / 2
            return np.mean(payoff) * np.exp(-r * T)
        case "control":
            # use vanilla call option as control variate
            d1 = (np.log(S_0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            real_price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            payoff_barrier = np.zeros(row)
            payoff_vanilla = np.zeros(row)
            for i in range(row):
                payoff_barrier[i] = (
                    max(stock[i, -1] - K, 0)
                    if check_valid(np.append(stock[i][::slicer], stock[i, -1]), [barrier for _ in range(len(stock[i][::slicer]) + 1)])
                    else rebate
                )
                payoff_vanilla[i] = max(stock[i, -1] - K, 0)
            # using linear regression to solve $\beta$, which is the slope
            a, _ = statistics.linear_regression(payoff_vanilla, payoff_barrier)
            return (np.mean(payoff_barrier) + a * (real_price - np.mean(payoff_vanilla))) * np.exp(-r * T)


print(case_1("simple"))
print(case_1("antithetic"))
print(case_1("control"))
print("*" * 20)
print(case_2("simple"))
print(case_2("antithetic"))
print(case_2("control"))
print("*" * 20)
print(case_3("simple"))
print(case_3("antithetic"))
print(case_3("control"))
print("*" * 20)
print(case_4("monthly", "simple"))
print(case_4("monthly", "antithetic"))
print(case_4("monthly", "control"))
print("*" * 20)
print(case_4("weekly", "simple"))
print(case_4("weekly", "antithetic"))
print(case_4("weekly", "control"))
print("*" * 20)
print(case_4("daily", "simple"))
print(case_4("daily", "antithetic"))
print(case_4("daily", "control"))
print("*" * 20)
