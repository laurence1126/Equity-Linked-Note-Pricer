import numpy as np
from math import exp, sqrt
from Curves import yield_curve_interpolate, forward_rate_curve, dividend_yield_curve
from GatheralLocalVol import gen_implied_vol_surface, gen_local_vol_surface
from DupireLocalVol import DupireLocalVolSurface
import matplotlib.pyplot as plt

# Generate random standard normal matrix
N = 3
M = 126
PATH = 5000
START_DATE = "2023-11-24"
DIVIDEND_DATE_700 = "2024-05-17"
DIVIDEND_DATE_5 = "2024-07-03"
DIVIDEND_DATE_941 = "2023-05-06"
CORR_MATRIX = np.array([[1, 0.404, 0.317], [0.404, 1, 0.376], [0.317, 0.376, 1]])
STOCK_INDEX = {"700 HK": 0, "5 HK": 1, "941 HK": 2}
DIVIDEND = {"700 HK": 2.80, "5 HK": 0.30, "941 HK": 2.53}
INITIAL_PRICE = {"700 HK": 321.2, "5 HK": 59.45, "941 HK": 63.35}
PR = 0


def get_corr_norm(n, m, path, corr_matrix):
    rand_norm = np.random.standard_normal((n, m, path))
    corr_norm = np.zeros((n, m, path))
    for i in range(path):
        # Get Cholesky factor of Correlation matrix
        cho_factor = np.linalg.cholesky(corr_matrix)
        # Get correlated normal matrix
        for j in range(m):
            corr_norm[:, j, i] = np.dot(cho_factor, rand_norm[:, j, i])
    return corr_norm


def stock_price(initial_price, stock_index, corr_norm, path, name, volsurface):
    m = 126
    s_pos = np.zeros((path, m + 1))
    s_neg = np.zeros((path, m + 1))
    r = forward_rate_curve(yield_curve_interpolate())
    q = dividend_yield_curve(name)

    for i in range(path):
        s_pos[i, 0] = initial_price.get(name)
        s_neg[i, 0] = initial_price.get(name)
        s_pos[i, 1] = s_pos[i, 0] * exp(
            (r[0] - q[0] - max(volsurface[0, int((1 - 0.85) * 1000)], 0) / 2) * (1 / 252)
            + sqrt(max(volsurface[0, int((1 - 0.85) * 1000)], 0)) * corr_norm[stock_index.get(name), 0, i] * sqrt(1 / 252)
        )
        s_neg[i, 1] = s_neg[i, 0] * exp(
            (r[0] - q[0] - max(volsurface[0, int(round(1 - 0.85, 3) * 1000)], 0) / 2) * (1 / 252)
            + sqrt(max(volsurface[0, int(round(1 - 0.85, 3) * 1000)], 0) / 2) * (-corr_norm[stock_index.get(name), 0, i]) * sqrt(1 / 252)
        )
        for j in range(1, m):
            s_pos[i, j + 1] = s_pos[i, j] * exp(
                (r[j] - q[j] - max(volsurface[j, int((round(s_pos[i, j - 1] / s_pos[i, j], 3) - 0.85) * 1000)], 0) / 2) * (1 / 252)
                + sqrt(max(volsurface[j, int((round(s_pos[i, j - 1] / s_pos[i, j], 3) - 0.85) * 1000)], 0))
                * corr_norm[stock_index.get(name), j, i]
                * sqrt(1 / 252)
            )
            s_neg[i, j + 1] = s_neg[i, j] * exp(
                (r[j] - q[j] - max(volsurface[j, int((round(s_neg[i, j - 1] / s_neg[i, j], 3) - 0.85) * 1000)], 0) / 2) * (1 / 252)
                + sqrt(max(volsurface[j, int((round(s_neg[i, j - 1] / s_neg[i, j], 3) - 0.85) * 1000)], 0))
                * -corr_norm[stock_index.get(name), j, i]
                * sqrt(1 / 252)
            )
    return np.concatenate((s_pos, s_neg))


# Get single path last price of three stocks
def get_last_price(i, tencent, hsbc, mobile, m):
    return {"700 HK": tencent[i, m - 1], "5 HK": hsbc[i, m - 1], "941 HK": mobile[i, m - 1]}


def find_laggard(last_price, initial_price):
    if last_price.get("700 HK") / initial_price.get("700 HK") < last_price.get("5 HK") / initial_price.get("5 HK"):
        if last_price.get("700 HK") / initial_price.get("700 HK") < last_price.get("941 HK") / initial_price.get("941 HK"):
            return "700 HK"
        else:
            return "941 HK"
    if last_price.get("700 HK") / initial_price.get("700 HK") > last_price.get("5 HK") / initial_price.get("5 HK"):
        if last_price.get("700 HK") / initial_price.get("700 HK") > last_price.get("941 HK") / initial_price.get("941 HK"):
            return "941 HK"
        else:
            return "5 HK"


def decide_scenario(name, last_price, initial_price):
    match name:
        case "700 HK":
            if last_price.get("700 HK") >= initial_price.get("700 HK"):
                return "700 HK", 1
            else:
                return "700 HK", 2
        case "5 HK":
            if last_price.get("5 HK") >= initial_price.get("5 HK"):
                return "5 HK", 1
            else:
                return "5 HK", 2
        case "941 HK":
            if last_price.get("941 HK") >= initial_price.get("941 HK"):
                return "941 HK", 1
            else:
                return "941 HK", 2


def redeem(scenario, pr, last_price, initial_price, name):
    match scenario:
        case 1:
            return 10000 * (1 + pr * (last_price.get(name) / initial_price.get(name) - 1))
        case 2:
            return 10000 * max(0.9, last_price.get(name) / initial_price.get(name))


def get_note_price(path, tencent, hsbc, mobile, initial_price, pr, m):
    summation = 0
    for i in range(int(2 * path)):
        LAST_PRICE = get_last_price(i, tencent, hsbc, mobile, m)
        LAGGARD = find_laggard(LAST_PRICE, initial_price)
        SCENARIO = decide_scenario(LAGGARD, LAST_PRICE, initial_price)
        summation = summation + redeem(SCENARIO[1], pr, LAST_PRICE, initial_price, SCENARIO[0])
    return 1 / (2 * path) * summation * exp(-0.0561137 * 0.5)


def run_pricer(title: str, surfaces: list):
    tencent_surface = surfaces[0]
    hsbc_surface = surfaces[1]
    mobile_surface = surfaces[2]

    prices = []
    for pr in np.arange(2, 2.5, 0.01):
        CORR_NORM = get_corr_norm(N, M, PATH, CORR_MATRIX)
        TENCENT = stock_price(INITIAL_PRICE, STOCK_INDEX, CORR_NORM, PATH, "700 HK", tencent_surface)
        HSBC = stock_price(INITIAL_PRICE, STOCK_INDEX, CORR_NORM, PATH, "5 HK", hsbc_surface)
        MOBILE = stock_price(INITIAL_PRICE, STOCK_INDEX, CORR_NORM, PATH, "941 HK", mobile_surface)
        price = get_note_price(PATH, TENCENT, HSBC, MOBILE, INITIAL_PRICE, pr, M)
        prices.append(price)
        print(pr, price)
        if round(price / 10000, 3) == 0.98:
            plt.plot(pr, price, "ro")
            point = (pr, price)
            pr_tar = pr
    plt.annotate(f"Desired PR is: {round(pr_tar, 2)}", point, (2.25, 9800), arrowprops={"arrowstyle": "->"})
    plt.plot(np.arange(2, 2.5, 0.01), prices)
    plt.xlabel("PR")
    plt.ylabel("Note Price")
    plt.title(f"{title}")
    plt.show()


if __name__ == "__main__":

    tencent_impliedvol_surface = gen_implied_vol_surface(stock_code="700 HK")
    hsbc_impliedvol_surface = gen_implied_vol_surface(stock_code="5 HK")
    mobile_impliedvol_surface = gen_implied_vol_surface(stock_code="941 HK")

    tencent_constvol_surface = np.full(tencent_impliedvol_surface.shape, 0.09)
    hsbc_constvol_surface = np.full(hsbc_impliedvol_surface.shape, 0.09)
    mobile_constvol_surface = np.full(mobile_impliedvol_surface.shape, 0.09)

    tencent_gatheral_localvol_surface = gen_local_vol_surface(stock_code="700 HK")
    hsbc_gatheral_localvol_surface = gen_local_vol_surface(stock_code="5 HK")
    mobile_gatheral_localvol_surface = gen_local_vol_surface(stock_code="941 HK")

    tencent_dupire_localvol_surface = DupireLocalVolSurface(stock_code="700 HK").vol_surface
    hsbc_dupire_localvol_surface = DupireLocalVolSurface(stock_code="5 HK").vol_surface
    mobile_dupire_localvol_surface = DupireLocalVolSurface(stock_code="941 HK").vol_surface

    constant_vol_surfaces = [tencent_constvol_surface, hsbc_constvol_surface, mobile_constvol_surface]
    implied_vol_surfaces = [tencent_impliedvol_surface, hsbc_impliedvol_surface, mobile_impliedvol_surface]
    gatheral_local_vol_surfaces = [tencent_gatheral_localvol_surface, hsbc_gatheral_localvol_surface, mobile_gatheral_localvol_surface]
    dupire_local_vol_surfaces = [tencent_dupire_localvol_surface, hsbc_dupire_localvol_surface, mobile_dupire_localvol_surface]

    run_pricer("Note Price vs PR", constant_vol_surfaces)
    run_pricer("Note Price vs PR", implied_vol_surfaces)
    run_pricer("Note Price vs PR", dupire_local_vol_surfaces)
