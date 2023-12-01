## Local Volatility (Gatheral's Method)

### Transformation From Black-Scholes Implied Volatility

For each point on the Black-Scholes implied volatility surface $w(y,T)=\sigma^2_{BS}T$, we can find the corresponding local volatility $v_L$ by the following transformation equation via:

$$v_L = \frac{\frac{\partial w}{\partial T}}{1-\frac{y}{w}\frac{\partial w}{\partial y}+\frac{1}{4}(-\frac{1}{4}-\frac{1}{w}+\frac{y^2}{w^2})(\frac{\partial w}{\partial y})^2+\frac{1}{2}\frac{\partial^2 w}{\partial y^2}}$$

Here we also transform the input moneyness $m$ to forward log-moneyness $y$ to fit the equation above. The partial derivatives $\frac{\partial w}{\partial y}$, $\frac{\partial^2 w}{\partial y^2}$, and $\frac{\partial w}{\partial T}$ are calculated via numerical differentiation method through Python packages.

Due to the fact that we fitted the Black-Scholes implied volatility surface with z-axis as $\sigma^2_{BS}$ before, we need to multiply the matrix of surface $\sigma^2_{BS}(y, T)_{(300 \times 126)}$ by $diag(T)_{(126\ \times 126)}$ via:

$$
w(y, T) =
    \begin{bmatrix}
        T_{1} & & \\
        & \ddots & \\
        & & T_{126}
    \end{bmatrix}
    \cdot \sigma^2_{BS}(y, T)
$$

With Python Code implementation:

```python
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
```

Plots for combining the local volatility surface with the Black-Scholes implied volatility surface:
![](img/local_vs_implied_combined.png)
