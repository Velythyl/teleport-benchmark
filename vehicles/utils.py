def rk_four(f, x, u, T):
    """Fourth-order Runge-Kutta numerical integration."""
    k_1 = f(x, u)
    k_2 = f(x + T * k_1 / 2.0, u)
    k_3 = f(x + T * k_2 / 2.0, u)
    k_4 = f(x + T * k_3, u)
    x_new = x + T / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
    return x_new
