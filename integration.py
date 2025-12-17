import numpy as np


def f1(x):
    return np.sin(x)


def f2(x):
    return np.exp(-(x**2))


def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))


def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1  # n должно быть четным
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


# Исследование точности для sin(x) от 0 до pi
a, b = 0, np.pi
exact_val = 2.0
intervals = [10, 20, 40, 80]

print(
    f"{'N':>5} | {'Err Trap':>12} | {'Ratio T':>8} | {'Err Simp':>12} | {'Ratio S':>8}"
)
print("-" * 60)

prev_err_t = None
prev_err_s = None

for n in intervals:
    res_t = trapezoidal_rule(f1, a, b, n)
    res_s = simpson_rule(f1, a, b, n)

    err_t = abs(exact_val - res_t)
    err_s = abs(exact_val - res_s)

    ratio_t = prev_err_t / err_t if prev_err_t else 0
    ratio_s = prev_err_s / err_s if prev_err_s else 0

    print(f"{n:5d} | {err_t:12.8e} | {ratio_t:8.2f} | {err_s:12.8e} | {ratio_s:8.2f}")

    prev_err_t = err_t
    prev_err_s = err_s
