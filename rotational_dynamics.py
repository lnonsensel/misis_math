import numpy as np
import matplotlib.pyplot as plt


def rk4_step(f, beta, tau, h, eps):
    k1 = h * f(tau, beta, eps)
    k2 = h * f(tau + h / 2, beta + k1 / 2, eps)
    k3 = h * f(tau + h / 2, beta + k2 / 2, eps)
    k4 = h * f(tau + h, beta + k3, eps)
    return beta + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def model_db_dtau(tau, beta, eps):
    return eps - np.sin(beta)


def solve_dynamics(eps, tau_max=50, h=0.01, beta0=0):
    steps = int(tau_max / h)
    tau_vals = np.linspace(0, tau_max, steps)
    beta_vals = np.zeros(steps)
    beta_vals[0] = beta0

    for i in range(1, steps):
        beta_vals[i] = rk4_step(
            model_db_dtau, beta_vals[i - 1], tau_vals[i - 1], h, eps
        )

    theta_vals = eps * tau_vals - beta_vals
    return tau_vals, theta_vals, beta_vals


# Визуализация
eps_values = [0.5, 0.9, 1.1, 1.5]
plt.figure(figsize=(12, 6))

for eps in eps_values:
    tau, theta, _ = solve_dynamics(eps)
    plt.plot(tau, theta, label=f"ε = {eps}")

plt.title("Динамика угла поворота частицы θ(τ)")
plt.xlabel("Безразмерное время τ")
plt.ylabel("Угол θ")
plt.legend()
plt.grid(True)
plt.show()
