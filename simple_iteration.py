import numpy as np
import matplotlib.pyplot as plt

# Параметры ячейки
e = 1.6e-19
kB = 1.38e-23
T = 298
Vt = (kB * T) / e
A = 100
m = 1.5
Jsc = 0.035
J0 = 2.5e-6
Rs = 0.05  # 50 mOhm
Rsh = 100  # Шунт


# МЕТОД ПРОСТЫХ ИТЕРАЦИЙ
def simple_iteration(V, tol=1e-10):
    # Начальное приближение: идеальный ток (Eq. 1)
    J = Jsc - J0 * (np.exp(V / (m * Vt)) - 1)
    iters = 0
    for i in range(1000):
        # Вычисляем правую часть уравнения (2)
        J_next = (
            Jsc
            - J0 * (np.exp((e * (V + J * A * Rs)) / (m * kB * T)) - 1)
            - (V + J * A * Rs) / (Rsh * A)
        )
        iters += 1
        if abs(J_next - J) < tol:
            return J_next, iters
        J = J_next
    return J, iters


# 2. МЕТОД НЬЮТОНА-РАФСОНА (Newton-Raphson)
def newton_raphson(V, tol=1e-10):
    J = Jsc - J0 * (np.exp(V / (m * Vt)) - 1)
    iters = 0
    for i in range(100):
        arg = (e * (V + J * A * Rs)) / (m * kB * T)
        exp_term = np.exp(arg)
        # f(J) = 0
        f = J - Jsc + J0 * (exp_term - 1) + (V + J * A * Rs) / (Rsh * A)
        # Производная f'(J)
        df = 1 + J0 * (e * A * Rs / (m * kB * T)) * exp_term + Rs / Rsh

        J_next = J - f / df
        iters += 1
        if abs(J_next - J) < tol:
            return J_next, iters
        J = J_next
    return J, iters


# Сбор данных для сравнения
V_vals = np.linspace(0, 0.8, 50)
data_newton = [newton_raphson(v) for v in V_vals]
data_simple = [simple_iteration(v) for v in V_vals]

# Построение графика сравнения скорости
# plt.figure(figsize=(10, 5))
plt.plot(V_vals, [d[1] for d in data_simple], "r-o", label="Простые итерации")
plt.plot(V_vals, [d[1] for d in data_newton], "b-s", label="Ньютон-Рафсон")
plt.yscale("log")  # Логарифмическая шкала
plt.title("Скорость сходимости методов (количество итераций)")
plt.xlabel("Напряжение V (Вольт)")
plt.ylabel("Число итераций")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()


J_newton = np.array([d[0] for d in data_newton])
J_simple = np.array([d[0] for d in data_simple])

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---

# График 1: Сравнение J-V кривых
plt.plot(V_vals, J_newton * 1000, "b-", linewidth=2, label="Ньютон-Рафсон")
plt.plot(V_vals, J_simple * 1000, "r--", linewidth=2, label="Простые итерации")
plt.title("Сравнение полученных значений J-V")
plt.xlabel("Напряжение (В)")
plt.ylabel("Плотность тока (мА/см²)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
