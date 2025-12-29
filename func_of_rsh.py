import numpy as np
import matplotlib.pyplot as plt

# --- КОНСТАНТЫ И ПАРАМЕТРЫ ---
e = 1.6e-19  # Элементарный заряд (Кл)
kB = 1.38e-23  # Постоянная Больцмана (Дж/К)
T = 298  # Температура (К)
A = 100  # Площадь ячейки (см^2)
J_sc = 0.035  # Ток КЗ (А/см^2)
J_0 = 2.5e-6  # Ток насыщения (А/см^2)
m = 1.5  # Фактор идеальности
Rs = 0.05  # Фиксированное Rs (Ом)
Vt = (kB * T) / e  # Термическое напряжение


# 1. Функция для поиска тока J методом Ньютона-Рафсона
def solve_J(V, Rsh):
    J = J_sc  # Начальное приближение
    for _ in range(50):
        # Вспомогательная экспонента
        arg = (e * (V + J * A * Rs)) / (m * kB * T)
        arg = np.clip(arg, -100, 100)  # Защита от переполнения
        term_exp = np.exp(arg)

        # Функция f(J) = 0
        f = J - J_sc + J_0 * (term_exp - 1) + (V + J * A * Rs) / (Rsh * A)
        # Производная f'(J)
        df = 1 + J_0 * (e * A * Rs / (m * kB * T)) * term_exp + Rs / Rsh

        J_new = J - f / df
        if abs(J_new - J) < 1e-7:
            return J_new
        J = J_new
    return J


# 2. Функция для поиска Voc (когда J=0)
def find_voc(Rsh):
    # При J=0 уравнение: J_sc - J_0(exp(e*V/(m*kB*T))-1) - V/(Rsh*A) = 0
    V = 0.3  # Начальное приближение
    for _ in range(50):
        arg = (e * V) / (m * kB * T)
        term_exp = np.exp(arg)
        f = J_sc - J_0 * (term_exp - 1) - V / (Rsh * A)
        df = -J_0 * (e / (m * kB * T)) * term_exp - 1 / (Rsh * A)
        V_new = V - f / df
        if abs(V_new - V) < 1e-7:
            return V_new
        V = V_new
    return V


# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
V_range = np.linspace(0, 0.8, 100)
plt.figure(figsize=(12, 5))

Rsh_axis = np.logspace(-1, 4, 50)
Voc_vals = [find_voc(r) for r in Rsh_axis]

plt.semilogx(Rsh_axis, Voc_vals, color="red", linewidth=2)
plt.title("Зависимость Voc от шунтирующего сопротивления")
plt.xlabel("Rsh (Ом) - логарифмическая шкала")
plt.ylabel("Voc (Вольт)")
plt.grid(True, which="both", ls="-")

plt.tight_layout()
plt.show()
