import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# КОНСТАНТЫ И ПАРАМЕТРЫ
# ==========================================
e = 1.6e-19  # Элементарный заряд (C)
kB = 1.38e-23  # Постоянная Больцмана (J/K)
T = 298  # Температура (K), 25 C
A = 100  # Площадь ячейки (cm^2)

# Параметры ячейки
J_sc = 0.035  # Ток КЗ (A/cm^2) -> 35 mA/cm^2 (среднее из [25-45])
J_0 = 2.5e-6  # Ток насыщения (A/cm^2) -> 0.0025 mA/cm^2


# Термическое напряжение Vt = kBT / e
Vt = (kB * T) / e

# ==========================================
# ЧАСТЬ 1: ИДЕАЛЬНАЯ ЯЧЕЙКА
# ==========================================
print("--- ЧАСТЬ 1: Идеальная ячейка ---")


def ideal_solar_cell(V, m, J_sc, J_0):
    """Уравнение (1): J = Jsc - J0 * (exp(V / (m*Vt)) - 1)"""
    # Защита от переполнения экспоненты
    arg = V / (m * Vt)
    arg = np.clip(arg, -100, 100)
    return J_sc - J_0 * (np.exp(arg) - 1)


# Диапазон напряжений для графика
V_ideal = np.linspace(0, 0.8, 100)

# 1. Построение графиков для разных m
# m_values = [1.5, 2.0, 2.5]
m_values = np.linspace(1.5, 2.5, 3)
plt.figure(figsize=(12, 5))

# График J-V
plt.subplot(1, 2, 1)
for m in m_values:
    J_vals = ideal_solar_cell(V_ideal, m, J_sc, J_0)
    # Отфильтруем отрицательные токи для красоты графика
    mask = J_vals >= 0
    plt.plot(V_ideal[mask], J_vals[mask] * 1000, label=f"m={m}")  # J в mA/cm^2

plt.title("J-V Характеристики (Идеальная)")
plt.xlabel("Напряжение V (Вольт)")
plt.ylabel("Плотность тока J (мА/см²)")
plt.grid(True)
plt.legend()

# График P-V
plt.subplot(1, 2, 2)
for m in m_values:
    J_vals = ideal_solar_cell(V_ideal, m, J_sc, J_0)
    P_vals = J_vals * V_ideal * 1000  # Мощность в мВт/см^2

    # Находим макс мощность
    p_max = np.max(P_vals)
    v_at_pmax = V_ideal[np.argmax(P_vals)]

    # Расчет Voc (теоретический)
    voc_theory = (m * Vt) * np.log((J_sc / J_0) + 1)

    print(f"m={m}: P_max={p_max:.2f} mW/cm^2, Voc={voc_theory:.3f} V")

    mask = J_vals >= 0
    plt.plot(V_ideal[mask], P_vals[mask], label=f"m={m}")

plt.title("P-V Характеристики (Идеальная)")
plt.xlabel("Напряжение V (Вольт)")
plt.ylabel("Мощность P (мВт/см²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# ЧАСТЬ 2: РЕАЛЬНАЯ ЯЧЕЙКА (Группа МЭН)
# ==========================================
print("\n--- ЧАСТЬ 2: Реальная ячейка (Группа МЭН) ---")
print("Варьируем Rsh, Rs = const")

# Параметры для части 2
m_real = 1.5  # Фиксируем m для сравнения сопротивлений
Rs = 0.05  # Rs = 50 mOhm = 0.05 Ohm (для площади 100cm2)
# Rsh должен быть минимум в 50 раз больше Rs, т.е. > 2.5 Ohm.
Rsh_values = [5, 20, 100, 1000]  # Ом

Rsh_values = [0.05, 0.5, 5, 5000000]


def solve_real_current_newton(V, Rs, Rsh, m, J_sc, J_0, A, tol=1e-6, max_iter=50):
    """
    Решение уравнения (2) методом Ньютона-Рафсона для нахождения J при заданном V.
    J выражается в A/cm^2.
    """
    # Начальное приближение: ток идеальной ячейки
    J = ideal_solar_cell(V, m, J_sc, J_0)

    for _ in range(max_iter):
        # Вспомогательные переменные
        term_exp = np.exp((e * (V + J * A * Rs)) / (m * kB * T))

        # Функция f(J) = 0
        f_val = J - J_sc + J_0 * (term_exp - 1) + (V + J * A * Rs) / (Rsh * A)

        # Производная f'(J)
        df_val = 1 + J_0 * (e * A * Rs / (m * kB * T)) * term_exp + (Rs * A) / (Rsh * A)

        # Шаг Ньютона
        delta = f_val / df_val
        J_new = J - delta

        if abs(J_new - J) < tol:
            return J_new
        J = J_new

    return J  # Возвращаем последнее значение, если не сошлось (или для области пробоя)


# Подготовка графиков
plt.figure(figsize=(12, 5))

# Сравнение с идеальной ячейкой (Rsh -> infinity, Rs -> 0)
V_range = np.linspace(0, 0.8, 100)
J_ideal = ideal_solar_cell(V_range, m_real, J_sc, J_0)
plt.subplot(1, 2, 1)
plt.plot(V_range, J_ideal * 1000, "k--", linewidth=2, label="Ideal (No R)")
plt.subplot(1, 2, 2)
plt.plot(V_range, J_ideal * V_range * 1000, "k--", linewidth=2, label="Ideal (No R)")

# Цикл по значениям Rsh
for Rsh in Rsh_values:
    J_real_curve = []
    for V in V_range:
        j_point = solve_real_current_newton(V, Rs, Rsh, m_real, J_sc, J_0, A)
        J_real_curve.append(j_point)
    J_real_curve = np.array(J_real_curve)
    P_real_curve = J_real_curve * V_range * 1000  # mW/cm^2

    # Фильтрация (J > 0) для отображения первого квадранта
    mask = J_real_curve >= 0
    # График J-V
    plt.subplot(1, 2, 1)
    plt.plot(V_range[mask], J_real_curve[mask] * 1000, label=f"Rsh={Rsh} $\Omega$")

    # График P-V
    plt.subplot(1, 2, 2)
    plt.plot(V_range[mask], P_real_curve[mask], label=f"Rsh={Rsh} $\Omega$")

    # Вывод максимумов
    if np.any(mask):
        p_max = np.max(P_real_curve[mask])
        print(f"Rsh={Rsh} Ohm: P_max={p_max:.2f} mW/cm^2")

# Настройка осей для Части 2
plt.subplot(1, 2, 1)
plt.title(f"J-V Реальная ячейка (Rs={Rs} $\Omega$)")
plt.xlabel("Напряжение (В)")
plt.ylabel("Плотность тока (мА/см²)")
plt.grid(True)
plt.legend()
plt.ylim(0, J_sc * 1000 + 5)

p_limit = 14.70 * 1.1

plt.subplot(1, 2, 2)
plt.title(f"P-V Реальная ячейка (Rs={Rs} $\Omega$)")
plt.xlabel("Напряжение (В)")
plt.ylabel("Мощность (мВт/см²)")
plt.grid(True)

# Устанавливаем лимиты:
plt.xlim(0, 0.8)  # Напряжение от 0 до 0.8 В (охватывает все Voc из задания)
plt.ylim(0, p_limit)

plt.tight_layout()
plt.show()
