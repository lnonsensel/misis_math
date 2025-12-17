import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Данные из задания ---
# Золото (Au) [cite: 33]
wl_au = np.array([0.4, 0.5, 0.652, 0.73, 0.85, 0.918, 1.033])
n_au = np.array([1.65, 0.916, 0.166, 0.164, 0.198, 0.222, 0.272])
k_au = np.array([1.956, 1.840, 3.15, 4.35, 5.63, 6.168, 7.07])

# Серебро (Ag) [cite: 35]
wl_ag = np.array([0.405, 0.5, 0.664, 0.75, 0.85, 0.918, 1.033])
n_ag = np.array([0.173, 0.13, 0.14, 0.146, 0.152, 0.18, 0.22])
k_ag = np.array([1.95, 2.974, 4.15, 4.908, 5.72, 6.183, 6.99])

# Диапазон интерполяции (0.3 - 1.2 микрон) [cite: 28]
wl_fine = np.linspace(0.3, 1.2, 500)


def process_metal(wl_exp, n_exp, k_exp, label):
    # Интерполяция n и k [cite: 27]
    # Используем 'cubic' для гладкости или 'linear' для строгого следования точкам
    f_n = interp1d(wl_exp, n_exp, kind="cubic", fill_value="extrapolate")
    f_k = interp1d(wl_exp, k_exp, kind="cubic", fill_value="extrapolate")

    n_fine = f_n(wl_fine)
    k_fine = f_k(wl_fine)

    # Расчет диэлектрической проницаемости eps = (n + ik)^2
    eps1 = n_fine**2 - k_fine**2
    eps2 = 2 * n_fine * k_fine

    # Расчет поляризуемости alpha (eps_ex = 1)
    eps_m = eps1 + 1j * eps2
    eps_ex = 1.0
    alpha = 3 * eps_ex * (eps_m - eps_ex) / (eps_m + 2 * eps_ex)

    return n_fine, k_fine, eps1, eps2, alpha


# Получение данных
n_au_f, k_au_f, e1_au, e2_au, alpha_au = process_metal(wl_au, n_au, k_au, "Au")
n_ag_f, k_ag_f, e1_ag, e2_ag, alpha_ag = process_metal(wl_ag, n_ag, k_ag, "Ag")

# --- Визуализация ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. График n и k для Золота [cite: 28]
axs[0, 0].plot(wl_fine, n_au_f, "r-", label="n (interp)")
axs[0, 0].plot(wl_fine, k_au_f, "b-", label="k (interp)")
axs[0, 0].scatter(wl_au, n_au, c="red", label="n (exp)")
axs[0, 0].scatter(wl_au, k_au, c="blue", label="k (exp)")
axs[0, 0].set_title("Показатель преломления Au")
axs[0, 0].legend()

# 2. График eps1 и eps2 для Золота и Серебра
axs[0, 1].plot(wl_fine, e1_au, "g-", label="eps1 Au")
axs[0, 1].plot(wl_fine, e1_ag, "g--", label="eps1 Ag")
axs[0, 1].axhline(-2, color="black", linestyle=":", label="Resonance condition")
axs[0, 1].set_title("Диэлектрическая проницаемость (Re)")
axs[0, 1].legend()

# 3. Поляризуемость (Abs(alpha)) для поиска резонанса
axs[1, 0].plot(wl_fine, np.abs(alpha_au), "orange", label="|alpha| Au")
axs[1, 0].plot(wl_fine, np.abs(alpha_ag), "cyan", label="|alpha| Ag")
axs[1, 0].set_title("Абсолютная поляризуемость |α|")
axs[1, 0].legend()

# 4. Мнимая часть поляризуемости (соответствует поглощению)
axs[1, 1].plot(wl_fine, alpha_au.imag, "orange", label="Im(alpha) Au")
axs[1, 1].plot(wl_fine, alpha_ag.imag, "cyan", label="Im(alpha) Ag")
axs[1, 1].set_title("Мнимая часть поляризуемости Im(α)")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Поиск резонансной длины волны (максимум поглощения)
res_au = wl_fine[np.argmax(alpha_au.imag)]
res_ag = wl_fine[np.argmax(alpha_ag.imag)]
print(f"Резонансная длина волны Au: {res_au:.3f} мкм")
print(f"Резонансная длина волны Ag: {res_ag:.3f} мкм")
