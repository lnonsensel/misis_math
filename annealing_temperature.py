import math
import matplotlib.pyplot as plt


alfa = 5.67e-8
Tex = 295.0
PI = math.pi


rho = 1e-7 / 2.5
h = 10.0
eps = 1.0
a = 1e-5


def g(t, Ian):
    jan = Ian / (PI * a * a)
    try:
        term_rad = (eps * alfa * Tex**3 * ((t**4) - 1)) / h
    except OverflowError:
        term_rad = 1e6
    return 1 + (jan**2 * rho * a) / (2 * h * Tex) - term_rad


def g1(t, Ian):
    jan = Ian / (PI * a * a)
    f1 = (
        1
        + (jan**2 * rho * a) / (2 * eps * alfa * Tex**4)
        - (h * (t - 1)) / (eps * alfa * Tex**3)
    )
    if f1 <= 0 or math.isnan(f1):
        return 1.0
    try:
        return f1**0.25
    except OverflowError:
        return 1.0


def find_t(Ian, t0=1.05, acc=1e-4, max_iter=200):
    t = t0
    for _ in range(max_iter):
        t_new = g(t, Ian)
        if not math.isfinite(t_new) or t_new > 100:
            break
        if abs(t_new - t) < acc:
            return t_new
        t = 0.7 * t + 0.3 * t_new

    t = t0
    for _ in range(max_iter):
        t_new = g1(t, Ian)
        if not math.isfinite(t_new) or t_new > 100:
            return None
        if abs(t_new - t) < acc:
            return t_new
        t = 0.7 * t + 0.3 * t_new
    return None


currents = [i / 1000 for i in range(5, 105, 5)]
temps = []

print(" I (мА)   →   T_an (°C)")
print("-" * 28)
for Ian in currents:
    t = find_t(Ian)
    if t:
        Tan = t * Tex - 273
        temps.append(Tan)
        print(f"{Ian * 1000:6.1f}      {Tan:7.2f}")
    else:
        temps.append(None)
        print(f"{Ian * 1000:6.1f}      (не сошлось)")


plt.figure(figsize=(7, 5))
valid_currents = [i * 1000 for i, T in zip(currents, temps) if T is not None]
valid_temps = [T for T in temps if T is not None]
plt.plot(valid_currents, valid_temps, "o-", lw=2, color="red")
# plt.title("Температура отжига провода vs Ток отжига")
plt.xlabel("Ток I_an, мА")
plt.ylabel("Температура T_an, °C")
plt.grid(True)
plt.show()
