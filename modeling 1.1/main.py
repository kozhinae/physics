import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Установим правильный бэкенд
import matplotlib.pyplot as plt

# Константы
e = 1.602e-19  # заряд электрона, Кл
m = 9.109e-31  # масса электрона, кг
mu_0 = 4 * np.pi * 1e-7  # магнитная проницаемость вакуума, Н/А^2

# Параметры задачи
D = 0.1  # диаметр соленоида, м
n = 1000  # витков на метр
Ra = 0.05  # радиус анода, м
Rk = 0.02  # радиус катода, м
U1 = 10  # начальное напряжение, В
U2 = 100  # конечное напряжение, В

# Функция для расчета магнитного поля внутри соленоида
def calc_B(Ic):
    return mu_0 * n * Ic

# Функция для расчета силы Лоренца и траектории электрона
def electron_trajectory(U, Ic):
    B = calc_B(Ic)
    v = np.sqrt(2 * e * U / m)  # скорость электрона
    omega = e * B / m  # циклотронная частота
    T = 2 * np.pi / omega  # период обращения
    t = np.linspace(0, T, 1000)
    x = (Ra - Rk) / 2 * np.cos(omega * t)
    y = (Ra - Rk) / 2 * np.sin(omega * t)
    return x, y

# Построение траектории электрона при заданных U и Ic
U = 50  # заданное напряжение, В
Ic = 0.1  # заданный ток, А
x, y = electron_trajectory(U, Ic)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Траектория движения электрона')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.axis('equal')
plt.grid(True)
plt.show()

# Построение диаграммы зависимости Ic от U
U_values = np.linspace(U1, U2, 100)
Ic_values = []

for U in U_values:
    Ic_min = 0
    Ic_max = 10
    while Ic_max - Ic_min > 0.01:
        Ic_mid = (Ic_min + Ic_max) / 2
        B_mid = calc_B(Ic_mid)
        omega_mid = e * B_mid / m
        T_mid = 2 * np.pi / omega_mid
        R_mid = (Ra - Rk) / 2
        if T_mid < 2 * np.pi * R_mid / np.sqrt(2 * e * U / m):
            Ic_min = Ic_mid
        else:
            Ic_max = Ic_mid
    Ic_values.append(Ic_mid)

# Построение диаграммы зависимости Ic от U
plt.figure(figsize=(10, 5))
plt.plot(U_values, Ic_values, label='Область, где электрон описывает окружность диаметром (Ra-Rk)')
plt.fill_between(U_values, Ic_values, alpha=0.3)
plt.title('Диаграмма зависимости Ic от U')
plt.xlabel('U, В')
plt.ylabel('Ic, А')
plt.legend()
plt.grid(True)
plt.show()
