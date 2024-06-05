import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Установим правильный бэкенд
import matplotlib.pyplot as plt

# Константы
mu_0 = 4 * np.pi * 1e-7  # магнитная проницаемость вакуума, Н/А^2

# Параметры задачи (можно изменять)
L = 100  # длина провода, м
d = 0.001  # диаметр провода, м
D = 0.1  # диаметр каркаса, м

# Функция для расчета числа витков N
def calc_N(L, D, d, l):
    return int(L / (np.pi * (D + d)))

# Функция для расчета индукции магнитного поля в центре катушки
def calc_B(N, I, l):
    return mu_0 * N * I / l

# Функция для расчета индуктивности катушки
def calc_L(N, D, l):
    return mu_0 * (N**2) * (np.pi * (D/2)**2) / l

# Диапазон значений длины катушки
l_values = np.linspace(0.01, 1, 100)  # длина катушки от 0.01 м до 1 м
B_values = []
L_values = []

# Рассчитать индукцию магнитного поля и индуктивность катушки для каждого значения длины
for l in l_values:
    N = calc_N(L, D, d, l)
    I = 1  # Ток, протекающий через катушку, А (например, 1 А)
    B = calc_B(N, I, l)
    L_ind = calc_L(N, D, l)
    B_values.append(B)
    L_values.append(L_ind)

# Построение графиков
plt.figure(figsize=(12, 6))

# График зависимости индукции магнитного поля от длины катушки
plt.subplot(1, 2, 1)
plt.plot(l_values, B_values, label='B(l)')
plt.title('Зависимость индукции магнитного поля от длины катушки')
plt.xlabel('Длина катушки, м')
plt.ylabel('Индукция магнитного поля, Тл')
plt.grid(True)
plt.legend()

# График зависимости индуктивности катушки от длины катушки
plt.subplot(1, 2, 2)
plt.plot(l_values, L_values, label='L(l)', color='red')
plt.title('Зависимость индуктивности катушки от длины катушки')
plt.xlabel('Длина катушки, м')
plt.ylabel('Индуктивность, Гн')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
