import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Установим правильный бэкенд
import matplotlib.pyplot as plt

mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума, Н/А^2

# Параметры соленоида (можно изменять)
D = 0.1  # Диаметр соленоида, м
L = 1.0  # Длина соленоида, м
N = 1000  # Число витков
I = 1.0  # Ток через соленоид, А

# Расчет магнитного поля в точке (x, y, z) от витка
def B_from_loop(x, y, z, loop_radius, current, loop_position):
    dx = x - loop_position[0]
    dy = y - loop_position[1]
    dz = z - loop_position[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r3 = r**3
    if r == 0:
        return np.array([0.0, 0.0, 0.0])
    dBx = mu_0 * current * loop_radius**2 * dz * dy / (2 * r3)
    dBy = mu_0 * current * loop_radius**2 * dz * dx / (2 * r3)
    dBz = mu_0 * current * loop_radius**2 * (loop_radius**2 - dx**2 - dy**2) / (2 * r3)
    return np.array([dBx, dBy, dBz])

# Сетка для визуализации
x = np.linspace(-0.2, 0.2, 100)
y = np.linspace(-0.2, 0.2, 100)
X, Y = np.meshgrid(x, y)
Bz = np.zeros(X.shape)

# Суперпозиция магнитных полей от всех витков
z_positions = np.linspace(-L/2, L/2, N)
for z in z_positions:
    for i in range(len(x)):
        for j in range(len(y)):
            B = B_from_loop(X[i, j], Y[i, j], 0, D/2, I, [0, 0, z])
            Bz[i, j] += B[2]

# Визуализация
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Bz, 50, cmap='inferno')
plt.colorbar(label='Bz (T)')
plt.title('Распределение магнитного поля Bz в плоскости (x, y)')
plt.xlabel('x (м)')
plt.ylabel('y (м)')
plt.grid(True)
plt.show()
