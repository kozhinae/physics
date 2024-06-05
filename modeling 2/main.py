import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Установим правильный бэкенд
import matplotlib.pyplot as plt

# Константы
λ = 500e-9  # длина волны, м
L = 1.0  # расстояние до экрана, м

# Размеры области моделирования
size = 1e-3  # размер области моделирования в метрах
N = 500  # количество точек в каждой координате

# Создаем координатные сетки
x = np.linspace(-size / 2, size / 2, N)
y = np.linspace(-size / 2, size / 2, N)
X, Y = np.meshgrid(x, y)

# Определяем амплитудное распределение объекта
# Пример: круглый апертур с радиусом 0.1 мм
R = 0.1e-3  # радиус апертуры, м
aperture = np.zeros((N, N))
aperture[X**2 + Y**2 <= R**2] = 1

# Выполняем двумерное преобразование Фурье
U = np.fft.fftshift(np.fft.fft2(aperture))

# Вычисляем координаты в плоскости наблюдения
fx = np.fft.fftfreq(N, d=x[1] - x[0])
fy = np.fft.fftfreq(N, d=y[1] - y[0])
FX, FY = np.meshgrid(fx, fy)

# Вычисляем распределение интенсивности в плоскости наблюдения
I = (np.abs(U)**2) * (λ * L / (size * N))**2

# Построение распределения интенсивности
plt.figure(figsize=(10, 5))
plt.imshow(I, extent=(fx.min(), fx.max(), fy.min(), fy.max()), cmap='inferno')
plt.title('Распределение интенсивности дифракции Фраунгофера')
plt.xlabel('fx (1/м)')
plt.ylabel('fy (1/м)')
plt.colorbar(label='Интенсивность')
plt.show()
