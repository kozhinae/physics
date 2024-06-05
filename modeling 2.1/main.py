import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Установим правильный бэкенд
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

# Константы
c = 3e8  # скорость света в вакууме, м/с
h = 6.626e-34  # постоянная Планка, Дж·с

# Параметры моделирования
N = 5  # количество щелей (1 <= N <= 10)
width = 20e-6  # ширина щели, м
period = 100e-6  # период решетки, м
L = 1.0  # расстояние до экрана, м
λ0 = 500e-9  # средняя длина волны, м
Δλ = 50e-9  # ширина спектра, м

# Размеры экрана
screen_size = 0.01  # размер экрана, м
screen_points = 1000  # количество точек на экране

# Создаем координатную сетку экрана
x = np.linspace(-screen_size / 2, screen_size / 2, screen_points)

# Функция для вычисления интенсивности монохроматического света
def intensity_mono(x, N, width, period, L, λ):
    k = 2 * np.pi / λ  # волновое число
    beta = k * width * x / L
    gamma = k * period * x / L
    I = (np.sin(beta / 2) / (beta / 2))**2 * (np.sin(N * gamma / 2) / np.sin(gamma / 2))**2
    return I

# Функция для вычисления интенсивности квазимонохроматического света
def intensity_quasi(x, N, width, period, L, λ0, Δλ):
    I_total = np.zeros_like(x)
    λ_min = λ0 - Δλ / 2
    λ_max = λ0 + Δλ / 2
    λ_values = np.linspace(λ_min, λ_max, 100)
    for λ in λ_values:
        I_total += intensity_mono(x, N, width, period, L, λ)
    return I_total / len(λ_values)

# Вычисление интенсивности для монохроматического света
I_mono = intensity_mono(x, N, width, period, L, λ0)

# Вычисление интенсивности для квазимонохроматического света
I_quasi = intensity_quasi(x, N, width, period, L, λ0, Δλ)

# Визуализация результатов
plt.figure(figsize=(12, 6))

# График для монохроматического света
plt.subplot(1, 2, 1)
plt.plot(x * 1e3, I_mono, color='blue')
plt.title('Интенсивность для монохроматического света')
plt.xlabel('Координата x, мм')
plt.ylabel('Интенсивность')
plt.grid(True)

# График для квазимонохроматического света
plt.subplot(1, 2, 2)
plt.plot(x * 1e3, I_quasi, color='red')
plt.title('Интенсивность для квазимонохроматического света')
plt.xlabel('Координата x, мм')
plt.ylabel('Интенсивность')
plt.grid(True)

plt.tight_layout()
plt.show()

# Визуализация цветного распределения интенсивности для квазимонохроматического света
def wavelength_to_rgb(wavelength):
    """Конвертация длины волны в диапазоне 380-780 нм в RGB"""
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 < wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 < wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 < wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 < wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 < wavelength <= 645:
        factor = 1.0
    elif 645 < wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
    else:
        factor = 0.0

    R = (R * factor)**gamma if R * factor > 0 else 0
    G = (G * factor)**gamma if G * factor > 0 else 0
    B = (B * factor)**gamma if B * factor > 0 else 0

    return [int(R * intensity_max), int(G * intensity_max), int(B * intensity_max)]

wavelengths = np.linspace(λ0 - Δλ / 2, λ0 + Δλ / 2, 100)
colors = np.array([wavelength_to_rgb(λ * 1e9) for λ in wavelengths]) / 255

# Создание изображения
image = np.zeros((screen_points, len(wavelengths), 3))

for i, λ in enumerate(wavelengths):
    I = intensity_mono(x, N, width, period, L, λ)
    color = colors[i]
    for j in range(screen_points):
        image[j, i] = I[j] * color

plt.figure(figsize=(10, 6))
plt.imshow(image, extent=[(λ0 - Δλ / 2) * 1e9, (λ0 + Δλ / 2) * 1e9, -screen_size / 2 * 1e3, screen_size / 2 * 1e3], aspect='auto')
plt.title('Цветное распределение интенсивности')
plt.xlabel('Длина волны, нм')
plt.ylabel('Координата x, мм')
plt.colorbar(label='Интенсивность')
plt.show()
