import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Параметры моделирования
num_particles = 200
num_steps = 1000
box_size = 7
particle_radius = 0.2

# Случайное распределение начальных позиций молекул газа в пределах контейнера
particles_position = (
    np.random.rand(num_particles, 2) * (box_size - 2 * particle_radius)
    + particle_radius
)
particles_velocity = (
    np.random.randn(num_particles, 2) * 0.5
)  # Нормальное распределение скоростей


# Функция для расчета центра инерции системы молекул
def calculate_center_of_mass(positions):
    return np.mean(positions, axis=0)


# Функция для обработки столкновений молекул со стенами
def handle_wall_collisions(positions, velocities, box_size):
    for i in range(len(positions)):
        for j in range(2):  # Обрабатываем x и y отдельно
            if (
                positions[i, j] < particle_radius
                or positions[i, j] > box_size - particle_radius
            ):
                velocities[i, j] = -velocities[
                    i, j
                ]  # Простое отражение скорости при ударе о стену


# Создание графика и анимации
fig, ax = plt.subplots()
trajectory = []  # Список для хранения позиций центра масс

for step in range(num_steps):
    particles_position += particles_velocity
    handle_wall_collisions(particles_position, particles_velocity, box_size)
    center_of_mass = calculate_center_of_mass(
        particles_position
    )  # Расчет центра инерции
    trajectory.append(center_of_mass)  # Добавление позиции в траекторию

    ax.clear()

    ax.plot(
        *zip(*trajectory), marker="o", markersize=3, color="green"
    )  # Визуализация траектории центра масс с меньшими кругами

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    plt.pause(0.01)

plt.show()
