import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Параметры моделирования
num_particles = 100  # Количество молекул газа
num_steps = 1000  # Количество шагов моделирования
box_size = 7  # Размер контейнера для газа
particle_radius = 0.2  # Радиус молекулы газа
solid_object_position = np.array([5.0, 5.0])  # Позиция твердого объекта
solid_object_radius = 0.5  # Радиус твердого объекта
solid_object_angular_velocity = 0.0  # Угловая скорость твердого объекта

# Момент инерции твердого объекта (для простоты считаем его как момент инерции круга)
solid_object_moment_of_inertia = 0.5 * solid_object_radius**2

# Случайное распределение начальных позиций молекул газа в пределах контейнера
particles_position = (
    np.random.rand(num_particles, 2) * (box_size - 2 * particle_radius)
    + particle_radius
)
particles_velocity = np.random.randn(num_particles, 2) * 0.5  # Увеличение скорости


# Функция для обработки столкновений молекул газа между собой с использованием метода ближайших соседей
def handle_collisions_between_particles(positions, velocities):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=2 * particle_radius)
    for i, j in pairs:
        dist = np.linalg.norm(positions[i] - positions[j])
        normal = (positions[j] - positions[i]) / dist
        tangent = np.array([-normal[1], normal[0]])
        v1n = np.dot(normal, velocities[i])
        v1t = np.dot(tangent, velocities[i])
        v2n = np.dot(normal, velocities[j])
        v2t = np.dot(tangent, velocities[j])
        v1n_new = (v1n * (particle_radius - dist) + 2 * particle_radius * v2n) / (
            particle_radius + dist
        )
        v2n_new = (v2n * (particle_radius - dist) + 2 * particle_radius * v1n) / (
            particle_radius + dist
        )
        velocities[i] = v1n_new * normal + v1t * tangent
        velocities[j] = v2n_new * normal + v2t * tangent


# Функция для обработки столкновений молекул с твердым объектом с учетом момента
def handle_collisions_with_object(
    positions,
    velocities,
    object_position,
    object_radius,
    object_angular_velocity,
    moment_of_inertia,
):
    attraction_factor = 2  # Увеличение коэффициента притяжения
    for i in range(len(positions)):
        direction = positions[i] - object_position
        distance = np.linalg.norm(direction)
        if distance < (particle_radius + object_radius):
            # Рассчитываем скорость молекулы относительно твердого объекта
            relative_velocity = velocities[i] - object_angular_velocity * np.array(
                [-direction[1], direction[0]]
            )
            # Рассчитываем момент импульса молекулы относительно твердого объекта
            angular_momentum = (
                np.cross(direction, np.append(relative_velocity, 0)) * moment_of_inertia
            )
            # Изменяем скорость молекулы
            # Меняем знак "-" на "+" для притяжения, увеличиваем эффект
            velocities[i] += (
                2
                * angular_momentum[:-1]
                * direction
                / (moment_of_inertia * distance**2)
            )


# # Функция для обработки столкновений молекул с твердым объектом без учета момента
# def handle_collisions_with_object(
#     positions, velocities, object_position, object_radius
# ):
#     for i in range(len(positions)):
#         direction = positions[i] - object_position
#         distance = np.linalg.norm(direction)
#         if distance < (particle_radius + object_radius):
#             # Изменяем скорость молекулы без учета момента
#             velocities[i] += (
#                 2 * np.dot(-direction, velocities[i]) * direction / distance**2
#             )


# Функция для обработки столкновений молекул со стенами
def handle_wall_collisions(positions, velocities, box_size):
    for i in range(len(positions)):
        for j in range(2):  # Обрабатываем x и y отдельно
            if positions[i, j] < particle_radius:
                velocities[i, j] = abs(velocities[i, j])
                positions[i, j] = particle_radius
            elif positions[i, j] > box_size - particle_radius:
                velocities[i, j] = -abs(velocities[i, j])
                positions[i, j] = box_size - particle_radius


# Создаем график
fig, ax = plt.subplots()

# Цикл по шагам моделирования
for step in range(num_steps):
    # Моделируем движение молекул газа
    particles_position += particles_velocity

    # Обрабатываем столкновения молекул газа между собой
    handle_collisions_between_particles(particles_position, particles_velocity)

    # Обрабатываем столкновения молекул с твердым объектом с учетом момента
    handle_collisions_with_object(
        particles_position,
        particles_velocity,
        solid_object_position,
        solid_object_radius,
        solid_object_angular_velocity,
        solid_object_moment_of_inertia,
    )

    # # Обрабатываем столкновения молекул с твердым объектом без учета момента
    # handle_collisions_with_object(
    #     particles_position,
    #     particles_velocity,
    #     solid_object_position,
    #     solid_object_radius,
    # )
    # Обрабатываем столкновения молекул со стенами
    handle_wall_collisions(particles_position, particles_velocity, box_size)

    # Обновляем позицию твердого объекта на основе его угловой скорости
    solid_object_position += np.array(
        [-solid_object_angular_velocity, solid_object_angular_velocity]
    )

    # Отображаем молекулы газа
    ax.clear()
    ax.scatter(
        particles_position[:, 0],
        particles_position[:, 1],
        s=10,
        color="blue",
        alpha=0.5,
    )

    # Отображаем твердый объект
    ax.add_patch(plt.Circle(solid_object_position, solid_object_radius, color="red"))

    # Устанавливаем ограничения на оси для представления контейнера
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    plt.pause(0.01)  # Быстрая пауза для плавной анимации

plt.show()
