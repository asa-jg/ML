import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## CONSTANTS ##
num_sensors = 10
measurement_noise_std = 0.3
sigma = 0.3
grid_size = 20

## SENSOR SETUP ##
angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)
sensor_positions = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])

def random_point_in_unit_circle():
    """Generate a random point uniformly within a unit circle."""
    while True:
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 <= 1:
            return np.array([x, y])

## EXPLOSION LOCATIONS ##
explosion_1 = random_point_in_unit_circle()
explosion_2 = random_point_in_unit_circle()

## OBSERVATIONS ##
def compute_vi(sensor_pos, s1, s2, noise_std):
    """Compute observed value v_i at a sensor."""
    d1 = np.linalg.norm(sensor_pos - s1)
    d2 = np.linalg.norm(sensor_pos - s2)
    noise = np.random.normal(0, noise_std)
    return 1 / (d1**2 + 0.1) + 1 / (d2**2 + 0.1) + sigma * noise

observed_values = np.array([
    compute_vi(sensor_pos, explosion_1, explosion_2, measurement_noise_std)
    for sensor_pos in sensor_positions
])

## GRID SETUP ##
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)
mask = X**2 + Y**2 <= 1
X_valid = X[mask]
Y_valid = Y[mask]
grid_points = np.vstack((X_valid.ravel(), Y_valid.ravel())).T

## BAYESIAN ANALYSIS ##
posterior = np.zeros(grid_points.shape[0])

for i, s1_candidate in enumerate(grid_points):
    for j, s2_candidate in enumerate(grid_points):
        expected_values = np.array([
            1 / (np.linalg.norm(sensor_pos - s1_candidate)**2 + 0.1) +
            1 / (np.linalg.norm(sensor_pos - s2_candidate)**2 + 0.1)
            for sensor_pos in sensor_positions
        ])
        likelihood = norm.pdf(observed_values, loc=expected_values, scale=measurement_noise_std).prod()
        posterior[i] += likelihood

posterior /= posterior.sum()
posterior_grid = np.zeros_like(X)
posterior_grid[mask] = posterior

## PLOTTING ##
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, posterior_grid, levels=100, cmap='coolwarm')
plt.plot(np.cos(np.linspace(0, 2 * np.pi, 500)), np.sin(np.linspace(0, 2 * np.pi, 500)), 'k', linewidth=2)
plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='blue', marker='^', s=100, label="Sensors")
plt.scatter(explosion_1[0], explosion_1[1], c='green', marker='*', s=200, label="Explosion 1 (True)")
plt.scatter(explosion_2[0], explosion_2[1], c='red', marker='*', s=200, label="Explosion 2 (True)")

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Posterior Distribution of Explosion Locations')
plt.colorbar(label='Posterior Probability Density')
# plt.legend()
plt.grid(True)
plt.show()
