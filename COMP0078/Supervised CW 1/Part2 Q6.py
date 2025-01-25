import numpy as np
import matplotlib.pyplot as plt

## LATEX INITIALISATION ##
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
np.random.seed(111)
points = np.random.rand(100, 2)
labels = np.random.randint(0, 2, 100)


def h_S_v(point, centres, labels, v=3):
    """
    Computes h_S_v for a given point.
    """
    distances = np.sqrt(np.sum((centres - point) ** 2, axis=1))
    indices = np.argsort(distances)[:v]
    nearest_lbls = labels[indices]
    counts = np.bincount(nearest_lbls, minlength=2)
    if counts[0] > counts[1]:
        return 0
    elif counts[1] > counts[0]:
        return 1
    else:
        return np.random.randint(0, 2)


## PLOTTING ##
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

grid_lbls = np.array([h_S_v(point, points, labels) for point in grid_points])
grid_lbls = grid_lbls.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6), dpi = 200)

contour = ax.contourf(xx, yy, grid_lbls, levels=[-1, 0, 1], colors=['#5371c2', '#57c975', '#ddddff'], alpha=0.6)

ax.scatter(points[labels == 0][:, 0], points[labels == 0][:, 1], color='blue', label=r'\textbf{Label 0}', edgecolor='k')
ax.scatter(points[labels == 1][:, 0], points[labels == 1][:, 1], color='green', label=r'\textbf{Label 1}',
           edgecolor='k')

ax.set_title(r'\textbf{Visualization of Hypothesis} $h_{S,v}$ \textbf{with} $|S|=100$ \textbf{and} $v=3$', fontsize=16)

ax.set_xlabel(r'\textbf{X-axis}', fontsize=14)
ax.set_ylabel(r'\textbf{Y-axis}', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()
