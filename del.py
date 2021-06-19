from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 1, 0.1)
y = x
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.add([xx ** 2 + yy ** 2])

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xx, yy, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()
