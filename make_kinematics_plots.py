import numpy as np
import matplotlib.pyplot as plt

expected = np.load("kinematics_grid_expected2.npy")
actual = np.load("kinematics_grid_actual2.npy")*1000
expected_nog = np.load("kinematics_grid_expected_no_g2.npy")
actual_nog = np.load("kinematics_grid_actual_no_g2.npy")*1000

error = actual - expected
err_norm = np.linalg.norm(error, axis=1)

err_nog = actual_nog - expected_nog
err_norm_nog = np.linalg.norm(err_nog, axis=1)

grid = np.zeros((5,9))
grid_nog = np.zeros((5,9))


x = np.linspace(-400, -100, 4)
y = np.linspace(-50, 350, 5)
lgridx, lgridy = np.meshgrid(x,y)
lpairs = np.vstack([lgridx.ravel(), lgridy.ravel()])

x = 0
y = np.linspace(150, 350, 3)
cgridy, cgridx = np.meshgrid(y, x)
cpairs = np.vstack([cgridx.ravel(), cgridy.ravel()])

x = np.linspace(400, 100, 4)
y = np.linspace(-50, 350, 5)
rgridy, rgridx = np.meshgrid(y, x)
rpairs = np.vstack([rgridx.ravel(), rgridy.ravel()])
points = np.hstack((lpairs, cpairs, rpairs))

n = points.shape[1]

for i in range(n):
    col = int((points[0,i] + 400)/100)
    row = int((points[1,i] + 50)/100)
    grid[row, col] = err_norm[i]
    grid_nog[row,col] = err_norm_nog[i]

print("mean compensated: ", np.mean(err_norm))
print("mean uncompensated: ", np.mean(err_norm_nog))

fig, ax = plt.subplots()
ax.set_title("Position Error, Gravity Compensation")
ax.set_xlabel("World Coord X")
ax.set_ylabel("World Coord Y")
ax.set_xticklabels([str(i) for i in range(-500, 500, 100)])
ax.set_yticklabels([str(i) for i in range(-150, 450, 100)])
im = plt.imshow(grid, cmap='hot', interpolation='nearest')
cbar = fig.colorbar(im)
cbar.set_label("Error [mm]", rotation=270)
fig.savefig("pos_eror_compensated.png")

fig2, ax2 = plt.subplots()
ax2.set_title("Position Error, No Gravity Compensation")
ax2.set_xlabel("World Coord X")
ax2.set_ylabel("World Coord Y")
ax2.set_xticklabels([str(i) for i in range(-500, 500, 100)])
ax2.set_yticklabels([str(i) for i in range(-150, 450, 100)])
im2 = plt.imshow(grid_nog, cmap='hot', interpolation='nearest')
cbar2 = fig2.colorbar(im2)
cbar2.set_label("Error [mm]", rotation=270)
fig2.savefig("pos_eror_uncompensated.png")

plt.show()
print(error)