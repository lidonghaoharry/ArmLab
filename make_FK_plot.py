import numpy as np
import matplotlib.pyplot as plt
import kinematics
dh_params = np.array([[0, -np.pi/2, 0.10391, 0],
                      [.20573, np.pi, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])

expected = np.load("kinematics_grid_expected2.npy",allow_pickle=True)
theta = np.load("recorded_pos.npy",allow_pickle=True)

n = len(theta)
err_norm = np.zeros(43)
for i in range(n):
    Ti = kinematics.FK_dh(dh_params, theta[i,0], 5)
    if i == 17: #retaking data for an outlier
        err_norm[i] = np.linalg.norm([-291.5, 359.8] - np.array([expected[i,0], expected[i,1]]))
    else:
        err_norm[i] = np.linalg.norm(Ti[:2,3]*1000 - np.array([expected[i,0], expected[i,1]]))
    print("----------------")
    print("wanted: " + str(np.array([expected[i,0], expected[i,1]])))
    print("got: " + str(Ti[:2,3]*1000))
    print("error: " + str(err_norm[i]))

grid = np.zeros((5,9))


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

print("mean error: ", np.mean(err_norm))

fig, ax = plt.subplots()
ax.set_title("Position Error, Forward Kinematics")
ax.set_xlabel("World Coord X")
ax.set_ylabel("World Coord Y")
ax.set_xticklabels([str(i) for i in range(-500, 500, 100)])
ax.set_yticklabels([str(i) for i in range(-150, 450, 100)])
im = plt.imshow(grid, cmap='hot', interpolation='nearest')
cbar = fig.colorbar(im)
cbar.set_label("Error [mm]", rotation=270)
fig.savefig("FK_error.png")

plt.show()