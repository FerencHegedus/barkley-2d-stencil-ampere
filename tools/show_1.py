import numpy as np
import matplotlib.pyplot as plt

res = 256

dir = r'C:\Users\Gedei Janos Jonatan\Documents\VII\PROJECT\copy_256'
# s1 = r'\u_sol_6_3_2.txt'
s1 = r'\u_sol_6_6.txt'

# Read data from files
with open(dir + s1, 'r') as file1:
    u_sol = np.array([float(file1.readline().strip()) for _ in range(res**2)])

# Reshape the arrays to 2D
u_sol = u_sol.reshape((res, res))

# Create the heatmap

plt.figure(figsize=(6, 4))
heatmap = plt.imshow(u_sol, cmap='winter', interpolation='nearest')
plt.gca().invert_yaxis()
#plt.colorbar()
plt.axis('on')  # Turn off the grid

plt.xlabel('', fontsize=30, fontweight='bold')
plt.ylabel('', fontsize=30, fontweight='bold')

colorbar = plt.colorbar(heatmap)
# Increase the tick label font size for both axes
plt.tick_params(axis='both', which='major', labelsize=30)
colorbar.ax.tick_params(labelsize=30)

plt.show()

# print("%20f",u_sol.max())

print("{:.200f}".format(u_sol.max()))