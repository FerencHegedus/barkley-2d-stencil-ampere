import numpy as np
import matplotlib.pyplot as plt

res = 256

dir = r'C:\Users\Gedei Janos Jonatan\Documents\VII\PROJECT\copy_256'
s1 = r'\u_sol_3_4_m.txt'
s2 = r'\u_comp_5000_d.txt'


# Read data from files
with open(dir + s1, 'r') as file1, open(dir + s2, 'r') as file2:
    u_sol1 = np.array([float(file1.readline().strip()) for _ in range(res**2)])
    u_sol2 = np.array([float(file2.readline().strip()) for _ in range(res**2)])

# Reshape the arrays to 2D
u_sol1 = u_sol1.reshape((res, res))
u_sol2 = u_sol2.reshape((res, res))

# Calculate the difference
u_diff = u_sol1 - u_sol2

# Create the heatmap
plt.figure(figsize=(12, 8))
heatmap = plt.imshow(u_diff, cmap='winter', interpolation='nearest')
plt.gca().invert_yaxis()
#plt.colorbar()
plt.axis('on')  # Turn off the grid

print("max(u_diff) = {:.200f}".format(u_diff.max()))

plt.xlabel('', fontsize=30, fontweight='bold')
plt.ylabel('', fontsize=30, fontweight='bold')

colorbar = plt.colorbar(heatmap)
# Increase the tick label font size for both axes
plt.tick_params(axis='both', which='major', labelsize=30)
colorbar.ax.tick_params(labelsize=30)

plt.show()

i=0
j=0
print(u_sol1[j][i])
print(u_sol2[j][i])
print(u_sol1[j][i] - u_sol2[j][i])

i=1
j=0
print(u_sol1[j][i])
print(u_sol2[j][i])
print(u_sol1[j][i] - u_sol2[j][i])

i=0
j=1
print(u_sol1[j][i])
print(u_sol2[j][i])
print(u_sol1[j][i] - u_sol2[j][i])

i=1
j=1
print(u_sol1[j][i])
print(u_sol2[j][i])
print(u_sol1[j][i] - u_sol2[j][i])