import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# ---------------------------------------------------------------------------------------------------------------------
print("-------------------------------------- Task 2 – Optimization, Gradients --------------------------------------")
# ---------------------------------------------------------------------------------------------------------------------


# ---------- Graphically determine local extreme values using a contour plot. ----------

# Define the function
def f(params):
    x, y = params
    return 8 * x * y - 4 * x ** 2 * y - 2 * x * y ** 2 + x ** 2 * y ** 2


# ---------- Numerically compute local extreme values using scipy.optimize.fmin. ----------
def fNegative(params):
    return -f(params)


initialGuesses = np.array([(-1, -1),
                           (-1, 0),
                           (-1, 1),
                           (0, 0),
                           (0, 1),
                           (1, 1)])

print("\nPotential local minimums")
for guess in initialGuesses:
    min = fmin(f, guess, disp=0)
    print(f(min), " at ", min)

print("\nPotential local maximums")
for guess in initialGuesses:
    max = fmin(fNegative, guess, disp=0)
    print(f(max), " at ", max)

print("\nComment: we can observe that there does not appear to be any local minimums as the coordinates are enormous "
      "\n\t\t for each possible value. However, there appears to be one local maximum of 4 at x=1, y=2")

# ---------- Trace the iterates in your contour plot. ----------
result = fmin(fNegative, (0, 0), disp=False, retall=True)[1]

# Generate x y, and z values
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f(np.meshgrid(x, y))

# Plot the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, 50, cmap='seismic')
plt.colorbar(contour)
plt.title('Contour plot of f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

iterates_x = [point[0] for point in result]
iterates_y = [point[1] for point in result]
plt.plot(iterates_x, iterates_y, 'ro-', markersize=5, label='Iterates')
plt.legend()
plt.show()
