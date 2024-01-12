from scipy import integrate
from scipy.optimize import fmin
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import math

# ---------------------------------------------------------------------------------------------------------------------
print("------------------ Task 1 – Integration, Curves, Surface integrals, directional derivatives ------------------")


# ---------------------------------------------------------------------------------------------------------------------
def squared(x): return x ** 2


def Rie_sum(f, a, b, n):
    #
    # Calculates the left riemann sum for a function, f.
    # First finds the delta, d, for each x-value.
    # Then for n rectangles, starting at x_0 = a, sums all the areas, ending at x_0 = b-d.
    #
    d = (b - a) / n
    ar = 0
    for x in range(n):
        ar += d * f(a)
        a += d
    return ar


# Verification of simple function f(x) = x^2
realValue = integrate.quad(squared, 23, 42)[0]
n = 2

for i in range(5):
    approximation = Rie_sum(squared, 23, 42, n)
    error = abs(realValue - approximation)

    print(f"For n={n}, Error: {error}")

    n *= 20


# Computing the length of the plane curve
def func(t):
    return np.sqrt((2 * t) ** 2 + (3 * t ** 2) ** 2)


realValue = integrate.quad(func, -2, 1)[0]
approximation = Rie_sum(func, -2, 1, 1000000)
error = abs(realValue - approximation)
print(f"Length of the curve using Riemann sum: {approximation}")
print(f"Error: {error}")

# ---------------------------------------------------------------------------------------------------------------------
print("\n------------------------------------- Task 2 – Optimization, Gradients --------------------------------------")


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

print("\nComment: we can observe that there does not appear to be any local minimums as the coordinates are very large "
      "\n\t\t for each possible value. However, there appears to be one local maximum of 4 at x=1, y=2")

# ---------- Trace the iterates in your contour plot. ----------
result = fmin(fNegative, (0, 0), disp=False, retall=True)[1]

# Generate x y, and z values
x = np.linspace(0, 2, 400)
y = np.linspace(0, 4, 400)
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

# ---------------------------------------------------------------------------------------------------------------------
print("\n------------------------------- Task 3 – Implicit functions, curves, surfaces -------------------------------")


# ---------------------------------------------------------------------------------------------------------------------

# ---------- Illustrate the smoothness ----------
def func(z, x, y):
    return x + 2 * y + z + np.exp(2 * z) - 1


def z(params):
    x, y = params
    return fsolve(func, 0, args=(x, y))[0]


x_range = np.linspace(-1, 1, 50)
y_range = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x_range, y_range)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = z((X[i, j], Y[i, j]))

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('3D plot of z(x, y) in the neighborhood of the origin')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')


# ---------- Determine the coefficients of the Taylor polynomial. ----------
def nthPartial(n, wrt):
    h = np.sqrt(np.finfo(float).eps)
    nC = lambda k: math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    if wrt == 'xy':
        res = (z((h, h)) - z((-h, h)) - z((h, - h)) + z((-h, -h))) / (4 * h ** 2)
    else:
        vals = np.zeros(n + 1)
        for k in range(n + 1):
            if wrt == 'x':
                vals[k] = (-1) ** (k + n) * nC(k) * z((k * h, 0))
            elif wrt == 'y':
                vals[k] = (-1) ** (k + n) * nC(k) * z((0, k * h))
        res = (1 / (h ** n)) * np.sum(vals)

    return res


f = z((0, 0))
dfx = nthPartial(1, 'x')
dfy = nthPartial(1, 'y')
dfxx = nthPartial(2, 'x')
dfxy = nthPartial(2, 'xy')
dfyy = nthPartial(2, 'y')

ic(f)
ic(dfx)
ic(dfy)
ic(dfxx)
ic(dfxy)
ic(dfyy)


# ---------- Make a 3D plot of P2(x, y) ----------
def P2(x, y):
    return f + dfx * x + dfy * y + (1 / 2) * (dfxx * (x ** 2) + 2 * dfxy * x * y + dfyy * (y ** 2))


fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, P2(X, Y), cmap='viridis', edgecolor='none')
ax2.set_title('3D plot of P2(x, y) centered around (0, 0)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('P2')


# ---------- Plot the absolute error ----------
def e(x, y):
    return np.abs(z((x, y)) - P2(x, y))


error = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        error[i, j] = e(X[i, j], Y[i, j])

fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, error, cmap='viridis', edgecolor='none')
ax3.set_title('3D plot of the absolute error')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Error')

plt.show()

# The error appears reasonable given the value decreasing the closer we get to the point of center (i.e.,
# the origin). Since we are wokring with a second degree polynomial, we can also observe some quadratic behavior of
# the function as it approaches the boarders of the graph.
