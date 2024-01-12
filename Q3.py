import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic

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
