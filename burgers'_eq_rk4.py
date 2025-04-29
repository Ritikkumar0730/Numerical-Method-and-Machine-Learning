import numpy as np
import matplotlib.pyplot as plt

# Define the spatial grid
L = 10.0   # Domain length
Nx = 100   # Number of spatial points
dx = L / Nx  # Grid spacing

# Define the time step
dt = 0.01
T = 2.0
Nt = int(T / dt)

# Define function for spatial derivative
def spatial_derivative(u):
    dudx = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)  # Central difference
    dudx[0] = (u[1] - u[0]) / dx   # Forward difference at boundary
    dudx[-1] = (u[-1] - u[-2]) / dx # Backward difference at boundary
    return dudx

# Define the function for time derivative (Advection equation)
def f(u):
    return -spatial_derivative(u)

# RK4 Step Function
def rk4_step(u, dt):
    k1 = dt * f(u)
    k2 = dt * f(u + k1 / 2)
    k3 = dt * f(u + k2 / 2)
    k4 = dt * f(u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6

# Initial condition: Gaussian pulse
x = np.linspace(0, L, Nx)
u = np.exp(-((x - L/2) ** 2) / 0.1)

# Time integration
for _ in range(Nt):
    u = rk4_step(u, dt)

# Plot result
plt.plot(x, u, label="Final Profile")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Advection with RK4")
plt.legend()
plt.show()
