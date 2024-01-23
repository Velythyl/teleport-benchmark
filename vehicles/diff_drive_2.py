"""
Example diffdrive_kinematic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import matplotlib

from vehicles._vehicle import DiffDrive, SpeedAngularDiffDrive

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Set the track of the vehicle [m]
ELL = 0.35



# %%
# FUNCTION DEFINITIONS

def openloop(t):
    """Specify open loop speed and angular rate inputs."""
    v = 0.5
    omega = 0.5 * np.sin(10 * t * np.pi / 180.0)
    return np.array([v, omega])


# %%
# RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((3, N))
u = np.zeros((2, N))

# Set the initial pose [m, m, rad], velocities [m/s, m/s]
x[0, 0] = 0.0
x[1, 0] = 0.0
x[2, 0] = np.pi / 2.0

vehicle = SpeedAngularDiffDrive(cur_x=x[0,0], cur_y=x[1,0], cur_rad=x[2,0], ell=ELL, tick_speed=T)

u[:, 0] = openloop(t[0])

# Run the simulation
for k in range(1, N):
    vehicle = vehicle.control(u[:,k-1])
    x[:, k] = vehicle.cur  #rk_four(diffdrive_f, x[:, k - 1], u[:, k - 1], T)
    u[:, k] = openloop(t[k])

# %%
# MAKE PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x[0, :])
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(412)
plt.plot(t, x[1, :])
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x[2, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C1", where="post", label="$v_L$")
plt.step(t, u[1, :], "C2", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel(r"$\bm{u}$ [m/s]")
plt.xlabel(r"$t$ [s]")
plt.legend()

# Save the plot
plt.savefig("./diffdrive_kinematic_fig1.pdf")

# Let's now use the class DiffDrive for plotting

from OLD.mobotpy.models import DiffDrive
vehicle = DiffDrive(ELL)

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x[0, :], x[1, :], "C0")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
plt.savefig("./diffdrive_kinematic_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(x, T, True, "./diffdrive_kinematic.gif")

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# from IPython.display import display
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()"""""