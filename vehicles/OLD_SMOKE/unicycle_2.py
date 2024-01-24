"""
Example unicycle_dynamic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import matplotlib

from vehicles.OLD.mobotpy.models import DiffDrive
from vehicles.vehicle import Unicycle

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 60.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Set the vehicle's mass and moment of inertia
M = 1.0  # kg
I = 1.0  # kg m^2

# Set the maximum lateral tire force [N]
LAMBDA_MAX = 0.1

# %%
# FUNCTION DEFINITIONS


# %%
# RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((6, N))
u = np.zeros((3, N))

# Set the initial conditions
x_init = np.zeros(6)
x_init[0] = 0.0
x_init[1] = 3.0
x_init[2] = 0.0
x_init[3] = 0.0
x_init[4] = 0.0
x_init[5] = 0.0
x[:, 0] = x_init

vehicle = Unicycle(cur_x=x_init[0], cur_y=x_init[1], cur_rad_1=x_init[2], cur_v_x=x_init[3], cur_v_y=x_init[4], cur_v_2=x_init[5], tick_speed=T, mass=M, inertia_moment=I, max_lateral_force=LAMBDA_MAX)

# Run the simulation
for k in range(1, N):

    # Make some force and torque inputs to steer the vehicle around
    if k < round(N / 6):
        u[0, k - 1] = 0.1  # Apply "large" force
        u[1, k - 1] = 0.0
    elif k < round(3 * N / 6):
        u[0, k - 1] = 0.0  # Stop accelerating
        u[1, k - 1] = -0.01  # Start turning
    elif k < round(4 * N / 6):
        u[0, k - 1] = 0.0
        u[1, k - 1] = 0.02  # Turn the other way
    else:
        u[0, k - 1] = 0.0
        u[1, k - 1] = 0.0

    vehicle = vehicle._control(u[:, k - 1])


    # Compute the lateral force applied to the vehicle's wheel
    #u[2, k - 1], x[:, k - 1] = lateral_force(x[:, k - 1])

    # Update the motion of the vehicle
    x[:, k] = vehicle.cur #rk_four(unicycle_f_dyn, x[:, k - 1], u[:, k - 1], T)

# %%
# MAKE PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
ax1 = plt.subplot(311)
plt.setp(ax1, xticklabels=[])
plt.plot(t, x[0, :], "C0")
plt.ylabel(r"$x$ [m]")
plt.grid(color="0.95")
ax2 = plt.subplot(312)
plt.setp(ax2, xticklabels=[])
plt.plot(t, x[1, :], "C0")
plt.ylabel(r"$y$ [m]")
plt.grid(color="0.95")
ax3 = plt.subplot(313)
plt.plot(t, x[2, :], "C0")
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("./unicycle_dynamic_fig1.pdf")

# Plot the lateral tire force
fig2 = plt.figure(2)
plt.plot(t[0 : N - 1], u[2, 0 : N - 1], "C0")
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$\lambda$ [N]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("./unicycle_dynamic_fig2.pdf")

"""
To keep thing simple, we plot the unicycle as a differential drive vehicle,
because the differential drive vehicle has the same nonholonomic constraints.
"""

# Set the track of the vehicle [m]
ELL = 1.0

# Let's now use the class DiffDrive for plotting
vehicle = DiffDrive(ELL)

# Plot the position of the vehicle in the plane
fig3 = plt.figure(3)
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
plt.savefig("./unicycle_dynamic_fig3.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(x, T, True, "./unicycle_dynamic.gif")

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()"""""