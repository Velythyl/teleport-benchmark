import jax
from flax import struct
from jax import numpy as jp

from vehicles.utils import rk_four


@struct.dataclass
class _Vehicle:
    cur_x: float
    cur_y: float

    cur_pos_vel: float
    cur_ang_vel: float

    @property
    def cur_pos(self):
        return jp.array([self.cur_x, self.cur_y])

    def cur_keys(self):
        raise NotImplementedError()

    def cur_vals(self):
        ret = []
        for k in self.cur_keys():
            ret.append( dict(vars(self))[k] )
        return jp.array(ret)

    def cur_dict(self):
        ret = {}
        for k, v in zip(self.cur_keys(), self.cur_vals()):
            ret[k] = v
        return ret

    @property
    def cur(self):
        return self.cur_vals()

    tick_speed: float

    def f(self, x, u):
        raise NotImplementedError()

    def parse_control(self, u):
        return u

    def pre_control(self, u):
        return u, self.cur

    @jax.jit
    def _control(self, u):
        u = self.parse_control(u)
        u, cur = self.pre_control(u)

        x = rk_four(self.f, cur, u, self.tick_speed)

        ret = dict(vars(self))
        idx = 0
        for k, v in zip(self.cur_keys(), self.cur_vals()):
            ret[k] = x[idx]
            idx += 1

        return self.replace(
            **ret
        )

    def control(self, pos_vel_delta, ang_vel_delta):
        new_pos_vel = self.cur_pos_vel + pos_vel_delta
        new_ang_vel = self.cur_ang_vel + ang_vel_delta

        new_vehicle = self._control(u=jp.array([new_pos_vel, new_ang_vel]))
        return new_vehicle.replace(cur_pos_vel=new_pos_vel, cur_ang_vel=new_ang_vel)

@struct.dataclass
class _Vehicle_1Angle(_Vehicle):
    cur_rad_1: float

    def cur_keys(self):
        return ["cur_x", "cur_y", "cur_rad_1"]

@struct.dataclass
class Unicycle(_Vehicle_1Angle):
    cur_v_x: float
    cur_v_y: float
    cur_v_2: float

    def cur_keys(self):
        return super(Unicycle, self).cur_keys() + ["cur_v_x", "cur_v_y", "cur_v_2"]

    mass: float
    inertia_moment: float
    max_lateral_force: float = 0.1

    def lateral_force(self, x):
        """Computes the lateral tire force for a single wheel.

        Parameters
        ----------
        x : ndarray of length 6
            The vehicle's state (x, y, theta, v_x, v_y, v_2).

        Returns
        -------
        lambda_f : float
            The computed lateral tire force [N].
        old_x : ndarray of length 6
            The vehicle's state with or without slip.
        """

        # Compute lateral force
        lambda_f = self.mass * x[5] * (x[3] * jp.cos(x[2]) + x[4] * jp.sin(x[2]))

        def slip():
            old_vx = x[3]
            old_vy = x[4]
            return old_vx, old_vy
        def no_slip():
            old_vx = (x[3] * jp.cos(x[2]) + x[4] * jp.sin(x[2])) * jp.cos(x[2])
            old_vy = (x[3] * jp.cos(x[2]) + x[4] * jp.sin(x[2])) * jp.sin(x[2])
            return old_vx, old_vy

        old_vx, old_vy = jax.lax.cond(
            jp.abs(lambda_f) > self.max_lateral_force,
            slip,
            no_slip
        )

        # Assign the new state
        old_x = jp.array([x[0], x[1], x[2], old_vx, old_vy, x[5]])

        # Return the output
        return jp.clip(lambda_f, -self.max_lateral_force, self.max_lateral_force), old_x

    def pre_control(self, u):
        lateral_force, new_cur = self.lateral_force(self.cur)
        new_u = jp.array([u[0], u[1], lateral_force])
        return new_u, new_cur

    def f(self, x, u):
        """Unicycle dynamic vehicle model.

        Parameters
        ----------
        x : ndarray of length 6
            The vehicle's state (x, y, theta, v_x, v_y, v_2).
        u : ndarray of length 3
            Force and torque applied to the wheel (f, tau, lambda_f).

        Returns
        -------
        f_dyn : ndarray of length 6
            The rate of change of the vehicle states.
        """
        f_dyn = jp.zeros(6)
        f_dyn = f_dyn.at[0].set( x[3])
        f_dyn = f_dyn.at[1].set( x[4])
        f_dyn = f_dyn.at[2].set( x[5])
        f_dyn = f_dyn.at[3].set( 1.0 / self.mass * (u[0] * jp.cos(x[2]) - u[2] * jp.sin(x[2])))
        f_dyn = f_dyn.at[4].set( 1.0 / self.mass * (u[0] * jp.sin(x[2]) + u[2] * jp.cos(x[2])))
        f_dyn = f_dyn.at[5].set( 1.0 / self.inertia_moment * u[1])
        return f_dyn





@struct.dataclass
class _Vehicle_2Angle(_Vehicle_1Angle):
    cur_rad_2: float

    def cur_keys(self):
        return ["cur_x", "cur_y", "cur_rad_1", "cur_rad_2"]


@struct.dataclass
class Trike(_Vehicle_2Angle):
    ell_W: float
    ell_T: float


    def f(self, x, u):
        """Tricycle vehicle kinematic model."""
        f = jp.zeros(4)
        f = f.at[0].set( u[0] * jp.cos(x[2]))
        f = f.at[1].set( u[0] * jp.sin(x[2]) )
        f = f.at[2].set( u[0] * 1.0 / self.ell_W * jp.tan(x[3]))
        f = f.at[3].set( u[1])
        return f

@struct.dataclass
class Ackermann(_Vehicle_2Angle):
    ell_W: float
    ell_T: float


    def f(self, x, u):
        """Ackermann steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's speed and steering angle rate.

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = jp.zeros(4)
        f = f.at[0].set( u[0] * jp.cos(x[2]))
        f = f.at[1].set( u[0] * jp.sin(x[2]) )
        f = f.at[2].set( u[0] * 1.0 / self.ell_W * jp.tan(x[3]))
        f = f.at[3].set( u[1] )
        return f

    def ackermann(self, x):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_L = jp.arctan(
            2 * self.ell_W * jp.tan(x[3]) / (2 * self.ell_W - self.ell_T * jp.tan(x[3]))
        )
        phi_R = jp.arctan(
            2 * self.ell_W * jp.tan(x[3]) / (2 * self.ell_W + self.ell_T * jp.tan(x[3]))
        )
        ackermann_angles = jp.array([phi_L, phi_R])
        return ackermann_angles

@struct.dataclass
class DiffDrive(_Vehicle_1Angle):
    ell: float  # track length


    def f(self, x, u):
        """Differential drive kinematic vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 3
            The vehicle's state (x, y, theta).
        u : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).

        Returns
        -------
        f : ndarray of length 3
            The rate of change of the vehicle states.
        """
        f = jp.zeros(3)
        f = f.at[0].set( 0.5 * (u[0] + u[1]) * jp.cos(x[2]))
        f = f.at[1].set( 0.5 * (u[0] + u[1]) * jp.sin(x[2]))
        f = f.at[2].set( 1.0 / self.ell * (u[1] - u[0]) )
        return f

    def uni2diff(self, u_in):
        """
        Convert speed and anular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        """
        v = u_in[0]
        omega = u_in[1]
        v_L = v - self.ell / 2 * omega
        v_R = v + self.ell / 2 * omega
        u_out = jp.array([v_L, v_R])
        return u_out

@struct.dataclass
class SpeedAngularDiffDrive(DiffDrive):
    def parse_control(self, u_in):
        """
        Convert speed and anular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        """
        v = u_in[0]
        omega = u_in[1]
        v_L = v - self.ell / 2 * omega
        v_R = v + self.ell / 2 * omega
        u_out = jp.array([v_L, v_R])
        return u_out

