from functools import partial

import jax.random
from flax import struct
from jax import numpy as jp
from matplotlib import pyplot as plt

from plt2gif import Plt2Gif
from simulator.map import Map
from simulator.read_map import load_png
from simulator.sensor import _SensorNoise, SensorReply, GroundtruthSensorReply, NoNoise, GaussianNoise
from simulator.sim import Sim
from vehicles.vehicle import _Vehicle, SpeedAngularDiffDrive


@struct.dataclass
class Observation:
    vehicle: _Vehicle
    sr: SensorReply

@struct.dataclass
class ObservationFramestack:
    obs: Observation
    index: int
    max_num: int

    @jax.jit
    def push(self, new_obs: Observation):
        flat_stack, obs_def = jax.tree_util.tree_flatten(self.obs)
        flat_obs, _ = jax.tree_util.tree_flatten(new_obs)

        ret_obs = []
        for ob, new_ob in zip(flat_stack, flat_obs):
            ret_obs.append(ob.at[self.index].set(new_ob))
        ret_obs = jax.tree_util.tree_unflatten(obs_def, ret_obs)

        index = jp.clip(self.index + 1, 0, self.max_num-1)
        return self.replace(obs=ret_obs, index=index)


@struct.dataclass
class _Gym:
    pass

@struct.dataclass
class Gym(_Gym):
    sim: Sim
    sensor: _SensorNoise

    framestack: Observation
    action_repeat: int

    @classmethod
    def init(cls, sim: Sim, sensor: _SensorNoise, num_framestack, action_repeat=1):
        gt_sr = sim.sense()
        sr = NoNoise()(None, gt_sr)

        obs = Observation(sim.vehicle, sr)

        flat_obs, obs_def = jax.tree_util.tree_flatten(obs)
        new_dim = jp.arange(num_framestack)

        def clone(_, inp):
            return inp
        expanded_obs = jax.tree_util.tree_unflatten(obs_def, [jax.vmap(partial(clone, inp=i))(new_dim) for i in flat_obs])

        return Gym(sim, sensor, framestack=ObservationFramestack(expanded_obs, 0, num_framestack), action_repeat=action_repeat)

    @jax.jit
    def step(self, key: jax.random.PRNGKey, action: jp.array) -> (_Gym, ObservationFramestack):
        #for _ in range(self.action_repeat): todo
        new_sim = self.sim.control_(action)

        gt_sr = new_sim.sense()
        sr = self.sensor(key, gt_sr)

        obs = Observation(new_sim.vehicle, sr)
        new_framestack = self.framestack.push(obs)

        return self.replace(sim=new_sim, framestack=new_framestack), new_framestack


if __name__ == "__main__":

    img, test_grid = load_png("../maps/costmap_full_room.png") #load_ascii("../maps/WarCraft III/Maps/bootybay.map")

    """
    test_grid = jp.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0]])"""
    old_pos = (250.,400.)
    #new_pos = (120.,140.)
    map = Map.init(img, test_grid)#.truncate_move(old_pos, new_pos)
    vehicle = SpeedAngularDiffDrive(cur_x=250., cur_y=400., cur_rad_1=0, ell=0.35, tick_speed=0.04, cur_pos_vel=100., cur_ang_vel=0)

    gym = Gym.init(Sim(map, vehicle, sensor_range=50.), GaussianNoise(0.2), 4, action_repeat=1)

    ret = gym.sim.map.shunt(jp.array([0,0]))

    import matplotlib
    matplotlib.use("TkAgg")
    gifmanager = Plt2Gif("../figs")

    key = jax.random.PRNGKey(0)

    for i in range(100):
        gym.sim.draw(lidar_hits=None)


        #plt.imshow(img, interpolation='nearest')
        #_, _, lidar_hits = sim.sense()
        #for new_pos in lidar_hits:
        #    old_pos = sim.vehicle.cur_pos
        #    plt.plot([old_pos[0]-0.5, new_pos[0]-0.5], [old_pos[1]-0.5, new_pos[1]-0.5], color="red")
        gifmanager.plt_show(save=False, show=False)
        plt.close()

        if (i % 10) == 0:
            key, rng = jax.random.split(key)
            sim = gym.sim.teleport_to_random_pos_(rng)
            gym = gym.replace(sim=sim)
        #plt.show(block=False)
        #plt.pause(0.1)
        key, rng = jax.random.split(key)
        gym, obs = gym.step(rng, jp.zeros(2))
    gifmanager.get_gif(save=True, show=False, path="../figs/evolution.gif")
    exit()



