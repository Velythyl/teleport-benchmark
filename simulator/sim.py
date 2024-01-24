import io

import distrax
import jax.image
import numpy as np
from flax import struct
from jax import numpy as jp

# https://www.movingai.com/benchmarks/grids.html
from jax._src.tree_util import Partial
from matplotlib import colors, pyplot as plt

from simulator.map import Map
from simulator.read_map import load_ascii, load_png
from simulator.sensor import NoNoise, _SensorNoise, GroundtruthSensorReply, SensorReply
from vehicles.vehicle import _Vehicle, SpeedAngularDiffDrive


@struct.dataclass
class _Sim:
    pass


@struct.dataclass
class Sim(_Sim):
    map: Map
    vehicle: _Vehicle

    key: jax.random.PRNGKey

    sensor_range: float = 50

    @jax.jit
    def split_key_(self) -> (_Sim, jax.random.PRNGKey):
        key, rng = jax.random.split(self.key)
        return self.replace(key=key), rng

    @jax.jit
    def teleport_to_pos_(self, new_x, new_y) -> _Sim:
        vehicle = self.vehicle.replace(cur_x=new_x, cur_y=new_y)
        return self.set_vehicle_(vehicle)

    @jax.jit
    def set_vehicle_(self, vehicle: _Vehicle) -> _Sim:
        return jax.lax.cond(
            self.map.is_pose_valid(vehicle.cur_pos),
            lambda: self.replace(vehicle=vehicle),
            lambda: self
        )

    @jax.jit
    def teleport_to_random_pos_(self) -> _Sim:
        # jank, might move the key out of the class, but right now i just want something simple and to not deal with the keys myself
        self, rng = self.split_key_()
        new_pos = self.map.sample_valid_pose(rng)
        return self.teleport_to_pos_(new_pos[0], new_pos[1])

    @jax.jit
    def sense_vehicle(self, vehicle) -> GroundtruthSensorReply:
        sensor_reply = self.map.omnidirectional_lidar(pos=vehicle.cur_pos, range=self.sensor_range, number_of_rays=50)
        return sensor_reply


    @jax.jit
    def sense(self) -> GroundtruthSensorReply:
        return self.sense_vehicle(self.vehicle)

    @jax.jit
    def control_from_vehicle_(self, new_vehicle) -> _Sim:
        # allows user to use fine-grained vehicle-specific control schemes instead of the generic ``control'' method
        issue, truncated = self.map.truncate_move(self.vehicle.cur_pos, new_vehicle.cur_pos)

        new_vehicle = jax.lax.cond(
            issue,
            lambda : new_vehicle.replace(cur_x=truncated[0], cur_y=truncated[1]),
            lambda : new_vehicle,
        )

        return self.replace(vehicle=new_vehicle)

    @jax.jit
    def control_(self, u) -> _Sim:
        vehicle = self.vehicle.control(u[0], u[1])
        return self.control_from_vehicle_(vehicle)

    def draw(self, vehicle=None, lidar_hits=[]) -> None:
        # allows playback of sim by simply recording each vehicle position

        # plot map
        plt.imshow(self.map.img, interpolation='nearest')

        if vehicle is None:
            vehicle = self.vehicle
        cur_pos = vehicle.cur_pos

        if lidar_hits is None:
            sensor_reply = self.sense_vehicle(vehicle)
            lidar_hits = sensor_reply.gt_hits

        for new_pos in lidar_hits:
            plt.plot([cur_pos[0]-0.5, new_pos[0]-0.5], [cur_pos[1]-0.5, new_pos[1]-0.5], color="red")

        circle1 = plt.Circle((cur_pos[0], cur_pos[1]), 0.2, color='r')
        plt.gca().add_patch(circle1)





if __name__ == "__main__":
    class Plt2Gif:
        def __init__(self, path=None):
            self.frames = []

            if path is None:
                path = f"/tmp/plt2gif"

            self.basepath = path

        def resolve_path(self, path1, path2):
            if path1 is None:
                assert path2 is not None
                return path2
            return path1

        @property
        def counter(self):
            return len(self.frames)

        def plt_show(self, save, show, path=None):
            import numpy as np

            fig = plt.gcf()
            if save:
                path = self.resolve_path(path, f"{self.basepath}/fig{self.counter}.png")
                fig.savefig(path)
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            im = data.reshape((int(h), int(w), -1))

            self.frames.append(im)

            if show:
                plt.show()

        def get_gif(self, save, show, path=None):
            from matplotlib import animation
            fig, ax = plt.subplots()
            ax.axis("off")
            fig.tight_layout()

            im = ax.imshow(self.frames[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

            def animate(i):
                im.set_array(self.frames[i + 1])
                return [im]

            anime = animation.FuncAnimation(fig, animate, frames=self.counter - 1)
            if save:
                path = self.resolve_path(path, f"{self.basepath}/evolution.png")
                anime.save(path, fps=10)
            if show:
                plt.show()

        def reset(self):
            self.frames = []


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

    sim = Sim(map, vehicle, sensor_range=50., key=jax.random.PRNGKey(0))

    import matplotlib
    matplotlib.use("TkAgg")
    gifmanager = Plt2Gif("../figs")
    for i in range(100):
        sim.draw(lidar_hits=None)

        #plt.imshow(img, interpolation='nearest')
        #_, _, lidar_hits = sim.sense()
        #for new_pos in lidar_hits:
        #    old_pos = sim.vehicle.cur_pos
        #    plt.plot([old_pos[0]-0.5, new_pos[0]-0.5], [old_pos[1]-0.5, new_pos[1]-0.5], color="red")
        gifmanager.plt_show(save=False, show=False)
        plt.close()

        if (i % 10) == 0:
            sim = sim.teleport_to_random_pos_()
        #plt.show(block=False)
        #plt.pause(0.1)

        sim = sim.control_(jp.zeros(2))
    gifmanager.get_gif(save=True, show=False, path="../figs/evolution.gif")
    exit()

    v = map.sample_valid_pose(jax.random.PRNGKey(0))
    _, _, lidar_hits = map.omnidirectional_lidar(old_pos, 200., number_of_rays=50)


    plt.imshow(img, interpolation='nearest')

    for new_pos in lidar_hits:
        plt.plot([old_pos[0]-0.5, new_pos[0]-0.5], [old_pos[1]-0.5, new_pos[1]-0.5], color="red")
    plt.show()
    exit()

    debug_grid = np.asarray(test_grid)
    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue'])
    bounds = [0,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(test_grid, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, test_grid.shape[-1], 1))
    ax.set_yticks(np.arange(-.5, test_grid.shape[-1], 1))
    ax.plot([old_pos[0]-0.5, new_pos[0]-0.5], [old_pos[1]-0.5, new_pos[1]-0.5], color="green")

    plt.show()
