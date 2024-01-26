from functools import partial

import jax.lax
import jax.random
from flax import struct
from jax import numpy as jp

from simulator.angles import angles_to_poses
from simulator.sensor import GroundtruthSensorReply


@struct.dataclass
class Map:
    img: jp.array
    grid: jp.array
    valid_poses: jp.array
    low_bound: int
    max_bound: int

    @classmethod
    def init(cls, img, grid):
        valid_poses = jp.where(grid == 0)
        valid_poses = jp.vstack(valid_poses).T.astype(jp.float32)

        assert grid.shape[0] == grid.shape[1]

        return Map(img, grid, valid_poses, low_bound=0, max_bound=grid.shape[0])

    @jax.jit
    def sample_valid_pose(self, key):
        return jax.random.choice(key, self.valid_poses)

    @jax.jit
    def is_pose_valid(self, pos):
        pos = pos.astype(jp.int32)
        return jp.logical_not(self.grid[pos[0], pos[1]] > 0)    # could also do ``== 0'' but this way we handle all float cases or wtv

    @partial(jax.jit, static_argnums=(3,))
    def _raytrace(self, old_pos, new_pos, NUM_TICKS=100):
        # janky version of Bresenham's line; returns every affected cell instead of the "most important"

        old_pos = jp.array(old_pos)
        new_pos = jp.array(new_pos)

        direction = new_pos - old_pos

        walkers = jp.copy(old_pos)

        def cell_for_pos(pos):
            return pos.astype(int)

        affected_cells = [cell_for_pos(old_pos)]
        corresponding_walkers = [old_pos]

        ticks = direction / NUM_TICKS
        for n in range(NUM_TICKS):
            walkers = walkers + ticks
            affected_cells.append(cell_for_pos(walkers))
            corresponding_walkers.append(walkers)

        affected_cells.append(cell_for_pos(new_pos))
        corresponding_walkers.append(walkers)

        affected_cells = jp.array(affected_cells)
        corresponding_walkers = jp.array(corresponding_walkers)

        def cell2type(cell):
            type = self.grid[cell[0], cell[1]]
            return type

        return affected_cells, corresponding_walkers, jax.vmap(cell2type)(affected_cells)

    @jax.jit
    def truncate_move(self, old_pos, new_pos):
        affected_cells, corresponding_walkers, types = self._raytrace(old_pos, new_pos)
        collision_index = jp.argmax(types) # finds first index with non-zero type
        collision = types[collision_index] > 0

        return collision, jax.lax.cond(
            collision,  # if there's an
            lambda : corresponding_walkers[collision_index-1],  # go to nearest index before collision
            lambda : jp.array(new_pos)  # otherwise proceed as usual
        )

    @partial(jax.jit, static_argnums=(3,4))
    def _omnidirectional_walker(self, pos, range: float, number_of_rays=1000, NUM_TICKS=1000):
        pos = jp.array(pos)

        # do all angles up to pi
        angles = jp.linspace(0, jp.pi * 2, num=number_of_rays)

        new_poses = angles_to_poses(pos, angles, range)

        def get_walker(new_pos):
            return self._raytrace(pos, new_pos, NUM_TICKS=NUM_TICKS)
        return jax.vmap(get_walker)(new_poses)

    @partial(jax.jit, static_argnums=(2,3))
    def shunt(self, pose, number_of_rays=50, NUM_TICKS=100):
        pose = jp.clip(pose, self.low_bound, self.max_bound)

        RANGE = jp.sqrt(self.low_bound ** 2 + self.max_bound ** 2) + 2  # 2 is padding
        # returns closest valid pose
        # todo find distance of furthest point from valid pos and use that instead of 150
        _, walkers, types = self._omnidirectional_walker(pose, RANGE, number_of_rays=number_of_rays, NUM_TICKS=NUM_TICKS)
        found_id = jp.argmin(types, axis=1)

        transition_point = walkers[jp.arange(number_of_rays),found_id]
        distance = jp.linalg.norm(transition_point - pose, axis=1)

        def coerce_dist(d):
            # cant use non-concrete boolean masks, so we gotta vmap over the array instead
            return jax.lax.cond(
                d == 0.0,
                lambda: jp.inf,    # impossible to attain normally
                lambda: d
            )
        distance = jax.vmap(coerce_dist)(distance)
        selected_ray = jp.argmin(distance)
        selected_walker = transition_point[selected_ray]

        return selected_walker

    @partial(jax.jit, static_argnums=(3,))
    def omnidirectional_lidar(self, pos, range: float, number_of_rays=1000) -> GroundtruthSensorReply:
        # todo refactor this to use the generic func

        pos = jp.array(pos)

        # do all angles up to pi
        angles = jp.linspace(0, jp.pi * 2, num=number_of_rays)

        new_poses = angles_to_poses(pos, angles, range)

        def find_walker(new_pos):
            affected_cells, corresponding_walkers, types = self._raytrace(pos, new_pos, NUM_TICKS=1000)
            hit_obstacle = jp.argmax(types)

            hits = (corresponding_walkers[hit_obstacle-1] + corresponding_walkers[hit_obstacle]) / 2
            distance = jp.linalg.norm(hits - pos)

            def real_hit():
                return distance, hits
            def no_hit():
                return range, corresponding_walkers[-1]

            return jax.lax.cond(
                hit_obstacle > 0,
                real_hit,
                no_hit
            )
        distance, hit_pos = jax.vmap(find_walker)(new_poses)
        return GroundtruthSensorReply(gt_pos=pos, gt_angles=angles, gt_distances=distance, gt_hits=hit_pos)


