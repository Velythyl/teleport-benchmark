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

    @classmethod
    def init(cls, img, grid):
        valid_poses = jp.where(grid == 0)
        valid_poses = jp.vstack(valid_poses).T.astype(jp.float32)
        return Map(img, grid, valid_poses)

    @jax.jit
    def sample_valid_pose(self, key):
        return jax.random.choice(key, self.valid_poses)

    @jax.jit
    def is_pose_valid(self, pos):
        pos = pos.astype(jp.int32)
        return jp.logical_not(self.grid[pos[0], pos[1]] > 0)    # could also do ``== 0'' but this way we handle all float cases or wtv

    @partial(jax.jit, static_argnums=(3,))
    def _affected_cells(self, old_pos, new_pos, NUM_TICKS=100):
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
        affected_cells, corresponding_walkers, types = self._affected_cells(old_pos, new_pos)
        issue = jp.argmax(types)
        return issue > 0, jax.lax.cond(
            issue > 0,
            lambda : corresponding_walkers[issue-1],
            lambda : jp.array(new_pos)
        )

    @partial(jax.jit, static_argnums=(3,))
    def omnidirectional_lidar(self, pos, range: float, number_of_rays=1000) -> GroundtruthSensorReply:
        pos = jp.array(pos)

        # do all angles up to pi
        angles = jp.linspace(0, jp.pi * 2, num=number_of_rays)

        #x = angles_to_poses(pos, angles, range)
        #cosses = jp.cos(angles)
        #sinnes = jp.sin(angles)
        #new_poses = jp.vstack((cosses, sinnes)).T * range

        new_poses = angles_to_poses(pos, angles, range)

        def find_walker(new_pos):
            affected_cells, corresponding_walkers, types = self._affected_cells(pos, new_pos, NUM_TICKS=1000)
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


