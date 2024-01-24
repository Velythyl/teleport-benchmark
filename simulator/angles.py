from jax import numpy as jp


def angles_to_poses(pos, angles, ranges):
    pos = jp.broadcast_to(pos[None], (angles.shape[0], 2))

    cosses = jp.cos(angles)
    sinnes = jp.sin(angles)

    new_poses = jp.vstack((cosses, sinnes)).T * ranges

    return pos + new_poses
