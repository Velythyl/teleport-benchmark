import jax.random
from flax import struct
from jax import numpy as jp

from simulator.angles import angles_to_poses


@struct.dataclass
class GroundtruthSensorReply:
    gt_pos: jp.array
    gt_angles: jp.array
    gt_distances: jp.array
    gt_hits: jp.array

    def make_reply(self, angles, distances, hits):
        return SensorReply(
            angles=angles,
            distances=distances,
            hits=hits,
            gt=self
        )

@struct.dataclass
class SensorReply:
    angles: jp.array
    distances: jp.array
    hits: jp.array

    gt: GroundtruthSensorReply


@struct.dataclass
class _SensorNoise:
    def __call__(self, key, gt: GroundtruthSensorReply) -> SensorReply:
        raise NotImplementedError()

@struct.dataclass
class NoNoise(_SensorNoise):
    def __call__(self, key, gt: GroundtruthSensorReply) -> SensorReply:
        return gt.make_reply(angles=gt.gt_angles, distances=gt.gt_distances, hits=gt.gt_hits)

@struct.dataclass
class GaussianNoise:
    sigma: float = 0.2
    def __call__(self, key, gt: GroundtruthSensorReply) -> SensorReply:
        new_distances = gt.gt_distances + jax.random.normal(key, shape=(gt.gt_distances.shape[0]))
        new_hits = angles_to_poses(gt.gt_pos, gt.gt_angles, new_distances)

        return gt.make_reply(angles=gt.gt_angles, distances=new_distances, hits=new_hits)


