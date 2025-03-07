import os
import json
from marshmallow import Schema, fields, validate # type: ignore
import random
import numpy as np

ALL_COLORS = ['red', 'blue']
ALL_OBJ_TYPE = ['ball', 'cube']
ALL_COMMANDS = [
    'to', 
    'left', 'right', 'behind', 'up', 'down',
    'up_right', 'up_left', 'up_behind', 
    'down_right', 'down_left', 'down_behind'
]

class SimEnvInitSchema(Schema):
    drones_loc = fields.List(
        fields.Tuple(fields.Float, fields.Float, fields.float), 
        required=True
    )
    objects_loc = fields.List(
        fields.Tuple(fields.Float, fields.Float, fields.float), 
        required=True
    )
    objects_color = fields.List(
        fields.String(validate=validate.OneOf(ALL_COLORS)), 
        required=True
    )
    objects_type = fields.List(
        fields.String(validate=validate.OneOf(ALL_OBJ_TYPE)), 
        required=True
    )
    drones_targets = fields.List(fields.Dict(
        keys = fields.Int(),
        values = fields.Str(validate=validate.OneOf(ALL_COMMANDS))
    ), required=False)
    theta_offset = fields.Float(required=True)
    theta_environment = fields.Float(required=True)
    PYBULLET_TO_GS_SCALING_FACTOR = fields.Float(required=True)

def parse_init_conditions(sim_dir):
    _schema = SimEnvInitSchema()
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'r') as f:
        init_conditions = json.load(f)
    return _schema.load(init_conditions)


def generate_closed_loop_2obj_env_init(
        objects_color, 
        PYBULLET_TO_GS_SCALING_FACTOR = 1.0
    ) -> SimEnvInitSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- N Choice

    """
    pybullet_rand_forward = random.uniform(1.5, 2)
    orthogonal_dist = 0.2

    # Convert to GS
    gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
    gs_offsets_from_camera = [[0, 0, 0], [gs_rand_forward, -1.5*orthogonal_dist, 0], [gs_rand_forward, 1.5* orthogonal_dist, 0]] # forward, right, up
    # gs_offsets_from_camera = [[0, 0, 0], [gs_rand_forward, -orthogonal_dist, 0], [gs_rand_forward, 0, 0], [gs_rand_forward, orthogonal_dist, 0]] # forward, right, up

    gs_offsets_xy = np.array(gs_offsets_from_camera)[1:, 0:2]
    gs_offsets_z = np.array(gs_offsets_from_camera)[1:, 2]

    max_yaw_offset = 0.1 * np.pi
    start_heights = [0.1 + 0.5]
    target_heights = ((0.1 + 0.5 + gs_offsets_z) / PYBULLET_TO_GS_SCALING_FACTOR).tolist()
    theta_offset = random.uniform(0, max_yaw_offset)
    theta_environment = random.random() * 2 * np.pi

    # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
    objects_relative = gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR * np.array([1, -1])

    # shuffle the order of the objects in the string list
    random.shuffle(objects_color)

    _schema = SimEnvInitSchema()
    init_conditions = {
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": pybullet_rand_forward,
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": objects_color,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
        "gs_objects_relative": np.array(gs_offsets_from_camera)[1:, 0:2].tolist()
    }
    init_conditions = _schema.load(init_conditions)

    return init_conditions