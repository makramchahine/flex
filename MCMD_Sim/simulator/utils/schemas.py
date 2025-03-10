import os
import json
from marshmallow import Schema, fields, validate # type: ignore
import random
import numpy as np

ALL_COLORS = ['red', 'blue', 'green']
ALL_OBJ_TYPE = ['ball', 'cube']
ALL_COMMANDS = [
    'to', 
    'left', 'right', 'behind', 'up', 'down',
    'up_right', 'up_left', 'up_behind', 
    'down_right', 'down_left', 'down_behind'
]
ALL_ENV = ['arena', 'samurai']

class SimEnvInitSchema(Schema):
    drones_loc = fields.List(
        fields.Tuple((fields.Float, fields.Float, fields.Float)), 
        required=True
    )
    objects_loc = fields.List(
        fields.Tuple((fields.Float, fields.Float, fields.Float)), 
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
        values = fields.String(validate=validate.OneOf(ALL_COMMANDS))
    ), required=False)
    theta_offset = fields.Float(required=True)
    theta_environment = fields.Float(required=True)
    env_name = fields.String(validate=validate.OneOf(ALL_ENV), required=True)
    PYBULLET_TO_GS_SCALING_FACTOR = fields.Float(required=True)
    gs_objects_relative = fields.List(fields.List(fields.Float), required=True)
    sim_dir = fields.String(required=False)

def parse_init_conditions(sim_dir):
    _schema = SimEnvInitSchema()
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'r') as f:
        init_conditions = json.load(f)
    return _schema.load(init_conditions)

def generate_closed_loop_1drone_2ball_env_init(
        env_name='arena', 
        PYBULLET_TO_GS_SCALING_FACTOR = 1.0, 
        command=None
    ) -> SimEnvInitSchema:
    """
    Specific implementation with weighted probabilities
    """
    objects_color = random.sample(ALL_COLORS, 2)
    object_type = ['ball', 'ball']

    pybullet_rand_forward = random.uniform(1.5, 2)
    orthogonal_dist = 0.2
    # Convert to GS
    gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
    gs_offsets_from_camera = [
        [0, 0, 0], 
        [gs_rand_forward, -1.5 * orthogonal_dist, 0],
        [gs_rand_forward, 1.5 * orthogonal_dist, 0]
    ]  # forward, right, up
    
    gs_offsets_xy = np.array(gs_offsets_from_camera)[1:, 0:2]
    gs_offsets_z = np.array(gs_offsets_from_camera)[1:, 2]

    drones_loc = [(0, 0, random.uniform(0.1, 1.0))]
    command = command or random.choice(ALL_COMMANDS)
    drone_targets = [{0: command}]

    max_yaw_offset = 0.1 * np.pi
    theta_offset = random.uniform(0, max_yaw_offset)
    theta_environment = random.random() * 2 * np.pi

    # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
    objects_xy = gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR * np.array([1, -1])
    objects_z = ((0.1 + 0.5 + gs_offsets_z) / PYBULLET_TO_GS_SCALING_FACTOR).tolist()

    _schema = SimEnvInitSchema()
    init_conditions = {
        "drones_loc": drones_loc,
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_loc": [(x, y, h) for (x, y), h in zip(objects_xy, objects_z)],
        "objects_color": objects_color,
        "objects_type": object_type,
        "drones_targets": drone_targets,
        "env_name": env_name,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
        "gs_objects_relative": np.array(gs_offsets_from_camera)[1:, 0:2].tolist()
    }
    init_conditions = _schema.load(init_conditions)

    return init_conditions