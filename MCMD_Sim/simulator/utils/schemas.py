import os
import json
from marshmallow import Schema, fields, validate # type: ignore
import random
import numpy as np
from .tasks import get_task_delta

PYBULLET_TO_GS_SCALING_FACTOR = 1.0
ALL_COLORS = ['red', 'blue', 'colorless', 'green', 'yellow', 'purple', '']
ALL_OBJ_TYPE = ['ball', 'cube', 'donut', 'pyramid', 'jeep', 'horse', 'dog', 'robot', 'rocket', 'palmtree', 'watermelon']
ALL_COMMANDS = [
    'towards', 
    'left', 'right',
    'above', 'below',
    # 'behind', 
    # 'up_right', 'up_left', 'up_behind', 
    # 'down_right', 'down_left', 'down_behind'
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
    progress_scale = fields.Float(required=False)
    env_name = fields.String(validate=validate.OneOf(ALL_ENV), required=True)
    PYBULLET_TO_GS_SCALING_FACTOR = fields.Float(required=True)
    gs_objects_relative = fields.List(fields.List(fields.Float), required=True)
    log_dir = fields.String(required=False)
    data_dir = fields.String(required=False)

class SimEnvInit_1DroneSchema(SimEnvInitSchema):
    target_idx = fields.Int(required=True)
    command = fields.String(required=True, validate=validate.OneOf(ALL_COMMANDS))


def parse_init_conditions(sim_dir):
    _schema = SimEnvInitSchema()
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'r') as f:
        init_conditions = json.load(f)
    return _schema.load(init_conditions)

def generate_closed_loop_1drone_2ball_env_init(
        env_name='arena', 
        command=None,
        target_idx=None,
        objs = None,
        log_path=None,
        data_path=None,
        progress_scale = 0.0,
    ) -> SimEnvInit_1DroneSchema:
    """
    Specific implementation with weighted probabilities
    """
    if objs is None:
        objects_color = random.sample(ALL_COLORS, 2)
        object_type = ['ball', 'ball']
    else:
        objects_color = [obj.split(' ')[0] for obj in objs]
        object_type = [obj.split(' ')[1] for obj in objs] 
    
    command = command or random.choice(ALL_COMMANDS)
    if target_idx is None: target_idx = random.choice([0, 1])
    # drone_targets = [{target_idx: command}]

    max_yaw_offset = 0.1 * np.pi
    ortho_dist = 0.2
    z_offset = 0.5 if env_name == 'arena' else 0.0

    theta_offset = random.uniform(0, max_yaw_offset)
    theta_environment = random.uniform(-np.pi, np.pi - max_yaw_offset)

    pybullet_rand_forward1 = random.uniform(1.5, 2.5)
    pybullet_rand_forward2 = random.uniform(1.5, 2.5)
    drone_zrand = random.uniform(0.5, 0.8) + z_offset
    # Convert to GS
    gs_rand_x1 = pybullet_rand_forward1 * PYBULLET_TO_GS_SCALING_FACTOR
    gs_rand_x2 = pybullet_rand_forward2 * PYBULLET_TO_GS_SCALING_FACTOR

    gs_offsets_from_camera = np.array([
        [0, 0, 0], 
        [gs_rand_x1, random.uniform(-2.5, -1.5) * ortho_dist, random.uniform(-0.4, 0.4) + drone_zrand],
        [gs_rand_x2, random.uniform(1.5, 2.5) * ortho_dist, random.uniform(-0.4, 0.4) + drone_zrand]
    ], dtype=np.float32)  # forward, right, up
    
    # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
    objects_xyz = (gs_offsets_from_camera[1:, :] / PYBULLET_TO_GS_SCALING_FACTOR) * np.array([1, -1, 1])
    drones_loc = [[0.0, 0.0, drone_zrand]]

    if progress_scale is not None and progress_scale > 0:
        #TODO maybe need to revisit or maybe its already good
        px = progress_scale
        py = 1 - (1 - px) ** (5/3)
        pz = py if command in {'above', 'below'} else 1 - (1 - px) ** (20/3)
        delta = get_task_delta(command)
        drones_loc[0][0] = (objects_xyz[target_idx, 0] + delta[0]) * progress_scale
        drones_loc[0][1] = random.uniform(-0.04, 0.04) #(objects_xy[target_idx][1] + delta[1]) * py
        drones_loc[0][2] = drones_loc[0][2] + (objects_xyz[target_idx, 2] - drones_loc[0][2]) * pz

        drone_pos = np.array(drones_loc[0][:2])
        target_pos = objects_xyz[target_idx, :2] - delta[:2] #* progress_scale
        relative_vector = np.array(target_pos) - drone_pos
        theta_offset = np.arctan2(relative_vector[1], relative_vector[0])

    _schema = SimEnvInit_1DroneSchema()
    init_conditions = {
        "drones_loc": drones_loc,
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_loc": objects_xyz.tolist(),
        "objects_color": objects_color,
        "objects_type": object_type,
        # "drones_targets": drone_targets,
        'target_idx': target_idx,
        'command': command,
        "progress_scale": progress_scale,
        "env_name": env_name,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
        "gs_objects_relative": np.array(gs_offsets_from_camera)[1:, 0:2].tolist()
    }
    if log_path is not None: init_conditions['log_dir'] = log_path
    if data_path is not None: init_conditions['data_dir'] = data_path
    init_conditions = _schema.load(init_conditions)

    return init_conditions