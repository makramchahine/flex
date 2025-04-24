from .schemas import SimEnvInit_1DroneSchema
from .schemas import ALL_COLORS, ALL_OBJ_TYPE, ALL_COMMANDS
import random
import numpy as np
from .tasks import get_task_delta
from typing import List, Tuple

PYBULLET_TO_GS_SCALING_FACTOR = 1.0

def get_grid_cell_centers(n: int, xyz_range, xyz_div, eps=0.1) -> List[Tuple[float, float, float]]:
    idx = np.random.choice(np.prod(xyz_div), size=n, replace=False)
    grid_size = (xyz_range[:, 1] - xyz_range[:, 0]) / xyz_div

    div_x, div_y, _ = xyz_div
    zi, rem = idx // (div_x * div_y), idx % (div_x * div_y)
    yi, xi = rem // div_x, rem % div_x

    centers = xyz_range[:, 0] + grid_size * (
        np.stack([xi, yi, zi], axis=1) + 0.5 + eps * (np.random.rand(n, 3) - 0.5)
    )
    return centers

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

def ICCLISchema2SimEnvSchema(
        init_cond, 
        env_name='samurai', 
        target_idx=None, 
        command=None, 
        log_path=None,
    ):
    """
    Convert InitConditionsClosedLoopInferenceSchema to SimEnvInit_1DroneSchema
    """
    _schema = SimEnvInit_1DroneSchema()
    try:
        init_conditions = _schema.load(init_cond)
    except:
        objects_color = []
        objects_type = []
        for obj_name in init_cond['objects_color']:
            obj = obj_name.split(' ')
            colr = '' if len(obj) == 1 else obj[0]
            objects_color.append(colr)
            objects_type.append(obj[-1])
        
        z_offset = 0.0 #if env_name == 'arena' else 0.0

        _schema = SimEnvInit_1DroneSchema()
        init_conditions = {
            "drones_loc": [(0, 0, init_cond['start_heights'][0])],
            "theta_offset": init_cond['theta_offset'],
            "theta_environment": init_cond['theta_environment'],
            "objects_loc": [(x + 0.4, y, h + z_offset) for (x, y), h in zip(init_cond['objects_relative'], init_cond['target_heights'])],
            "objects_color": objects_color,
            "objects_type": objects_type,
            'target_idx': target_idx,
            'command': command,
            "env_name": env_name,
            "PYBULLET_TO_GS_SCALING_FACTOR": init_cond['PYBULLET_TO_GS_SCALING_FACTOR'],
            "gs_objects_relative": init_cond['gs_objects_relative']
        }
        if log_path is not None: init_conditions['log_dir'] = log_path
        init_conditions = _schema.load(init_conditions)
    return init_conditions


def generate_closed_loop_1drone_nobject_env_init(
        objs,             # REQUIRED: list of strings like ["red ball", "blue ball", ...]
        env_name='arena',
        target_idx=None,
        command=None,
        log_path=None,
        x_range = (1.0, 3.0),
        y_range = (-0.8, 0.8),
        z_range = (-0.4, 0.8),
    ) -> SimEnvInit_1DroneSchema:
    """
    General initialization for 1-drone and N objects.
    Expects `objs` to be a list of "color type" strings like "red ball".
    """
    assert objs is not None and len(objs) > 0, "objs must be a non-empty list of 'color type' strings."

    n_objects = len(objs)
    objects_color, objects_type = [], []
    for obj_name in objs:
        obj = obj_name.split(' ')
        colr = '' if len(obj) == 1 else obj[0]
        objects_color.append(colr)
        objects_type.append(obj[-1])

    command = command or random.choice(ALL_COMMANDS)
    if target_idx is None:
        target_idx = random.randint(0, n_objects - 1)

    max_yaw_offset = 0 #0.2 * np.pi  # Setting 0 allows much more control on testing
    theta_offset = random.uniform(0, max_yaw_offset)
    theta_environment = random.uniform(-np.pi, np.pi - max_yaw_offset)
    # z_offset = 0.5 if env_name == 'arena' else 0.0

    drone_z = random.uniform(0.5, 1.2) #+ z_offset
    drones_loc = np.array([[0.0, 0.0, drone_z]])

    xyz_range = np.array([x_range, y_range, z_range])
    xyz_div = np.array([3, 3, 3])
    objects_xyz = get_grid_cell_centers(n_objects, xyz_range, xyz_div, eps=0.1)
    objects_xyz = np.array(objects_xyz) * np.array([1, -1, 1]) + drones_loc

    _schema = SimEnvInit_1DroneSchema()
    init_conditions = {
        "drones_loc": drones_loc,
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_loc": objects_xyz,
        "objects_color": objects_color,
        "objects_type": objects_type,
        "target_idx": target_idx,
        "command": command,
        "env_name": env_name,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
        "gs_objects_relative": objects_xyz
    }
    if log_path is not None: init_conditions['log_dir'] = log_path
    return _schema.load(init_conditions)



# def generate_closed_loop_1drone_objs_test_env_init(
#         objs : list[str],
#         gs_offset_coeff,
#         env_name='arena', 
#         command=None,
#         target_idx=None,
#         log_path=None,
#         PYBULLET_TO_GS_SCALING_FACTOR = 1.0
#     ) -> SimEnvInit_1DroneSchema:
#     """
#     #TODO
#     Specific implementation with weighted probabilities
#     """
#     num_obj = len(objs)
#     objects_color, object_type = [], []
#     pybullet_rand_forward = random.uniform(1.5, 2)
#     orthogonal_dist = 0.2
#     # Convert to GS
#     gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
#     gs_offsets_from_camera = [
#         [0, 0, 0], 
#         [gs_rand_forward, -1.5 * orthogonal_dist, random.uniform(0.6, 0.9)],
#         [gs_rand_forward, 1.5 * orthogonal_dist, random.uniform(0.5, 0.8)]
#     ]  # forward, right, up

#     for obj_name in objs:
#         obj = obj_name.split(' ')
#         colr = '' if len(obj) == 1 else obj[0]
#         objects_color.append(colr)
#         object_type.append(obj[-1])

    
#     gs_offsets_xy = np.array(gs_offsets_from_camera)[1:, 0:2]
#     gs_offsets_z = np.array(gs_offsets_from_camera)[1:, 2]

#     z_offset = 0.5 if env_name == 'arena' else 0.0
#     drones_loc = [[0, 0, random.uniform(0.5, 0.8) + z_offset]]
#     command = command or random.choice(ALL_COMMANDS)

#     if target_idx is None: target_idx = random.randomint(0, num_obj - 1)
#     # drone_targets = [{target_idx: command}]

#     max_yaw_offset = 0.1 * np.pi / (num_obj - 1)
#     theta_offset = random.uniform(0, max_yaw_offset)
#     theta_environment = random.uniform(-np.pi, np.pi - max_yaw_offset)

#     # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
#     objects_xy = gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR * np.array([1, -1])
#     objects_z = ((0.1 + 0.5 + gs_offsets_z + z_offset) / PYBULLET_TO_GS_SCALING_FACTOR).tolist()

#     _schema = SimEnvInit_1DroneSchema()
#     init_conditions = {
#         "drones_loc": drones_loc,
#         "theta_offset": theta_offset,
#         "theta_environment": theta_environment,
#         "objects_loc": [(x, y, h) for (x, y), h in zip(objects_xy, objects_z)],
#         "objects_color": objects_color,
#         "objects_type": object_type,
#         # "drones_targets": drone_targets,
#         'target_idx': target_idx,
#         'command': command,
#         "env_name": env_name,
#         "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
#         "gs_objects_relative": np.array(gs_offsets_from_camera)[1:, 0:2].tolist()
#     }
#     if log_path is not None: init_conditions['log_dir'] = log_path
#     init_conditions = _schema.load(init_conditions)

#     return init_conditions
