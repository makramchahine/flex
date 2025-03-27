import os
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import random

from .pybullet_drones import CtrlAviary, DSLPIDControl, SimplePIDControl
from .utils.enums import DroneModel, ImageType
from .utils import SimEnvInitSchema, SimConfig, SimLogger, i2str, SimUtils
from .utils.tasks import generate_instruction, get_task_delta
from .components import SimObject, SimDrone

COLORS = {
    "White": [255, 255, 255, 255],
    "Red": [255, 0, 0, 255],
    "Blue": [0, 0, 255, 255],
    "Green": [0, 255, 0, 255],
    "Purple": [128, 0, 128, 255],
    "Black": [0, 0, 0, 255],
    "Yellow": [255, 255, 0, 255],
    "Brown": [139, 69, 19, 255],
    "Orange": [255, 165, 0, 255],
    "Pink": [255, 192, 203, 255],
}

class InitConditionParser:
    def __init__(self, init_conditions, prefix=''):
        command = init_conditions['command']
        target_idx = init_conditions['target_idx']
        self.objs_color = init_conditions["objects_color"]

        log_path = init_conditions.get('log_dir', '/home/alex/flex/MCMD_Sim/results/')
        data_path = init_conditions.get('data_dir', None)
        cur_dt = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        cur_fld = f'{prefix}{command}_{target_idx}{self.objs_color[target_idx]}_{cur_dt}'
        SimConfig.log_path = os.path.join(log_path, cur_fld)
        if data_path is not None:
            SimConfig.data_path = os.path.join(data_path, cur_fld)
        else:
            SimConfig.data_path = os.path.join(SimConfig.log_path, 'data')

        os.makedirs(SimConfig.log_path, exist_ok=True)
        os.makedirs(SimConfig.data_path, exist_ok=True)
        os.makedirs(os.path.join(SimConfig.log_path, 'pybullet_seg'), exist_ok=True)

        self.drone_targets = [{target_idx : command}] #init_conditions['drones_targets']
        self.objs_loc = init_conditions['objects_loc']
        self.objs_type = init_conditions['objects_type']
        self.drones_loc = init_conditions['drones_loc']
        self.env_name = init_conditions['env_name']

        SimConfig.theta_offset = init_conditions["theta_offset"]
        SimConfig.theta_env = init_conditions["theta_environment"]

        with open(os.path.join(SimConfig.log_path, 'init_conditions.json'), 'w') as f:
            json.dump(init_conditions, f)


class MCMDSimulator: #Multiple Command Multiple Drones
    def __init__(self, init_conditions: SimEnvInitSchema, record_hz: str, log_prefix=''):
        self.record_freq_hz = record_hz
        init_cond = InitConditionParser(init_conditions, prefix=log_prefix)
        self.objs = [
            SimObject(loc_xy, SimConfig.theta_env, colr, obj_type) 
            for loc_xy, colr, obj_type in zip(init_cond.objs_loc, init_cond.objs_color, init_cond.objs_type)
        ]
        self.drones = [
            SimDrone(i, loc_xy, SimConfig.theta_env, SimConfig.theta_offset)
            for i, loc_xy in enumerate(init_cond.drones_loc)
        ]
        SimConfig.NUM_DRONES = len(self.drones)
        self.drones_target = init_cond.drone_targets # List of map of target indicies in self.objs to command
        self.env_name = init_cond.env_name
        assert len(self.drones_target) == SimConfig.NUM_DRONES, 'Atleast one drone does not have target defined'

        self.logger = SimLogger()
        self.logger.log_objs(self.objs)
        self.seg_color_map = {}
        self.COLORS = {
            "White": [255, 255, 255, 255],
            "Red": [255, 0, 0, 255],
            "Blue": [0, 0, 255, 255],
            "Green": [0, 255, 0, 255],
            "Purple": [128, 0, 128, 255],
            "Black": [0, 0, 0, 255],
            "Yellow": [255, 255, 0, 255],
            "Brown": [139, 69, 19, 255],
            "Orange": [255, 165, 0, 255],
            "Pink": [255, 192, 203, 255],
        }

#-----------------Simulation Helper Functions-----------------------------------
    def export_snap(self, drone_index):
        rgb, dep, seg = self.env._getDroneImages(drone_index)
        if len(np.unique(seg)) <= 4: self.no_object_in_sight += 1

        cseg = np.zeros((seg.shape[0], seg.shape[1], 4))
        for val in np.unique(seg):
            mask = seg == val
            if val in self.seg_color_map:
                color_values = self.seg_color_map[val]
            else:
                color_name, color_values = random.choice(list(self.COLORS.items()))
                self.logger.log_text(f'{val}: {color_values} -- {color_name}')
                del self.COLORS[color_name]  # Remove the color from the dictionary
                self.seg_color_map[val] = color_values
            cseg[mask] = np.array(color_values)
        
        self.env._exportImage(
            img_type=ImageType.RGB,
            img_input=cseg,
            path=os.path.join(SimConfig.log_path, f'pybullet_seg{i2str(drone_index)}'),
            frame_num=int(self.simulation_counter),
        )
        self.env._exportImage(
            img_type=ImageType.RGB,
            img_input=rgb,
            path=SimConfig.data_path,#os.path.join(SimConfig.log_path, f'pybullet_pics{i2str(drone_index)}'),
            frame_num=int(self.simulation_counter),
        )
        return rgb

    def set_num_steps_until_record(self):
        hz = self.record_freq_hz
        if type(hz) == int:
            # 79 is 3Hz; 26 is 9Hz
            self.num_steps_until_record = round(SimConfig.CONTROL_FREQ_HZ / hz) - 1
        elif hz == "1-10":
            x = np.linspace(0, 216, 217)
            weights = 3 * norm.pdf(x, loc=21, scale=30) + norm.pdf(x, loc=217-21, scale=100)
            weights /= weights.sum()
            self.num_steps_until_record = np.random.choice(np.arange(24, 241), p=weights)
        else:
            raise ValueError(f"Incorrect hz value: {hz}")

#-----------------Simulation Setup Code  ----------------------------------------------
    def setup_simulation(self):
        objs_details = {"colors": [], "locations": []}
        for obj in self.objs:
            objs_details['colors'].append(f'{obj.colr} {obj.obj_type}')
            objs_details['locations'].append(obj.loc_abs)

        AGGR_PHY_STEPS = int(SimConfig.SIMULATION_FREQ_HZ / SimConfig.CONTROL_FREQ_HZ) if SimConfig.AGGREGATE else 1
        INIT_XYZS = np.array([d.traj_pos[0] for d in self.drones])
        INIT_RPYS = np.array([d.traj_rpy[0] for d in self.drones])
        self.env = CtrlAviary(
            drone_model=SimConfig.DRONE_MODEL,
            num_drones=SimConfig.NUM_DRONES,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=SimConfig.PHYSICS,
            neighbourhood_radius=SimConfig.NEIGHBORHOOD_RADIUS,
            freq=SimConfig.SIMULATION_FREQ_HZ,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=SimConfig.GUI,
            record=SimConfig.RECORD_VISION,
            obstacles=SimConfig.OBSTACLES,
            user_debug_gui=SimConfig.USER_DEBUG_GUI,
            custom_obj_location=objs_details,
            env_name=self.env_name
        )
        self.env.IMG_RES = np.array([224, 224])
        self.env.reset()
        self.simulation_counter = 0

        PIDControl = DSLPIDControl if SimConfig.DRONE_MODEL in [DroneModel.CF2X, DroneModel.CF2P] else SimplePIDControl #[DroneModel.HB]
        self.ctrl = [PIDControl(drone_model=SimConfig.DRONE_MODEL) for _ in range(SimConfig.NUM_DRONES)]
        #! Simulation Params
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / SimConfig.CONTROL_FREQ_HZ)) # 1
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(SimConfig.NUM_DRONES)}
        self.no_object_in_sight = -1

    
class MCMDSimSampler(MCMDSimulator):
    def __init__(self, init_conditions, record_hz):
        super().__init__(init_conditions, record_hz)

#*---------- Trajectory Precomputation ------------------------------------------
    def precompute_trajectory(self, turn_only=False, random_walk=False, export_traj=False):
        all_instructions = ''
        for drone_idx, sim_drone in enumerate(self.drones):
            for i, (target_idx, command) in enumerate(self.drones_target[drone_idx].items()):
                target = self.objs[target_idx]
                inst_text = generate_instruction(target.colr, command)
                all_instructions += f'{SimConfig.log_path}--{inst_text}\n'
                text_out = sim_drone._setup_target(target, task_delta=get_task_delta(command, target_idx))
                self.logger.log_text(f'{inst_text}\n{text_out}')
                print(text_out)
                # if i == 0: sim_drone._init_stable_trajectory()
                sim_drone._compute_trajectory(turn_only)

            sim_drone.traj_pos = np.array(sim_drone.traj_pos)
            sim_drone.traj_rpy = np.array(sim_drone.traj_rpy)

            if random_walk: sim_drone.add_noise_to_traj()
            sim_drone.has_precomputed_traj = True
            with open(os.path.join(SimConfig.data_path, 'label.txt'), 'w', encoding='utf-8') as f:
                f.write(inst_text)
            if export_traj:
                np.savetxt(
                    os.path.join(SimConfig.log_path, f'traj{i2str(drone_idx)}.csv'),
                    np.hstack([sim_drone.traj_pos, sim_drone.traj_rpy]),
                    delimiter=','
                )
                SimLogger.plot_trajectory(
                    SimUtils.traj2xyz_relative_to_base_env(sim_drone.traj_pos[::25], SimConfig.theta_env),
                    sim_drone.traj_rpy[:, 2][::25],
                    'traj',
                    [SimLogger.parse_obj(obj) for obj in self.objs]
                )
            print(f'Drone_{drone_idx} trajectory computed for target_{target_idx}')
        return all_instructions
    
#-----------------Simulation Helper Functions-----------------------------------
    def check_exausted_steps(self):
        return all([
            self.simulation_counter >= step
            for step in self.STEPS
        ]) or self.no_object_in_sight > 1
    
#-----------------Simulation Setup Code  ----------------------------------------------
    def setup_simulation(self):
        super().setup_simulation()
        num_active_drones = 0
        self.STEPS = []
        for i, sim_drone in enumerate(self.drones):
            if sim_drone.has_precomputed_traj:
                num_active_drones += 1
                steps = int(self.CTRL_EVERY_N_STEPS * sim_drone.traj_pos.shape[0])
                self.STEPS.append(steps)
                self.export_snap(i)

        assert len(self.drones) == num_active_drones, 'Atleast one drone does not have computed trajectory'

#----------------------------- Simulation Action and Control ------------------------------------
    def __step_simulation(self):
        self.set_num_steps_until_record()

        step_counter = 0
        step_complete = [False for _ in range(len(self.drones))]
        while not all(step_complete):#recorded_image == False and self.simulation_counter < self.STEPS:
            obs, reward, done, info = self.env.step(self.action)

            if self.simulation_counter % self.CTRL_EVERY_N_STEPS == 0:
                for j, sim_drone in enumerate(self.drones):
                    if not step_complete[j]:
                        state = obs[str(j)]["state"]
                        self.action[str(j)], _, _ = self.ctrl[j].computeControlFromState(
                            control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                            state=state,
                            target_pos=sim_drone.traj_pos[self.simulation_counter],
                            target_rpy=sim_drone.traj_rpy[self.simulation_counter]
                        )
                    else:
                        self.action[str(j)] = np.array([0,0,0,0])

            #* Network Frequency is 30hz
            if step_counter >= self.num_steps_until_record: # and self.simulation_counter>self.env.SIM_FREQ:
                for d, sim_drone in enumerate(self.drones):
                    if not step_complete[d]:
                        self.export_snap(d)
                        step_complete[d] = True
                self.logger.log_timestep(self.simulation_counter *  1.0 / SimConfig.SIMULATION_FREQ_HZ)

            self.simulation_counter += 1
            for i, step in enumerate(self.STEPS):
                if self.simulation_counter >= step: step_complete[i] = True

            step_counter += 1
        return obs

    #------------------------------Simulation All in one run code ---------------------------
    def run_simulation_to_completion(self, turn_only=False, random_walk=False):
        """ Creates training images with the stored trajectory """
        inst = self.precompute_trajectory(turn_only=turn_only, random_walk=random_walk, export_traj=True)
        self.setup_simulation()

        while not self.check_exausted_steps():
            obs = self.__step_simulation()
            self.logger.log_obs(obs)
            self.logger.log_action(self.action)
        self.env.close()

        self.logger.export_plots()
        self.logger.export_video()

        return inst
    

class MCMDSimEval(MCMDSimulator):
    def __init__(self, init_conditions, record_hz, log_prefix=''):
        super().__init__(init_conditions, record_hz, log_prefix)
        self.setup_simulation()
        self.drones = [drone.loc_rel for drone in self.drones]
        self.REC_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.record_freq_hz ))
        
#----------------------------- Simulation Action and Control ------------------------------------
    def step_action(self, vel_cmds):
        imgs = []
        updated_action = False
        vel_cmd_world = {}
        while (self.simulation_counter % self.REC_EVERY_N_STEPS) != 0 or not updated_action:
            self.logger.log_action(self.action)
            obs, reward, done, info = self.env.step(self.action)
            state = obs[str(0)]["state"]
            yaw = state[9]

            #* Network Frequency is 30hz
            if self.simulation_counter % self.REC_EVERY_N_STEPS== 0:
                for d in range(SimConfig.NUM_DRONES):
                    img = self.export_snap(d)
                    imgs.append(img[:, :, :3])
                    vel_cmd_world[d] = SimUtils.convert_vel_cmd_to_world_frame(vel_cmds[d], yaw)

                self.logger.log_obs(obs)
                updated_action = True
                
            if self.simulation_counter % self.CTRL_EVERY_N_STEPS == 0:
                for d in range(SimConfig.NUM_DRONES):
                    self.action[str(d)], _, _ = self.ctrl[d].computeControl(
                                                        control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3],  # same as the current position
                                                        target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                        target_vel=vel_cmd_world[d][0:3],
                                                        target_rpy_rates=np.array([0, 0, vel_cmds[d][3]])
                                                )
            self.simulation_counter += 1
        return imgs
    
    def close(self):
        self.env.close()
        self.logger.export_plots()
        self.logger.export_video()
        return SimConfig.log_path
