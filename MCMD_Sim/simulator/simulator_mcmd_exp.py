import os
import numpy as np
from scipy.stats import norm
from datetime import datetime

from .pybullet_drones import CtrlAviary, DSLPIDControl, SimplePIDControl
from .utils.enums import DroneModel, ImageType
from .utils import SimEnvInitSchema, SimConfig, SimLogger, i2str, SimUtils
from .components import SimObject, SimDrone

class InitConditionParser:
    def __init__(self, init_conditions):
        log_path = init_conditions.get('sim_dir', '/home/alex/flex/MCMD_Sim/results/')
        SimConfig.log_path = os.path.join(log_path, datetime.now().strftime('%Y_%m_%d_%H_%M'))

        self.objs_color = init_conditions["objects_color"]
        if 'drones_targets' in init_conditions:
            self.drone_targets = init_conditions['drones_targets']
            self.objs_loc = init_conditions['objects_loc']
            self.objs_type = init_conditions['objects_type']
            self.drones_loc = init_conditions['drones_loc']
            self.env_name = init_conditions['env_name']
        else:
            self.drones_target = [{0:'to'}] 
            #TODO
            drones_height = init_conditions["start_heights"]
            objs_height = init_conditions["target_heights"]
            objs_rel = init_conditions["objects_relative"]

        SimConfig.theta_offset = init_conditions["theta_offset"]
        SimConfig.theta_env = init_conditions["theta_environment"]


class MCMDSimulator: #Multiple Command Multiple Drones
    def __init__(self, init_conditions: SimEnvInitSchema, record_hz: str):
        self.record_freq_hz = record_hz
        init_cond = InitConditionParser(init_conditions)
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
        
#*---------- Trajectory Precomputation -------------------------
    def precompute_trajectory(self, turn_only=False, random_walk=False, export_traj=False):
        for drone_idx, sim_drone in enumerate(self.drones):
            for i, (target_idx, command) in enumerate(self.drones_target[drone_idx].items()):
                sim_drone._setup_target(self.objs[target_idx], task=command)
                if i == 0: sim_drone._init_stable_trajectory()
                sim_drone._compute_trajectory(turn_only)

            sim_drone.traj_pos = np.array(sim_drone.traj_pos)
            sim_drone.traj_rpy = np.array(sim_drone.traj_rpy)

            if random_walk: sim_drone.add_noise_to_traj()
            sim_drone.has_precomputed_traj = True
            if export_traj:
                np.savetxt(
                    os.path.join(SimConfig.log_path, f'traj{i2str(drone_idx)}.csv'),
                    np.hstack([sim_drone.traj_pos, sim_drone.traj_rpy]),
                    delimiter=','
                )
                SimLogger.plot_trajectory(
                    SimUtils.traj2xyz_relative_to_base_env(sim_drone.traj_pos[::10], SimConfig.theta_env),
                    sim_drone.traj_rpy[:, 2],
                    'traj',
                    [SimLogger.parse_obj(obj) for obj in self.objs]
                )
            print(f'Drone_{drone_idx} trajectory computed for target_{target_idx}')

#-----------------Simulation Helper Functions-----------------------------------
    def export_snap(self, drone_index):
        rgb, dep, seg = self.env._getDroneImages(drone_index)
        self.env._exportImage(
            img_type=ImageType.RGB,
            img_input=rgb,
            path=os.path.join(SimConfig.log_path, f'pybullet_pics{i2str(drone_index)}'),
            frame_num=int(self.simulation_counter),
        )
    def set_num_steps_until_record(self, hz):
        if type(hz) == int:
            # 79 is 3Hz; 26 is 9Hz
            num_steps_until_record = round(SimConfig.CONTROL_FREQ_HZ / hz) - 1
        elif hz == "1-10":
            x = np.linspace(0, 216, 217)
            weights = 3 * norm.pdf(x, loc=21, scale=30) + norm.pdf(x, loc=217-21, scale=100)
            weights /= weights.sum()
            num_steps_until_record = np.random.choice(np.arange(24, 241), p=weights)
        else:
            raise ValueError(f"Incorrect hz value: {hz}")
        return num_steps_until_record
    
    def check_exausted_steps(self):
        return all([
            self.simulation_counter >= step
            for step in self.STEPS
        ])

#-----------------Simulation Setup Code  ----------------------------------------------
    def setup_simulation(self, mode='collect', env_name='arena'):
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
            env_name=env_name
        )
        self.env.IMG_RES = np.array([256, 144])
        self.env.reset()
        self.simulation_counter = 0
        self.logger = SimLogger()
        self.logger.log_objs(self.objs)
        self.mode = mode
        if mode == 'collect': 
            self.__setup_control()
        else:
            self.drones = [drone.loc_rel for drone in self.drones]

    def __setup_control(self):
        PIDControl = DSLPIDControl if SimConfig.DRONE_MODEL in [DroneModel.CF2X, DroneModel.CF2P] else SimplePIDControl #[DroneModel.HB]
        self.ctrl = [PIDControl(drone_model=SimConfig.DRONE_MODEL) for _ in range(SimConfig.NUM_DRONES)]
        #! Simulation Params
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / SimConfig.CONTROL_FREQ_HZ)) # 1
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(SimConfig.NUM_DRONES)}

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
    def step_action(self, action):
        get_action = lambda i : action[min(i, len(action)-1)]
        actions = {str(i): get_action(i) for i in range(SimConfig.NUM_DRONES)}
        self.logger.log_action(actions)
        obs, reward, done, info = self.env.step(actions)
        if self.simulation_counter % self.env.SIM_FREQ == 0:
            for d in range(len(self.drones)):
                self.export_snap(d)
                state = obs[str(d)]["state"]
            self.logger.log_obs(obs)
        self.simulation_counter += 1

    def __step_simulation(self):
        num_steps_until_record = self.set_num_steps_until_record(self.record_freq_hz)

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
            if step_counter >= num_steps_until_record and self.simulation_counter>self.env.SIM_FREQ:
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
    def run_simulation_to_completion(self, env_name='arena', turn_only=False, random_walk=False):
        """ Creates training images with the stored trajectory """
        # assert self.mode == 'collect', 'Simulator is not setup with collect mode'
        self.precompute_trajectory(turn_only=turn_only, random_walk=random_walk, export_traj=True)
        self.setup_simulation(mode='collect', env_name=env_name)

        while not self.check_exausted_steps():
            obs = self.__step_simulation()
            self.logger.log_obs(obs)
            self.logger.log_action(self.action)
        self.env.close()

        self.logger.export_plots()
