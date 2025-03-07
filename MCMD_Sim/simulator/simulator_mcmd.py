import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from ...gym_pybullet_drones.gym_pybullet_drones import CtrlAviary, DSLPIDControl, SimplePIDControl, Logger
from .enums import DroneModel, ImageType
from .schemas import InitConditionsSchema

from .simulator_mcmd_utils import SimUtils, SimConfig
from .simulator_mcmd_obj import SimObject, SimDrone

class InitConditionParser:
    def __init__(self, init_conditions):
        self.drones_height = init_conditions["start_heights"]
        self.objs_height = init_conditions["target_heights"]
        self.objs_color = init_conditions["objects_color"]
        self.objs_rel = init_conditions["objects_relative"]

        SimConfig.theta_offset = init_conditions["theta_offset"]
        SimConfig.theta_env = init_conditions["theta_environment"]


class MCMDSimulator: #Multiple Command Multiple Drones
    def __init__(self, sim_dir: str, init_conditions: InitConditionsSchema, record_hz: str, env_name='arena'):
        self.sim_dir = sim_dir
        self.record_freq_hz = record_hz
        init_cond = InitConditionParser(init_conditions)
        self.objs = [
            SimObject(loc_xy, SimConfig.theta_env, colr, height) 
            for loc_xy, colr, height in zip(init_cond.objs_rel, init_cond.objs_color, init_cond.objs_height)
        ]
        self.drones = [
            SimDrone(loc_xy, SimConfig.theta_env, SimConfig.theta_offset, height)
            for loc_xy, height in zip([(0,0)], init_cond.drones_height)
        ]
        SimConfig.NUM_DRONES = len(self.drones)
        self.drones_target = [{1:''}] # List of map of target indicies in self.objs to command
        assert len(self.drones_target) == SimConfig.NUM_DRONES, 'Atleast one drone does not have target defined'

        self.env_name = env_name
        self.custom_timesteps = []

        
#*---------- Trajectory Precomputation -------------------------
    def precompute_trajectory(self, turn_only=False, random_walk=False):
        for drone_idx, sim_drone in enumerate(self.drones):
            for i, (target_idx, command) in enumerate(self.drones_target[drone_idx].items()):
                sim_drone._setup_target(self.objs[target_idx], task=command)
                if i == 0: sim_drone._init_stable_trajectory()
                sim_drone._compute_trajectory(turn_only)

            sim_drone.traj_pos = np.array(sim_drone.traj_pos)
            sim_drone.traj_rpy = np.array(sim_drone.traj_rpy)

            if random_walk: sim_drone.add_noise_to_traj()
            sim_drone.has_precomputed_traj = True
            print(f'Drone_{drone_idx} trajectory computed for target_{target_idx}')

#-----------------Simulation Helper Functions-----------------------------------
    def export_snap(self, drone_index):
        rgb, dep, seg = self.env._getDroneImages(drone_index)
        self.env._exportImage(
            img_type=ImageType.RGB,
            img_input=rgb,
            path=self.sim_dir + f"/pybullet_pics{drone_index}",
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

#-----------------Simulation Main Code (setup, step, run) ----------------------------------------------
    def setup_simulation(self, drone=None, objs_details=None):
        if drone is None: drone = SimConfig.DRONE_MODEL
        if objs_details is None:
            objs_details = {"colors": [], "locations": []}
            for obj in self.objs:
                objs_details['colors'].append(obj.colr)
                objs_details['locations'].append(obj.loc_abs)

        AGGR_PHY_STEPS = int(SimConfig.SIMULATION_FREQ_HZ / SimConfig.CONTROL_FREQ_HZ) if SimConfig.AGGREGATE else 1
        INIT_XYZS = np.array([d.traj_pos[0] for d in self.drones])
        INIT_RPYS = np.array([d.traj_rpy[0] for d in self.drones])
        self.env = CtrlAviary(
            drone_model=drone,
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
        self.env.IMG_RES = np.array([256, 144])
        self.env.reset()

        PIDControl = DSLPIDControl if drone in [DroneModel.CF2X, DroneModel.CF2P] else SimplePIDControl #[DroneModel.HB]
        self.ctrl = [PIDControl(drone_model=drone) for _ in range(SimConfig.NUM_DRONES)]

        #! Simulation Params
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / SimConfig.CONTROL_FREQ_HZ)) # 1
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(SimConfig.NUM_DRONES)}
        self.simulation_counter = 0

        num_active_drones = 0
        self.STEPS = []
        for i, sim_drone in enumerate(self.drones):
            if sim_drone.has_precomputed_traj:
                num_active_drones += 1
                steps = int(self.CTRL_EVERY_N_STEPS * sim_drone.traj_pos.shape[0])
                self.STEPS.append(steps)
                os.makedirs(os.path.join(self.sim_dir, f"pybullet_pics{i}"), exist_ok=True)
                self.export_snap(i)

        assert len(self.drones) == num_active_drones, 'Atleast one drone does not have computed trajectory'
            
        self.global_pos_array = []
        self.vel_array = []
        self.timestepwise_displacement_array = []
        self.vel_cmds = []

        self.logger = Logger(
            logging_freq_hz=SimConfig.SIMULATION_FREQ_HZ,
            num_drones=SimConfig.NUM_DRONES,
            output_folder="/".join(self.sim_dir.split('/')[:-1]),
            colab=SimConfig.COLLAB,
        )

    def step_simulation(self, hz):
        num_steps_until_record = self.set_num_steps_until_record(hz)

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
                        self.logger.log(
                            drone=d,
                            timestamp=round(self.simulation_counter),
                            state=obs[str(d)]["state"],
                            control=np.hstack([
                                sim_drone.traj_pos[self.simulation_counter, 0:2], 
                                sim_drone.traj_pos[0, 2], 
                                sim_drone.traj_rpy[0], 
                                np.zeros(6)]
                            )
                        )
                        step_complete[d] = True
                self.custom_timesteps.append(self.simulation_counter *  1.0 / SimConfig.SIMULATION_FREQ_HZ)

            self.simulation_counter += 1
            for i, step in enumerate(self.STEPS):
                if self.simulation_counter >= step: step_complete[i] = True

            step_counter += 1
        return state
    
    def run_simulation_to_completion(self):
        """ Creates training images with the stored trajectory """
        self.setup_simulation()

        while not self.check_exausted_steps():
            state = self.step_simulation(self.record_freq_hz)
            self.global_pos_array.append(SimUtils.get_x_y_z_yaw_relative_to_base_env(state, SimConfig.theta_env))
            self.vel_array.append(SimUtils.get_vx_vy_vz_yawrate_rel_to_self(state))

            if len(self.global_pos_array) > 1:
                rel_disp = SimUtils.get_relative_displacement(self.global_pos_array[-1], self.global_pos_array[-2], -SimConfig.theta_env)
                self.timestepwise_displacement_array.append(rel_disp)
        self.env.close()
    
#---------------------Plotting Code----------------------------------------------
    def export_plots(self):
        self.global_pos_array = np.array(self.global_pos_array)
        self.timestepwise_displacement_array = np.array(self.timestepwise_displacement_array)
        self.vel_array = np.array(self.vel_array)

        x_data = self.global_pos_array[:, 0]
        y_data = self.global_pos_array[:, 1]
        yaw = self.global_pos_array[:, 3]
        x_data_integrated = np.cumsum(self.timestepwise_displacement_array[:, 0])
        y_data_integrated = np.cumsum(self.timestepwise_displacement_array[:, 1])
        yaw_integrated = np.cumsum(self.timestepwise_displacement_array[:, 3])

        # ! Path Plot
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(x_data, y_data, alpha=0.5)
        axs[0].plot(x_data_integrated, y_data_integrated, alpha=0.5)
        axs[0].set_aspect('equal', adjustable='box')
        for i, obj in enumerate(self.objs):
            axs[0].plot(obj.loc_rel[0], obj.loc_rel[1], 'ro')
        axs[0].legend(["Drone Position", "Disp Int", "Target"])
        
        axs[1].plot(yaw)
        axs[1].plot(yaw_integrated)
        axs[1].set_ylabel('Yaw')
        axs[1].legend(["Yaw", "Yaw Disp Int"])
        fig.savefig(self.sim_dir + "/sim_pos.jpg")

        # ! Velocity Plot
        fig3, axs3 = plt.subplots(2, 2)
        labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
        for idx, label in enumerate(labels):
            axs3.flat[idx].plot(self.vel_array[:, idx])
            axs3.flat[idx].set_ylabel(label)
        fig3.savefig(self.sim_dir + "/sim_velocity.jpg")

        np.savetxt(os.path.join(self.sim_dir, 'sim_pos.csv'), np.array(self.global_pos_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'sim_vel.csv'), np.array(self.vel_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'timestepwise_displacement.csv'), np.array(self.timestepwise_displacement_array), delimiter=',')
        
        # ! timestepwise_displacement as a 2x2 subplot
        # timestepwise_displacement_data = np.array(self.timestepwise_displacement_array)
        # fig2, axs2 = plt.subplots(2, 2)
        # labels = ['X Displacement', 'Y Displacement', 'Z Displacement', 'Yaw Displacement']
        # for idx, label in enumerate(labels):
        #     axs2.flat[idx].plot(timestepwise_displacement_data[:, idx])
        #     axs2.flat[idx].set_ylabel(label)
        # fig2.savefig(self.sim_dir + "/timestepwise_displacement.jpg")

        # self.plot_grid(
        #     data_list=[
        #         [{'X Velocity': self.vel_array[:, 0]}, {'Y Velocity': self.vel_array[:, 1]}],
        #         [{'Z Velocity': self.vel_array[:, 2]}, {'Yaw Velocity': self.vel_array[:, 3]}]
        #     ],
        #     filename='sim_velocity'
        # )
        # sim_pos_row0 = {'Sim Position':(x_data, y_data), 'Disp Int':(x_data_integrated, y_data_integrated)}
        # for i, obj in enumerate(self.objs):
        #     sim_pos_row0[f'target{i}'] = (obj.loc_rel[0], obj.loc_rel[1], f'{['r', 'b', 'g'][max(2, i)]}o')
        # self.plot_grid(
        #     data_list=[
        #         [sim_pos_row0],
        #         [{'Yaw':[yaw], 'Yaw Disp Int':[yaw_integrated]}]
        #     ],
        #     filename='sim_pos'
        # )
