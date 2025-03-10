import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .mcmd_utils import SimConfig, SimUtils

i2str = lambda j : '' if j == 0 else f'_{j}'

class SimLogger:
    def __init__(self):
        self.sim_timesteps = []
        self.objs = None
        self.global_pos_array = [[] for _ in range(SimConfig.NUM_DRONES)]
        self.vel_array = [[] for _ in range(SimConfig.NUM_DRONES)]
        self.timestepwise_displacement_array = [[] for _ in range(SimConfig.NUM_DRONES)]
        self.vel_cmds = [[] for _ in range(SimConfig.NUM_DRONES)]

    def log_objs(self, objs):
        self.objs = []
        for obj in objs:
            self.objs.append(self.parse_obj(obj))

    def log_timestep(self, timestep):
        self.sim_timesteps.append(timestep)

    def log_obs(self, obs):
        for i in range(SimConfig.NUM_DRONES):
            state = obs[str(i)]['state']
            self.global_pos_array[i].append(SimUtils.get_x_y_z_yaw_relative_to_base_env(state, SimConfig.theta_env))
            self.vel_array[i].append(SimUtils.get_vx_vy_vz_yawrate_rel_to_self(state))
            if len(self.global_pos_array[i]) > 1:
                rel_disp = SimUtils.get_relative_displacement(
                    self.global_pos_array[i][-1], 
                    self.global_pos_array[i][-2], 
                    -SimConfig.theta_env
                )
                self.timestepwise_displacement_array[i].append(rel_disp)

    def log_action(self, action):
        for i in range(SimConfig.NUM_DRONES):
            self.vel_cmds[i].append(action[str(i)])

    @staticmethod
    def parse_obj(obj):
        return {
            'x': obj.loc_rel[0],
            'y': obj.loc_rel[1],
            'z': obj.loc_rel[2],
            'style': f"{obj.colr[0]}{'s' if obj.obj_type=='cube' else 'o'}"
        }

    @staticmethod
    def plot_trajectory(pos_data, yaw_data, plotname, objs=None):        
        pos_id = {'x': 0, 'y':1, 'z':2}
        for i, sbplt in enumerate(['xy', 'yz', 'xz']):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(pos_data[:, pos_id[sbplt[0]]], pos_data[:, pos_id[sbplt[1]]], marker='>', alpha=0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(sbplt[0])
            ax.set_ylabel(sbplt[1])
            ax.set_title(f'{sbplt} Graph with Objects')
            if objs is not None:
                for obj in objs:
                    ax.plot(obj[sbplt[0]], obj[sbplt[1]], obj['style'])
                    circle = Circle((obj[sbplt[0]], obj[sbplt[1]]), 0.5, color='red', fill=False, linestyle='--')
                    ax.add_patch(circle)
            fig.savefig(SimConfig.log_path + f"/{plotname}_{sbplt}.jpg")
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(yaw_data)
        ax.set_ylabel('Yaw')
        fig.savefig(SimConfig.log_path + f"/{plotname}_yaw.jpg")

    def export_plots(self):
        for i in range(SimConfig.NUM_DRONES):
            sim_dir = SimConfig.log_path
            self.global_pos_array[i] = np.array(self.global_pos_array[i])
            self.timestepwise_displacement_array[i] = np.array(self.timestepwise_displacement_array[i])
            self.vel_array[i] = np.array(self.vel_array[i])

            self.plot_trajectory(
                self.global_pos_array[i],
                self.global_pos_array[i][:, 3],
                'sim_pos',
                self.objs
            )

            # x_data = self.global_pos_array[i][:, 0]
            # y_data = self.global_pos_array[i][:, 1]
            # yaw = self.global_pos_array[i][:, 3]
            # x_data_integrated = np.cumsum(self.timestepwise_displacement_array[i][:, 0])
            # y_data_integrated = np.cumsum(self.timestepwise_displacement_array[i][:, 1])
            # yaw_integrated = np.cumsum(self.timestepwise_displacement_array[i][:, 3])

            # # ! Path Plot
            # fig, axs = plt.subplots(2, 1)
            # axs[0].plot(x_data, y_data, alpha=0.5)
            # axs[0].plot(x_data_integrated, y_data_integrated, alpha=0.5)
            # axs[0].set_aspect('equal', adjustable='box')
            # for obj_loc in self.objs_loc:
            #     axs[0].plot(obj_loc[0], obj_loc[1], 'ro')
            # axs[0].legend(["Drone Position", "Disp Int", "Target"])
            
            # axs[1].plot(yaw)
            # axs[1].plot(yaw_integrated)
            # axs[1].set_ylabel('Yaw')
            # axs[1].legend(["Yaw", "Yaw Disp Int"])
            # fig.savefig(sim_dir + f"/sim_pos{i2str(i)}.jpg")

            # ! Velocity Plot
            fig3, axs3 = plt.subplots(2, 2)
            labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
            for idx, label in enumerate(labels):
                # print(i, idx, self.vel_array)
                axs3.flat[idx].plot(self.vel_array[i][:, idx])
                axs3.flat[idx].set_ylabel(label)
            fig3.savefig(sim_dir + f"/sim_velocity{i2str(i)}.jpg")

            np.savetxt(os.path.join(sim_dir, f'sim_pos{i2str(i)}.csv'), self.global_pos_array[i], delimiter=',')
            np.savetxt(os.path.join(sim_dir, f'sim_vel{i2str(i)}.csv'), self.vel_array[i], delimiter=',')
            np.savetxt(os.path.join(sim_dir, f'timestepwise_displacement{i2str(i)}.csv'), self.timestepwise_displacement_array[i], delimiter=',')
            
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

    