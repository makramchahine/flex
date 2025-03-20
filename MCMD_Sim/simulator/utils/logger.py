import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
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
        self.logs = ''

    def log_text(self, txt):
        self.logs += f'{txt}\n'

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
            ax.plot(pos_data[:, pos_id[sbplt[0]]], pos_data[:, pos_id[sbplt[1]]], marker='o', alpha=0.5, color='blue')
            if sbplt == 'xy':
                for i in range(0, yaw_data.shape[0], 5):
                    x, y = pos_data[i, pos_id['x']], pos_data[i, pos_id['y']]
                    ang = yaw_data[i] - SimConfig.theta_env + 2* np.pi
                    dx, dy = 0.13 * np.cos(ang), 0.13 * np.sin(ang)
                    ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, color='red')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(sbplt[0])
            ax.set_ylabel(sbplt[1])
            ax.set_title(f'{sbplt} Graph with Objects')
            if objs is not None:
                for obj in objs:
                    ax.plot(obj[sbplt[0]], obj[sbplt[1]], obj['style'])
                    circle = Circle((obj[sbplt[0]], obj[sbplt[1]]), 0.5, color='red', fill=False, linestyle='--')
                    ax.add_patch(circle)
            fig.savefig(SimConfig.log_path + f"/{plotname}_{sbplt}_{SimConfig.log_path[-13:]}.jpg")
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

            # ! Velocity Plot
            fig3, axs3 = plt.subplots(2, 2)
            labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
            for idx, label in enumerate(labels):
                axs3.flat[idx].plot(self.vel_array[i][:, idx])
                axs3.flat[idx].set_ylabel(label)
            fig3.savefig(sim_dir + f"/sim_velocity{i2str(i)}.jpg")

            np.savetxt(os.path.join(SimConfig.data_path, f'pos{i2str(i)}.csv'), self.global_pos_array[i], delimiter=',')
            np.savetxt(os.path.join(SimConfig.data_path, f'data_out{i2str(i)}.csv'), self.vel_array[i], delimiter=',')
            np.savetxt(os.path.join(sim_dir, f'timestepwise_displacement{i2str(i)}.csv'), self.timestepwise_displacement_array[i], delimiter=',')
            with open(os.path.join(SimConfig.log_path, 'logs.txt'), 'w', encoding='utf-8') as f:
                f.write(self.logs)
            
    def export_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(SimConfig.log_path, f'out_video_{SimConfig.log_path[-13:]}.mp4'), fourcc, 20, (224, 224))
        images = [f for f in os.listdir(SimConfig.data_path) if f.endswith('.png')]
        images.sort()
        for i, image in enumerate(images):
            if i < 8000:
                image = os.path.join(SimConfig.data_path, image)
                img = Image.open(image)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img)

        out.release()

    