import os
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from .enums import DroneModel, Physics
import matplotlib.pyplot as plt

@dataclass
class SimConfig:
    #Environments Config
    NUM_DRONES = 1
    DRONE_MODEL = DroneModel("cf2x")
    PHYSICS = Physics("pyb")
    NEIGHBORHOOD_RADIUS = 10
    SIMULATION_FREQ_HZ = 240
    GUI = False
    RECORD_VISION = False
    OBSTACLES = True
    USER_DEBUG_GUI = False

    #Additional pyb settings
    CONTROL_FREQ_HZ = 240
    COLLAB = False
    AGGREGATE = True

    #Additional culetka constants
    APPROX_CORRECT_YAW = 0.0001
    DEFAULT_SEARCHING_YAW = 0.05
    DROP_SPEED = 0.1
    DROP_MAX_HEIGHT = 0.3
    DEFAULT_DROP_POINT_DIST = 0.05
    DEFAULT_CRITICAL_YAW_SPEED = (0.2 * np.pi * 2) / 3
    DEFAULT_LIFT_SPEED = 0.05

    #Conditional Environment Variable
    theta_offset = None
    theta_env = None
    log_path = None


class SimUtils:
    """Utility functions for drone navigation and coordinate conversions."""

    @staticmethod
    def signed_angular_distance(theta1: float, theta2: float) -> float:
        """Computes signed angular distance from theta1 to theta2."""
        return ((theta2 - theta1 + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def convert_to_global(rel_pos: Tuple[float, float], theta: float, rel_height=None) -> Tuple[float, float]:
        """Converts relative position to global coordinates based on rotation theta."""
        x = np.cos(theta) * rel_pos[0] - np.sin(theta) * rel_pos[1]
        y = np.sin(theta) * rel_pos[0] + np.cos(theta) * rel_pos[1]
        if len(rel_pos) > 2: rel_height = rel_pos[2]
        if rel_height is not None:
            return x, y, rel_height
        return x, y

    @staticmethod
    def distance_to_target(obj_xy: Tuple[float, float], target_xy: Tuple[float, float]) -> float:
        """Computes Euclidean distance between an object and a target."""
        return np.linalg.norm(np.array(obj_xy) - np.array(target_xy))

    @staticmethod
    def angle_between_two_points(source: Tuple[float, float], target: Tuple[float, float]) -> float:
        """Computes the angle between two points."""
        return np.arctan2(target[1] - source[1], target[0] - source[0])

    @staticmethod
    def convert_to_relative(global_pos, theta):
        return (np.cos(theta) * global_pos[0] + np.sin(theta) * global_pos[1], -np.sin(theta) * global_pos[0] + np.cos(theta) * global_pos[1])

    @staticmethod
    def get_x_y_z_yaw_relative_to_base_env(state, Theta):
        # Convert to relative drone coordinates by factoring in Theta
        relative_state = SimUtils.convert_to_relative([state[0], state[1]], Theta)
        return np.array([relative_state[0], relative_state[1], state[2], state[9]])
    
    @staticmethod
    def traj2xyz_relative_to_base_env(traj, theta):
        # Convert to relative drone coordinates by factoring in Theta
        xyz = np.zeros_like(traj)
        xyz[:, 0] = np.cos(theta) * traj[:, 0] + np.sin(theta) * traj[:, 1]
        xyz[:, 1] = np.cos(theta) * traj[:, 1] - np.sin(theta) * traj[:, 0]
        xyz[:, 2] = traj[:, 2]
        return xyz

    @staticmethod
    def get_vx_vy_vz_yawrate_rel_to_self(state):
        global_vx = state[10]
        global_vy = state[11]
        yaw = state[9]

        # convert from world_frame to body_frame
        rel_vx = global_vx * np.cos(yaw) + global_vy * np.sin(yaw)
        rel_vy = -global_vx * np.sin(yaw) + global_vy * np.cos(yaw)

        return np.array([rel_vx, rel_vy, state[12], state[15]])

    @staticmethod
    def get_relative_displacement(later_state, earlier_state, environment_theta):
        relative_xy = SimUtils.convert_to_relative([later_state[0] - earlier_state[0], later_state[1] - earlier_state[1]], earlier_state[3] + environment_theta)
        return np.array([relative_xy[0], relative_xy[1], later_state[2] - earlier_state[2], SimUtils.signed_angular_distance(earlier_state[3], later_state[3])])

# def plot_grid(data_list, savepath):
#         n, m = len(data_list), len(data_list[0])
#         fig_size = (min(10, 6*n), min(10, 6*m))
#         fig, axes = plt.subplots(n, m, figsize = fig_size)
#         for i, data_row in enumerate(data_list):
#             for j, data in enumerate(data_row):
#                 ax = axes[i][j]
#                 ax.plot(data)

#         fig.savefig(savepath)

