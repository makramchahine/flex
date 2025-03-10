import random
import numpy as np
from collections import deque

from .utils.mcmd_utils import SimUtils, SimConfig

CRITICAL_DIST = 0.5
CRITICAL_DIST_BUFFER = 0.1
TARGET_NUM_TIMESTEPS_TO_CRITICAL = (55, 90) # this affects the rate of drone control recovery
CONTROL_STEP_NORMALIZATION = 3
FINISH_COUNTER_THRESHOLD = 32
ATF = (64 + 8) * SimConfig.SIMULATION_FREQ_HZ / CONTROL_STEP_NORMALIZATION #Approx Total Frame

class SimObject:
    def __init__(self, loc_rel, theta, colr=None, height=None):
        if height is None:
            assert len(loc_rel) == 3, 'missing height information'
        else:
            loc_rel = (loc_rel[0], loc_rel[1], height)
        self.loc_rel = loc_rel
        self.loc_abs = SimUtils.convert_to_global(loc_rel, theta)
        self.colr = colr
        self.theta = theta

class SimDrone(SimObject):
    def __init__(self, loc_rel, theta, theta_offset, height, target_obj=None):
        super().__init__(loc_rel, theta, height=height)
        if target_obj is not None: self._setup_target(target_obj)
        self.traj_pos = [[*self.loc_abs]]
        self.traj_rpy = [[0, 0, theta + theta_offset]]
        self.has_precomputed_traj = False

        self.I = {'x':random.uniform(0.025, 0.075), 'yaw':0.075, 'z':0.075}
        self.P = {
            'x': 0.18, #random.uniform(0.18, 0.2),
            'yaw': 0.2, #0.1 for 1 dist
            'z': 0.2
        }
        self.checkpoint_frame = -1
        self.critical_dist = CRITICAL_DIST
        self.critical_dist_buffer = CRITICAL_DIST_BUFFER
        
    def _setup_target(self, target_obj:SimObject, task=None):
        self.target = target_obj
        self.task = task
        self.init_theta = self.traj_rpy[-1][-1]
        loc_rel = SimUtils.convert_to_relative(self.traj_pos[-1][:2], self.theta)
        self.final_theta = SimUtils.angle_between_two_points(loc_rel[:2], target_obj.loc_rel[:2])
        self.critical_action = target_obj.colr

        dist_0_x = loc_rel[0] - target_obj.loc_rel[0]
        dist_0_yaw = 0 - SimConfig.theta_offset
        dist_0_z = self.traj_pos[-1][2] - target_obj.loc_abs[2]
        num_frames = random.randint(TARGET_NUM_TIMESTEPS_TO_CRITICAL[0], TARGET_NUM_TIMESTEPS_TO_CRITICAL[1])
        num_control_steps = num_frames / CONTROL_STEP_NORMALIZATION * SimConfig.CONTROL_FREQ_HZ
        self.eta_per_control = {
            'x': (dist_0_x - 0.5) / num_control_steps,
            'yaw': dist_0_yaw / num_control_steps,
            'z': dist_0_z / num_control_steps,
        }
        self.frame_counter = 0
        self.finish_counter = 0
        self.reached_critical = False
        self.dist_buffer = deque(maxlen=5)

    def _init_stable_trajectory(self):
        """ Holds still for one second of simulation """
        assert hasattr(self, 'target'), 'Drone does not have any target'
        for _ in range(SimConfig.SIMULATION_FREQ_HZ):
            self.__step_trajectory(hold=True)

    def _compute_trajectory(self, turn_only):
        assert hasattr(self, 'target'), 'Drone does not have any target'
        if turn_only:
            for _ in range(240 * 8):
                self.__step_trajectory(turn_only=True)
        else:
            done = False
            while not done:
                done = self.has_dist_converged()
                self.__step_trajectory()
                if self.reached_critical:
                    self.finish_counter += 1
                    done = self.finish_counter > FINISH_COUNTER_THRESHOLD * 30
                done = done or self.frame_counter > ATF
    
    def _get_adj_speeds(self, dist, yaw_dist, height_dist):
        speed = self._get_adj_speed(dist, 'x')
        yaw_speed = self._get_adj_speed(yaw_dist, 'yaw')
        lift_speed = self._get_adj_speed(height_dist, 'z')
        return speed, yaw_speed, lift_speed
    
    def _get_adj_speed(self, dist, key):
        assert key in {'x', 'yaw', 'z', 'y'}, 'invalid key to get adj speed'
        speed = dist / SimConfig.CONTROL_FREQ_HZ * self.P[key]
        if abs(dist) > abs(self.eta_per_control[key] * self.I[key]):
            speed += self.eta_per_control[key] * self.I[key]
        return speed

    def __step_trajectory(self, hold=False, turn_only=False, turn_mode=False):
        """
        Modifies:
            self.traj_pos, self.traj_rpy, self.final_theta, self.reached_critical
        """
        speed = 0
        #Getting last position and yaw
        lx, ly, lz = self.traj_pos[-1]
        lr, lp, lyaw = self.traj_rpy[-1]

        dist = SimUtils.distance_to_target((lx, ly, lz), self.target.loc_abs)
        self.dist_buffer.append(dist)
        yaw_dist = SimUtils.signed_angular_distance(lyaw, self.final_theta + SimConfig.theta_env)
        height_dist = self.target.loc_abs[2] - lz
        dist_to_crit = dist - 0.49

        speed, yaw_speed, lift_speed, new_theta = 0, 0, 0, self.init_theta
        if not hold:
            if dist > self.critical_dist and not self.reached_critical:
                speed, yaw_speed, lift_speed = self._get_adj_speeds(dist_to_crit, yaw_dist, height_dist)
                new_theta = self.final_theta + SimConfig.theta_env if abs(yaw_dist) < SimConfig.APPROX_CORRECT_YAW else lyaw + yaw_speed
                if dist - self.critical_dist > self.critical_dist_buffer:
                    self.final_theta = SimUtils.angle_between_two_points((lx, ly), self.target.loc_abs[:2]) - SimConfig.theta_env
            else:
                if self.checkpoint_frame == -1: self.checkpoint_frame = len(self.traj_pos)
                yaw_speed = SimConfig.DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / SimConfig.CONTROL_FREQ_HZ
                if turn_mode: yaw_speed, lift_speed = self._compute_turn_effects((lx, ly), yaw_speed, lift_speed)
                new_theta = lyaw + yaw_speed

        # Reset xy speed and lift speed if turn_only
        if turn_only:
            speed = 0
            lift_speed = 0

        delta_z = lift_speed
        if self.critical_action == 'G':
            if self.reached_critical:
                delta_z = -(lz * SimConfig.DROP_SPEED) / (SimConfig.CONTROL_FREQ_HZ * SimConfig.DROP_MAX_HEIGHT)
            self.reached_critical = self.reached_critical or dist < SimConfig.DEFAULT_DROP_POINT_DIST
            new_height = max(lz + delta_z, self.target.loc_abs[2])
        else:
            self.reached_critical = self.reached_critical or dist < self.critical_dist
            new_height = (lz + delta_z) if (lift_speed != 0 or hold) else self.target.loc_abs[2]
    
        delta_pos = SimUtils.convert_to_global([speed, 0], new_theta)
        self.traj_pos.append([lx + delta_pos[0], ly + delta_pos[1], new_height])
        self.traj_rpy.append([0, 0, new_theta])

        self.frame_counter += 1
        return speed

    def _compute_turn_effects(self, last_xy, yaw_speed, lift_speed):
        if self.critical_action == 'R': 
            yaw_speed += SimConfig.DEFAULT_CRITICAL_YAW_SPEED / SimConfig.CONTROL_FREQ_HZ
        elif self.critical_action == 'B':
            yaw_speed -= SimConfig.DEFAULT_CRITICAL_YAW_SPEED / SimConfig.CONTROL_FREQ_HZ
        elif self.critical_action == 'G':
            lift_speed = SimConfig.DEFAULT_LIFT_SPEED / SimConfig.CONTROL_FREQ_HZ
            if not self.reached_critical:
                self.final_theta = SimUtils.angle_between_two_points(last_xy, self.target.loc_abs[:2]) - SimConfig.theta_env # continue to face target
        else:
            assert False, f"critical_action: {self.critical_action}"

        return yaw_speed, lift_speed
    
    def add_noise_to_traj(self):
        new_mean = 0 #random.uniform(-0.15, 0.15)
        xyz_noise_matrix = np.random.normal(0, 0.01, size=(3, self.checkpoint_frame))
        xyz_noise_matrix -= np.mean(xyz_noise_matrix, axis=1, keepdims=True)
        yaw_noise_matrix = np.random.normal(new_mean, 0.001, size=(1, self.checkpoint_frame))
        yaw_noise_matrix -= np.mean(yaw_noise_matrix, axis=1, keepdims=True)

        self.traj_pos[:self.checkpoint_frame, 0:3] += xyz_noise_matrix.T
        self.traj_rpy[:self.checkpoint_frame, 2] += yaw_noise_matrix[0].T

    def has_dist_converged(self, tol=1e-4):
        if len(self.dist_buffer) < self.dist_buffer.maxlen:
            return False

        diffs = np.abs(np.diff(self.dist_buffer))
        return np.all(diffs < tol)
