import random
import numpy as np
from collections import deque

from .simulator_mcmd_utils import SimUtils, SimConfig

CRITICAL_DIST = 0.5
CRITICAL_DIST_BUFFER = 0.1
TARGET_NUM_TIMESTEPS_TO_CRITICAL = (55, 90) # this affects the rate of drone control recovery
CONTROL_STEP_NORMALIZATION = 3
FINISH_COUNTER_THRESHOLD = 32
ATF = 64 * SimConfig.SIMULATION_FREQ_HZ / CONTROL_STEP_NORMALIZATION #Approx Total Frame

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

        self.I = {'x':random.uniform(0.025, 0.075), 'yaw':0.075, 'z':0.075, 'y':0.075}
        self.P = {
            'x': 0.18, #random.uniform(0.18, 0.2),
            'yaw': 0.2, #0.1 for 1 dist
            'z': 0.2,
            'y':0.2
        }
        self.checkpoint_frame = -1
        self.critical_dist = CRITICAL_DIST
        self.critical_dist_dest = CRITICAL_DIST * 0.1
        self.critical_dist_buffer = CRITICAL_DIST_BUFFER
        
    def _setup_target(self, target_obj:SimObject, task=None):
        self.target = target_obj
        self.destination = (target_obj.loc_rel[0] + 1.0, target_obj.loc_rel[1] - 1.0, target_obj.loc_abs[2])
        print(target_obj.colr, self.destination, 'target locked')
        self.task = task
        self.init_theta = self.traj_rpy[-1][-1]
        loc_rel = SimUtils.convert_to_relative(self.traj_pos[-1][:2], self.theta)
        self.final_theta = SimUtils.angle_between_two_points(loc_rel[:2], target_obj.loc_rel[:2])
        self.critical_action = target_obj.colr

        dist_0_x = loc_rel[0] - self.destination[0]
        dist_0_yaw = 0 - SimConfig.theta_offset
        dist_0_z = self.traj_pos[-1][2] - self.destination[2]
        dist_0_y = loc_rel[1] - self.destination[1]
        num_frames = random.randint(TARGET_NUM_TIMESTEPS_TO_CRITICAL[0], TARGET_NUM_TIMESTEPS_TO_CRITICAL[1])
        num_control_steps = num_frames / CONTROL_STEP_NORMALIZATION * SimConfig.CONTROL_FREQ_HZ
        self.eta_per_control = {
            'x': dist_0_x / num_control_steps,
            'yaw': dist_0_yaw / num_control_steps,
            'z': dist_0_z / num_control_steps,
            'y': dist_0_y / num_control_steps
        }
        self.frame_counter = 0
        self.finish_counter = 0
        self.reached_critical = False
        self.dist_buffer = deque(maxlen=5)
        self.destination = SimUtils.convert_to_global(self.destination, target_obj.theta)

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
                # done = self.has_dist_converged()
                self.__step_trajectory()
                if self.reached_critical:
                    self.finish_counter += 1
                    done = self.finish_counter > FINISH_COUNTER_THRESHOLD * 30
                done = done or self.frame_counter > ATF

    def __step_trajectory(self, hold=False, turn_only=False, turn_mode=False):
        """
        Uses a parametric curve to determine the next step, ensuring the drone moves 
        smoothly towards self.destination while staying close to a critical sphere around self.target.
        """
        if hold or turn_only or turn_mode: 
            return self.__step_trajectory2(hold=hold, turn_only=turn_only, turn_mode=turn_mode)

        P0 = np.array(self.traj_pos[-1])
        lyaw = self.traj_rpy[-1][2] 
        yaw_dist = SimUtils.signed_angular_distance(lyaw, self.final_theta + SimConfig.theta_env)

        Pd = np.array(self.destination)
        Pt = np.array(self.target.loc_abs)
        rc = self.critical_dist  # Radius of sphere around target
        soft_rc = rc + np.clip(np.random.randn() * 0.05, -0.05, 0.05)

        direction = Pd - P0
        dist_dest = np.linalg.norm(direction) 
        direction /= dist_dest

        midpoint = (P0 + Pd) / 2  
        projected = Pt + soft_rc * (midpoint - Pt) / np.linalg.norm(midpoint - Pt)

        def bezier_curve(t):
            return (1 - t)**2 * P0 + 2 * (1 - t) * t * projected + t**2 * Pd

        t_step = dist_dest / (6000 - self.frame_counter)#0.0002
        next_pos = bezier_curve(t_step)

        new_theta = self.init_theta
        if dist_dest > self.critical_dist_dest and not self.reached_critical:
            yaw_speed = self._get_adj_speed(yaw_dist, 'yaw')
            new_theta = self.final_theta + SimConfig.theta_env if abs(yaw_dist) < SimConfig.APPROX_CORRECT_YAW else lyaw + yaw_speed
            if dist_dest - self.critical_dist_dest > self.critical_dist_buffer:
                self.final_theta = SimUtils.angle_between_two_points(self.traj_pos[-1][:2], self.target.loc_abs[:2]) - SimConfig.theta_env
        else:
            self.reached_critical = True
            yaw_speed = SimConfig.DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / SimConfig.CONTROL_FREQ_HZ
            new_theta = lyaw + yaw_speed

        self.traj_pos.append(next_pos.tolist())
        self.traj_rpy.append([0, 0, new_theta])

        self.frame_counter += 1
        return np.linalg.norm(Pd - next_pos) 
    
    def __step_trajectory2(self, hold=False, turn_only=False, turn_mode=False):
        """
        Modifies:
            self.traj_pos, self.traj_rpy, self.final_theta, self.reached_critical
        """
        speed = 0
        # Getting last position and yaw
        lx, ly, lz = self.traj_pos[-1]
        lr, lp, lyaw = self.traj_rpy[-1]

        dist_dest = SimUtils.distance_to_target((lx, ly, lz), self.destination)
        dist_target = SimUtils.distance_to_target((lx, ly, lz), self.target.loc_abs)
        self.dist_buffer.append(dist_dest)

        yaw_dist = SimUtils.signed_angular_distance(lyaw, self.final_theta + SimConfig.theta_env)
        theta_dest = SimUtils.angle_between_two_points((lx, ly), self.destination[:2])
        height_dist = self.destination[2] - lz
        dist_to_crit = dist_dest #- 0.49

        speed_x, speed_y, yaw_speed, lift_speed, new_theta = 0, 0, 0, 0, self.init_theta
        if not hold:
            if dist_dest > self.critical_dist_dest and not self.reached_critical:
                speed_x, yaw_speed, lift_speed = self._get_adj_speeds(dist_to_crit, yaw_dist, height_dist)
                speed_y = self._get_adj_speed(dist_dest * np.sin(theta_dest - lyaw), 'y')
                new_theta = self.final_theta + SimConfig.theta_env if abs(yaw_dist) < SimConfig.APPROX_CORRECT_YAW else lyaw + yaw_speed
                if dist_dest - self.critical_dist_dest > self.critical_dist_buffer:
                    self.final_theta = SimUtils.angle_between_two_points((lx, ly), self.target.loc_abs[:2]) - SimConfig.theta_env
            else:
                self.reached_critical = True
                yaw_speed = SimConfig.DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / SimConfig.CONTROL_FREQ_HZ
                if turn_mode: yaw_speed, lift_speed = self._compute_turn_effects((lx, ly), yaw_speed, lift_speed)
                new_theta = lyaw + yaw_speed

        if turn_only:
            speed_x, speed_y, lift_speed = 0, 0, 0

        delta_z = lift_speed
        if self.critical_action == 'G':
            if self.reached_critical:
                delta_z = -(lz * SimConfig.DROP_SPEED) / (SimConfig.CONTROL_FREQ_HZ * SimConfig.DROP_MAX_HEIGHT)
            self.reached_critical = self.reached_critical or dist_dest < SimConfig.DEFAULT_DROP_POINT_DIST
            new_height = max(lz + delta_z, self.target.loc_abs[2])
        else:
            self.reached_critical = self.reached_critical or dist_target < self.critical_dist
            new_height = (lz + delta_z) if lift_speed != 0 else self.destination[2]

        delta_pos = SimUtils.convert_to_global([speed_x, speed_y], new_theta)
        self.traj_pos.append([lx + delta_pos[0], ly + delta_pos[1], new_height])
        self.traj_rpy.append([0, 0, new_theta])

        self.frame_counter += 1
        return speed

    
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
