import random
import numpy as np

from ..utils import SimUtils, SimConfig
from .simobjects import SimDrone, ATF


class BezierSimDrone(SimDrone):
    def _step_trajectory(self, hold=False, turn_only=False, turn_mode=False):
        """
        Uses a parametric curve to determine the next step, ensuring the drone moves 
        smoothly towards self.destination while staying close to a critical sphere around self.target.
        """
        if hold or turn_only or turn_mode: 
            return super()._step_trajectory(hold=hold, turn_only=turn_only, turn_mode=turn_mode)

        P0 = np.array(self.traj_pos[-1])
        lyaw = self.traj_rpy[-1][2] 
        yaw_dist = SimUtils.signed_angular_distance(lyaw, self.final_theta + SimConfig.theta_env)

        Pd = np.array(self.destination)
        Pt = np.array(self.target.loc_abs)
        rc = self.critical_dist + self.critical_dist_buffer  # Radius of sphere around target
        soft_rc = rc + np.clip(np.random.randn() * 0.05, -0.05, 0.05)

        direction = Pd - P0
        dist_dest = np.linalg.norm(direction)
        dist_target = np.linalg.norm(Pt - P0)
        direction /= dist_dest

        midpoint = (P0 + Pd) / 2  
        projected = Pt + soft_rc * (midpoint - Pt) / np.linalg.norm(midpoint - Pt)

        t_step = dist_dest / (ATF + 800 - self.frame_counter)#0.0002
        next_pos = (1 - t_step)**2 * P0 + 2 * (1 - t_step) * t_step * projected + t_step**2 * Pd


        new_theta = self.init_theta
        if dist_dest > self.critical_dist_dest:# and dist_target > self.critical_dist and not self.reached_critical:
            yaw_speed = self._get_adj_speed(yaw_dist, 'yaw')
            new_theta = self.final_theta + SimConfig.theta_env if abs(yaw_dist) < SimConfig.APPROX_CORRECT_YAW else lyaw + yaw_speed
            if dist_target - self.critical_dist_dest > self.critical_dist_buffer:
                self.final_theta = SimUtils.angle_between_two_points(self.traj_pos[-1][:2], self.target.loc_abs[:2]) - SimConfig.theta_env
        else:
            self.reached_critical = True
            yaw_speed = SimConfig.DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / SimConfig.CONTROL_FREQ_HZ
            new_theta = lyaw + yaw_speed

        self.traj_pos.append(next_pos.tolist())
        self.traj_rpy.append([0, 0, new_theta])

        self.frame_counter += 1
        return np.linalg.norm(Pd - next_pos) 
