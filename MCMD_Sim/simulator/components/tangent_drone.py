import random
import numpy as np

from ..utils import SimUtils, SimConfig
from .simobjects import SimDrone

class DirectStepSimDrone(SimDrone):
    def _step_trajectory(self, hold=False, turn_only=False, turn_mode=False):
        """
        Directly computes displacement toward the destination, corrects yaw, 
        and ensures the drone avoids the critical sphere around the target.
        """
        if hold or turn_only or turn_mode: 
            return super()._step_trajectory(hold=hold, turn_only=turn_only, turn_mode=turn_mode)

        P0 = np.array(self.traj_pos[-1])   # Current position
        lyaw = self.traj_rpy[-1][2]        # Last yaw angle

        Pd = np.array(self.destination)    # Destination
        Pt = np.array(self.target.loc_abs) # Target position
        rc = self.critical_dist            # Radius of the critical sphere

        direction = Pd - P0
        dist_dest = np.linalg.norm(direction) 
        direction /= dist_dest

        step_size = 5e-4
        next_pos = P0 + step_size * direction

        dist_target = np.linalg.norm(next_pos - Pt)
        if dist_target < rc:
            tangent_vec = (next_pos - Pt) / dist_target
            next_pos = Pt + rc * tangent_vec

        yaw_dist = SimUtils.signed_angular_distance(lyaw, self.final_theta + SimConfig.theta_env)
        new_theta = self.init_theta
        if dist_dest > 0:#self.critical_dist_dest and dist_target > self.critical_dist and not self.reached_critical:
            yaw_speed = self._get_adj_speed(yaw_dist, 'yaw')
            new_theta = self.final_theta + SimConfig.theta_env if abs(yaw_dist) < SimConfig.APPROX_CORRECT_YAW else lyaw + yaw_speed
            if dist_target - self.critical_dist_dest > 2 * self.critical_dist_buffer:
                self.final_theta = SimUtils.angle_between_two_points(self.traj_pos[-1][:2], self.target.loc_abs[:2]) - SimConfig.theta_env

        self.traj_pos.append(next_pos.tolist())
        self.traj_rpy.append([0, 0, new_theta])

        self.frame_counter += 1
        return np.linalg.norm(Pd - next_pos)
