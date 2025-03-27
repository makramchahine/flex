import numpy as np
from .simobjects import SimDrone


class SplineSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False, n_points = 8800):
        """
        Vectorized trajectory using Bezier curve smoothing.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)

        P0 = np.array(self.traj_pos[-1])  
        Pd = np.array(self.destination)   
        Pt = np.array(self.target.loc_abs) 
        rc = self.critical_dist + self.critical_dist_buffer
        rc += np.clip(np.random.randn() * 0.05, -0.05, 0.05)

        direction = Pd - P0
        dist_dest = np.linalg.norm(direction)
        direction /= dist_dest

        midpoint = (P0 + Pd) / 2
        projected = Pt + rc * (midpoint - Pt) / np.linalg.norm(midpoint - Pt)

        t_steps = np.linspace(0, 1, n_points)
        
        t_steps = np.column_stack([
            1 - (1 - t_steps)**2, #decayer((t_steps, 1.5)),
            1 - (1 - t_steps)**5, #decayer((t_steps, 1.5)),
            1 - (1 - t_steps)**12  #decayer((t_steps, 2.5))#np.clip(t_steps/t_steps[4000], 0, 1)
        ])

        bezier_points = (
            np.ones_like(t_steps) * P0 +
            2 * t_steps * (projected - P0) +
            t_steps**2 * (Pd - 2 * projected + P0)
        )
        
        # transition_weight = t_steps[:, 0] * 0.38
        # delta_target_xy = Pt[:2] - bezier_points[:, :2]
        # delta_dest_xy = Pd[:2] - bezier_points[:, :2]
        # target_yaw_angles = np.unwrap(np.arctan2(delta_target_xy[:, 1], delta_target_xy[:, 0]))
        # dest_yaw_angles = np.unwrap(np.arctan2(delta_dest_xy[:, 1], delta_dest_xy[:, 0]))
        # yaw_angles = (
        #     (1 - transition_weight) * target_yaw_angles +
        #     transition_weight * dest_yaw_angles
        # )

        trans_w = np.linspace(0, 1, n_points) ** 1.6
        dd = Pd[:2] - Pt[:2]
        if dd[1] != 0:
            # sign = -np.sign(dd[1])
            dist = np.linalg.norm(dd)
            # dd = dd / (np.linalg.norm(dd) + 1e-5)
            dd = (
                0.8 * dd + 
                2 * dd[::-1] * np.array([1, -1]) * np.sign(Pd[0]) * np.sign(dd[1])
            ) #* np.array([1, -1]) * sign
        # print("dd", dd, "dist", dist, "Pt", Pt, "Pd", Pd) 
        P_look = Pt[:2][None, :] + trans_w[:, None] * dd[None, :]
        delta_xy = P_look - bezier_points[:, :2]
        yaw_angles = np.unwrap(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]))

        self.traj_pos += bezier_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points