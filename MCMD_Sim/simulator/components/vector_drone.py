import numpy as np
from .simobjects import SimDrone

class BezierSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False, n_points = 8800):
        """
        Vectorized trajectory using Bezier curve smoothing.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)
        
        return self._compute_trajectory_planar_linear()
    
    def _compute_trajectory_free_bezier(self, n_points=8800):
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
        for i in range(3):
            projected[i] = np.clip(projected[i], min(P0[i], Pd[i]), max(P0[i], Pd[i]))

        t_steps = np.linspace(0, 1, n_points)
        t_steps = np.column_stack([
            1 - (1 - t_steps)**2,
            1 - (1 - t_steps)**5, 
            1 - (1 - t_steps)**12  
        ])

        bezier_points = (
            np.ones_like(t_steps) * P0 +
            2 * t_steps * (projected - P0) +
            t_steps**2 * (Pd - 2 * projected + P0)
        )

        delta = np.diff(bezier_points[:, :2], axis=0)
        yaw_angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        yaw_angles = np.append(yaw_angles, yaw_angles[-1])

        # trans_w = np.linspace(0, 1, n_points) ** 1.6
        # dd = Pd[:2] - Pt[:2]
        # if dd[1] != 0:
        #     dd = (
        #         0.8 * dd + 
        #         2 * dd[::-1] * np.array([1, -1]) * np.sign(Pd[0]) * np.sign(dd[1])
        #     )
        # P_look = Pt[:2][None, :] + trans_w[:, None] * dd[None, :]
        # delta_xy = P_look - bezier_points[:, :2]
        # yaw_angles = np.unwrap(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]))

        self.traj_pos += bezier_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points

    def _compute_trajectory_planar_linear(self, n_points=8800):
        P0 = np.array(self.traj_pos[-1])
        Pd = np.array(self.destination)
        Pt = np.array(self.target.loc_abs)
        
        # Compute plane normal
        N = np.cross(Pt - P0, Pd - P0)
        N /= np.linalg.norm(N)

        # Basis vectors
        u = (Pd - P0) / np.linalg.norm(Pd - P0)
        v = np.cross(N, u)
        
        # Generate linear trajectory in 2D
        t = np.linspace(0, 1, n_points)
        t = 1 - (1 - t)**5
        P_2d = np.column_stack((t, np.zeros_like(t)))
        
        # Map back to 3D
        P_3d = P0 + P_2d[:, 0:1] * u + P_2d[:, 1:2] * v

        # Compute yaw angles
        delta = np.diff(P_3d[:, :2], axis=0)
        yaw_angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        yaw_angles = np.append(yaw_angles, yaw_angles[-1])

        self.traj_pos += P_3d.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]
        self.frame_counter += n_points



