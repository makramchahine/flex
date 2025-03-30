import numpy as np
from .simobjects import SimDrone
from scipy.interpolate import splprep, splev


class SplineSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False, n_points = 8800):
        """
        Vectorized trajectory using Sine curve smoothing.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)

        P0 = np.array(self.traj_pos[-1])  
        Pd = np.array(self.destination)   
        Pt = np.array(self.target.loc_abs)

        c_steps = np.linspace(0, 1, n_points)
        c_steps = np.column_stack([
            1 - (1 - c_steps)**3,
            1 - (1 - c_steps)**5,
            1 - (1 - c_steps)**20
        ])
        t_yaw = np.linspace(0, 1, n_points) ** 1.6

        c_dir = Pd - Pt
        d_dir = Pd - P0

        c_values = (0.2/np.linalg.norm(c_dir)) * np.sin(np.pi * c_steps) 
        trajectory = P0[None, :] + c_steps * d_dir[None, :] + c_values * c_dir

        c_ortho = np.array([c_dir[1], -c_dir[0], 0])
        sgn = np.sign(np.dot(c_ortho, d_dir))
        dd = 0.8 * c_dir + 2 * c_ortho * sgn

        delta_xy = Pt[None, :2] + t_yaw[:, None] * dd[None, :2] - trajectory[:, :2]
        # delta_xy = np.diff(trajectory[:, :2], axis=0)
        yaw_angles = np.unwrap(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]))
        # yaw_angles = np.append(yaw_angles, yaw_angles[-1])

        self.traj_pos += trajectory.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points