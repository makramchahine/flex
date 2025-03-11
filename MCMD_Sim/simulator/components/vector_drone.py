import numpy as np
from .simobjects import SimDrone

class VectorizedSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False):
        """
        Vectorized version of trajectory generation for improved efficiency.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)

        P0 = np.array(self.traj_pos[-1])  
        Pd = np.array(self.destination)   
        Pt = np.array(self.target.loc_abs) 
        rc = self.critical_dist           

        n_points = 5000
        t_values = np.linspace(0, 1, n_points).reshape(-1, 1)  
        displacement_vector = Pd - P0
        points = P0 + t_values * displacement_vector 

        distances_to_target = np.linalg.norm(points - Pt, axis=1)
        inside_sphere_mask = distances_to_target < rc

        corrected_points = points.copy()
        correction_needed = corrected_points[inside_sphere_mask] - Pt
        correction_norms = np.linalg.norm(correction_needed, axis=1, keepdims=True)
        corrected_points[inside_sphere_mask] = Pt + rc * correction_needed / correction_norms

        delta_xy = Pt[:2] - corrected_points[:, :2]
        yaw_angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])

        self.traj_pos += corrected_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points

class SmoothCorrectedSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False):
        """
        Vectorized trajectory with smooth correction near the critical sphere.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)

        P0 = np.array(self.traj_pos[-1])  
        Pd = np.array(self.destination)   
        Pt = np.array(self.target.loc_abs) 
        rc = self.critical_dist           
        soft_zone = rc + 0.2               # Soft correction zone slightly beyond rc

        n_points = 5000
        t_values = np.linspace(0, 1, n_points).reshape(-1, 1)
        displacement_vector = Pd - P0
        points = P0 + t_values * displacement_vector  

        distances_to_target = np.linalg.norm(points - Pt, axis=1)
        inside_sphere_mask = distances_to_target < rc
        near_sphere_mask = (distances_to_target >= rc) & (distances_to_target < soft_zone)

        corrected_points = points.copy()

        correction_needed = corrected_points[inside_sphere_mask] - Pt
        correction_norms = np.linalg.norm(correction_needed, axis=1, keepdims=True)
        corrected_points[inside_sphere_mask] = Pt + rc * correction_needed / correction_norms

        blend_factor = (soft_zone - distances_to_target[near_sphere_mask]) / (soft_zone - rc)
        correction_vec = corrected_points[near_sphere_mask] - Pt
        correction_norms = np.linalg.norm(correction_vec, axis=1, keepdims=True)
        correction_vec /= correction_norms
        corrected_points[near_sphere_mask] = (
            corrected_points[near_sphere_mask] * (1 - blend_factor[:, None]) +
            (Pt + rc * correction_vec) * blend_factor[:, None]
        )

        delta_xy = Pt[:2] - corrected_points[:, :2]
        yaw_angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])

        self.traj_pos += corrected_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points

class BezierSmoothingSimDrone(SimDrone):
    def _compute_trajectory(self, turn_only=False):
        """
        Vectorized trajectory using Bézier curve smoothing.
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
        dist_target = np.linalg.norm(Pt - P0)
        direction /= dist_dest

        midpoint = (P0 + Pd) / 2
        projected = Pt + rc * (midpoint - Pt) / np.linalg.norm(midpoint - Pt)

        def bezier_curve(t):
            return (1 - t)**2 * P0 + 2 * (1 - t) * t * projected + t**2 * Pd

        n_points = 5000
        t_values = np.linspace(0, 1, n_points)
        t_values = 1 - np.exp(-1.5*t_values)
        
        bezier_points = np.array([bezier_curve(t) for t in t_values])
        delta_xy = Pt[:2] - bezier_points[:, :2]
        yaw_angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])

        self.traj_pos += bezier_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points


class NWLSimDrone(SimDrone): #Naming will be later
    def _compute_trajectory(self, turn_only=False):
        """
        Vectorized trajectory with smooth correction near the critical sphere.
        """
        assert hasattr(self, 'target'), 'Drone does not have any target'

        if turn_only: 
            return super()._compute_trajectory(turn_only)

        P0 = np.array(self.traj_pos[-1])  
        Pd = np.array(self.destination)   
        Pt = np.array(self.target.loc_abs) 
        rc = self.critical_dist           
        soft_zone = rc + 0.3

        n_points = 5000
        t_values = np.linspace(0, 1, n_points).reshape(-1, 1)
        t_values = 1 - np.exp(-1.5*t_values)

        displacement_vector = Pd - P0
        points = P0 + t_values * displacement_vector  

        # if len(self.traj_pos) > 500:
        #     prev_points = np.array(self.traj_pos[-50:])
        #     def bezier_blend(t, p0, p1, p2):
        #         return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            
        #     transition_points = np.array([bezier_blend(t, prev_points[-1], prev_points[-1] + 0.3 * (Pd - prev_points[-1]), Pd) for t in np.linspace(0, 1, 50)])
        #     points[:50] = transition_points

        distances_to_target = np.linalg.norm(points - Pt, axis=1)
        inside_sphere_mask = distances_to_target < rc
        near_sphere_mask = (distances_to_target >= rc) & (distances_to_target < soft_zone)

        corrected_points = points.copy()

        correction_needed = corrected_points[inside_sphere_mask] - Pt
        correction_norms = np.linalg.norm(correction_needed, axis=1, keepdims=True)
        corrected_points[inside_sphere_mask] = Pt + rc * correction_needed / correction_norms

        blend_factor = (soft_zone - distances_to_target[near_sphere_mask]) / (soft_zone - rc)
        correction_vec = corrected_points[near_sphere_mask] - Pt
        correction_norms = np.linalg.norm(correction_vec, axis=1, keepdims=True)
        correction_vec /= correction_norms
        corrected_points[near_sphere_mask] = (
            corrected_points[near_sphere_mask] * (1 - blend_factor[:, None]) +
            (Pt + rc * correction_vec) * blend_factor[:, None]
        )

        delta_xy = Pt[:2] - corrected_points[:, :2]
        yaw_angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])

        self.traj_pos += corrected_points.tolist()
        self.traj_rpy += [[0, 0, yaw] for yaw in yaw_angles]

        self.frame_counter += n_points