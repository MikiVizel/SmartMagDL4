"""CAtNIPP environment driven by online SmartMagDL magnetic observations."""
import copy
from itertools import product
import numpy as np
from classes import PRMController
from gp_ipp import GaussianProcessForIPP
from magnetic_simulator import MagneticScene, MagneticConfig
from parameters import ADAPTIVE_AREA


class Env:
    def __init__(self, sample_size=200, k_size=20, start=None, destination=None,
                 obstacle=None, budget_range=(6,8), save_image=False, seed=None,
                 magnetic_config=None):
        self.sample_size, self.k_size = sample_size, k_size
        self.seed = seed
        self.config = magnetic_config or MagneticConfig.from_environment()
        self.rng = np.random.default_rng(seed)
        self.budget_range = budget_range
        self.budget0 = self.budget = float(self.rng.uniform(*budget_range))
        self.start = np.asarray(start if start is not None else self.rng.random(2)).reshape(1,2)
        self.destination = np.asarray(destination if destination is not None else self.rng.random(2)).reshape(1,2)
        self.obstacle = [] if obstacle is None else obstacle
        if self.obstacle:
            raise ValueError('This integration supports the obstacle-free SmartMagDL square only')
        if not 2 <= k_size <= sample_size+2:
            raise ValueError('Require 2 <= k_size <= sample_size+2')
        for coord in (self.start,self.destination):
            if np.any(coord<0) or np.any(coord>1):
                raise ValueError('start and destination must be in [0,1]^2')
        if np.allclose(self.start,self.destination):
            raise ValueError('Use distinct start and destination coordinates')
        state = np.random.get_state()
        try:
            for self.roadmap_attempt in range(1, 33):
                self.prm = PRMController(sample_size, self.obstacle, self.start,
                                         self.destination, budget_range, k_size)
                self.node_coords, self.graph = self.prm.runPRM(
                    seed=int(self.rng.integers(2**31)))
                shortest = self.prm.dijkstra_dist[1]['0']
                if np.isfinite(shortest):
                    break
            else:
                raise ValueError('No directed path from start to goal in 32 roadmaps. '
                                 'Increase k_size to improve connectivity.')
        finally:
            np.random.set_state(state)
        if shortest > self.budget0 + 1e-10:
            raise ValueError(f'Shortest graph route is {shortest * self.config.domain_m:.3f} m; '
                             f'budget is {self.budget0 * self.config.domain_m:.3f} m.')
        self.save_image=save_image
        self.frame_files=[]
        self.current_node_index=1
        self.route=[]

    def reset(self, seed=None):
        episode_seed = seed if seed is not None else self.seed
        self.underlying_distribution = MagneticScene(episode_seed,self.config)
        self.ground_truth = self.get_ground_truth()
        from smartmag_backend import make_belief
        self.gp_ipp = make_belief(self.node_coords, self.config)
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        self.node_info,self.node_std = self.gp_ipp.update_node()
        self.node_info0,self.node_std0 = self.node_info.copy(),self.node_std.copy()
        self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        self.cov_trace0 = self.cov_trace
        self.MI = self.MI0 = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        self.episode_return = 0.0
        self.budget = self.budget0
        self.current_node_index=1
        self.sample=self.start[0].copy()
        self.dist_residual=0.
        self.route=[]
        self.frame_files=[]
        self._plot_grid_size = None
        if getattr(self, 'capture_directory', None):
            self.plot([1], self.capture_episode, 0, self.capture_directory)
        return self.node_coords,self.graph,self.node_info,self.node_std,self.budget

    def _sample_segment(self,start,end,sample_length,measurement):
        delta = end-start
        dist=float(np.linalg.norm(delta))
        if dist<=1e-14:
            return
        heading=float(np.arctan2(delta[1],delta[0]))
        if sample_length is None or sample_length == 0:
            samples=np.asarray([end])
            self.dist_residual=0.
        else:
            if sample_length < 0 or not np.isfinite(sample_length):
                raise ValueError('sample_length must be finite and nonnegative')
            first=sample_length-self.dist_residual
            distances=np.arange(first,dist+1e-12,sample_length)
            distances=np.minimum(distances,dist)
            samples=start+distances[:,None]*delta/dist
            self.dist_residual = dist-distances[-1] if len(distances) else self.dist_residual+dist
        if len(samples):
            # Hypothetical rollouts must NEVER read the hidden magnetic field.
            values = self.underlying_distribution.observe(samples,heading) if measurement else np.zeros(len(samples))
            for pt,value in zip(samples,values):
                self.gp_ipp.add_observed_point(pt,value,measurement=measurement)
            self.sample=samples[-1].copy()

    def step(self,next_node_index,sample_length,measurement=True):
        if next_node_index not in range(len(self.node_coords)):
            raise ValueError('Invalid node')
        if str(next_node_index) not in self.graph[str(self.current_node_index)]:
            raise ValueError('Action must follow a PRM edge')
        start,end=self.node_coords[[self.current_node_index,next_node_index]]
        dist=float(np.linalg.norm(end-start))
        if dist>self.budget+1e-9:
            raise ValueError('Action exceeds remaining budget')
        self._sample_segment(start,end,sample_length,measurement)
        self.gp_ipp.update_gp()
        self.node_info,self.node_std=self.gp_ipp.update_node()
        if measurement:
            self.high_info_area=self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
            self.RMSE=self.gp_ipp.evaluate_RMSE(self.ground_truth)
        trace=self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        reward=0.
        if next_node_index in self.route[-2:]:
            reward=-.1
        elif self.cov_trace>trace:
            reward=(self.cov_trace-trace)/max(self.cov_trace,1e-12)
        done=next_node_index==0
        if done:
            reward-=trace/len(self.ground_truth)
        self.episode_return += float(reward)
        self.cov_trace=trace
        self.budget=max(0.,self.budget-dist)
        self.current_node_index=next_node_index
        self.route.append(next_node_index)
        if measurement and getattr(self, 'capture_directory', None):
            if done or len(self.route) % self.capture_every == 0:
                self.plot([1] + self.route, self.capture_episode, len(self.route), self.capture_directory)
        return reward,done,self.node_info,self.node_std,self.budget

    def route_step(self,route,sample_length,measurement=True):
        # Same mutating protocol as upstream; callers deepcopy for lookahead.
        for start,end in zip(route[:-1],route[1:]):
            start,end=np.asarray(start),np.asarray(end)
            dist=float(np.linalg.norm(end-start))
            if measurement and dist>self.budget+1e-9:
                raise ValueError('Route exceeds budget')
            self._sample_segment(start,end,sample_length,measurement)
            if measurement:
                self.budget=max(0.,self.budget-dist)
        self.gp_ipp.update_gp()
        self.node_info,self.node_std=self.gp_ipp.update_node()
        if measurement:
            self.high_info_area=self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
            self.RMSE=self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace=self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        return self.cov_trace

    def get_ground_truth(self):
        return self.underlying_distribution.distribution_function(GaussianProcessForIPP.grid())

    def plot(self,route,n,step,path,testID=0,CMAES_route=False,sampling_path=False):
        from visualization import plot_episode
        return plot_episode(self, route, n, step, path, test_id=testID, coordinates=CMAES_route)

    def enable_capture(self, directory, episode, every=1):
        self.capture_directory = str(directory)
        self.capture_episode = episode
        self.capture_every = every
