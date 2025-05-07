
    
import torch
import matplotlib.pyplot as plt

"""
Storing the traversability hashmap on GPU significantly speeds up querying for traversability
"""
class WaypointsUtil:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WaypointsUtil, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # Initialize your singleton class here
            self.waypoints = None
            self.num_plots = 0

    def waypoints(self):
        if self.waypoints is None:
            raise ValueError("waypoints hashmap is not set.")

    """
    Get traversability value of an x, y coordinate
    """
    def get_traversability(self, poses : torch.Tensor):
        if self.traversability_hashmap is None:
            return torch.ones(poses.shape[0], device=poses.device)

        if self.device is None:
            self.traversability_hashmap = torch.tensor(self.traversability_hashmap, device=poses.device)
            self.device = poses.device
        
        xs, ys = poses[:, 0], poses[:, 1]
        x_idx, y_idx = self.get_map_id(xs, ys)
        return self.traversability_hashmap[y_idx, x_idx]
    
    """
    Helper function to get the map id given x, y coordinates
    """
    def get_map_id(self, x, y):
        x_idx = ((x + self.width/2.0 + self.row_spacing/2.0) / self.row_spacing).long()
        y_idx = ((y + self.height/2 + self.col_spacing/2) / self.col_spacing).long()
        x_idx = torch.clamp(x_idx, 0, self.num_rows-1)
        y_idx = torch.clamp(y_idx, 0, self.num_cols-1)
        return x_idx, y_idx