import torch
import matplotlib.pyplot as plt

"""
Storing the traversability hashmap on GPU significantly speeds up querying for traversability
"""
class TraversabilityHashmapUtil:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraversabilityHashmapUtil, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # Initialize your singleton class here
            self.traversability_hashmap = None
            self.num_plots = 0

    def plot_traversability_hashmap(self):
        if self.traversability_hashmap is None:
            raise ValueError("Traversability hashmap is not set.")
            
        plt.imshow(self.traversability_hashmap.cpu().numpy(), cmap='viridis', origin='lower')
        plt.colorbar(label='Traversability')
        plt.title('Traversability Hashmap')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.show()
    
    def plot_traversability_hashmap_xy(self, x, y):
        if self.traversability_hashmap is None:
            raise ValueError("Traversability hashmap is not set.")
        

        print("x and y:")
        print(x)
        print(y)
        plot_map_copy = self.traversability_hashmap.cpu().numpy().copy()
        for i in range(len(plot_map_copy)):
            for j in range(len(plot_map_copy[0])):
                if plot_map_copy[i][j] == 1:
                    plot_map_copy[i][j] = 0.5
    
        plot_map_copy[y, x] = 1
        plt.imshow(plot_map_copy, cmap='viridis', origin='lower')
        plt.colorbar(label='Traversability')
        plt.title('Traversability Hashmap')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.savefig(f"/home/yandabao/Desktop/IRL/wheeled_gym/visualizations/rgb/{self.num_plots}.png")
        plt.close()
        self.num_plots += 1
    
    def set_traversability_hashmap(self, traversability_hashmap, map_size, spacing):
        self.num_rows, self.num_cols = map_size
        self.row_spacing, self.col_spacing = spacing
        self.traversability_hashmap = traversability_hashmap
        self.width = self.num_rows * self.row_spacing
        self.height = self.num_cols * self.col_spacing
        self.device = None

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
