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

            self.num_rows_list = []
            self.num_cols_list = []
            self.row_spacing_list = []
            self.col_spacing_list = []
            self.traversability_hashmap_list = []
            self.width_list = []
            self.height_list = []
            self.origin_list = []

            self.device = None


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
    
    def set_traversability_hashmap(self, traversability_hashmap, map_size, spacing, origin):
        self.num_rows, self.num_cols = map_size
        self.row_spacing, self.col_spacing = spacing
        self.traversability_hashmap = traversability_hashmap
        self.width = self.num_rows * self.row_spacing
        self.height = self.num_cols * self.col_spacing
        self.origin = origin
        self.device = None


    def add_traversability_hashmap(self, i, traversability_hashmap, map_size, spacing, origin):

        # ugly, but at least the length of the list is reset everytime
        if i == 0:
            self.num_rows_list = []
            self.num_cols_list = []
            self.row_spacing_list = []
            self.col_spacing_list = []
            self.traversability_hashmap_list = []
            self.width_list = []
            self.height_list = []
            self.origin_list = []

        self.num_rows_list.append(map_size[0])
        self.num_cols_list.append(map_size[1])
        self.row_spacing_list.append(spacing[0])
        self.col_spacing_list.append(spacing[1])
        self.traversability_hashmap_list.append(traversability_hashmap)
        self.width_list.append(map_size[0]*spacing[0])
        self.height_list.append(map_size[1]*spacing[1])
        self.origin_list.append(origin)

    """
    Get traversability value of an x, y coordinate
    """
    def get_traversability(self, poses : torch.Tensor, map_lvl=0):
        if len(self.traversability_hashmap_list) == 0:
            return torch.ones(poses.shape[0], device=poses.device)

        if self.device is None:
            # self.traversability_hashmap = torch.tensor(self.traversability_hashmap, device=poses.device)
            self.traversability_hashmap_list = torch.tensor(self.traversability_hashmap_list, device=poses.device)

            self.device = poses.device
        
        origin = self.origin_list[map_lvl]
        traversability_hashmap = self.traversability_hashmap_list[map_lvl]

        xs = poses[:, 0] - origin[0], 
        ys = poses[:, 1] - origin[1],
        x_idx, y_idx = self.get_map_idx(xs, ys, map_lvl)

        return traversability_hashmap[y_idx, x_idx].clone().detach().to(dtype=torch.bool)    
    
    """
    Helper function to get the map id given x, y coordinates
    """
    def get_map_idx(self, x, y, map_lvl):
        x_idx = ((x[0] + self.width_list[map_lvl]/2 + self.row_spacing_list[map_lvl]/2) / self.row_spacing_list[map_lvl]).long()
        y_idx = ((y[0] + self.height_list[map_lvl]/2 + self.col_spacing_list[map_lvl]/2) / self.col_spacing_list[map_lvl]).long()
        x_idx = torch.clamp(x_idx, 0, self.num_rows_list[map_lvl]-1)
        y_idx = torch.clamp(y_idx, 0, self.num_cols_list[map_lvl]-1)
        
        return x_idx, y_idx
