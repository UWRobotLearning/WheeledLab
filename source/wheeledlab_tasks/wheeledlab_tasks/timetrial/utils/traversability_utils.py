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



    def get_traversability(self, poses: torch.Tensor, map_levels: torch.Tensor):
        """
        Get traversability values for multiple agents, each potentially on different maps.
        
        Args:
            poses: Tensor of shape [num_agents, 2] containing (x,y) positions
            map_levels: Tensor of shape [num_agents] containing map level for each agent
            
        Returns:
            Tensor of shape [num_agents] with traversability values
        """
        if len(self.traversability_hashmap_list) == 0:
            return torch.ones(poses.shape[0], device=poses.device)

        # Convert to tensor if needed
        if isinstance(map_levels, int):
            map_levels = torch.full((poses.shape[0],), map_levels, device=poses.device)
        
        if self.device is None:
            self.traversability_hashmap_list = [torch.tensor(m, device=poses.device) for m in self.traversability_hashmap_list]
            self.device = poses.device

        # Initialize output
        traversability = torch.zeros(poses.shape[0], device=poses.device, dtype=torch.bool)
        
        # Process each unique map level
        unique_map_levels = torch.unique(map_levels)
        for map_level in unique_map_levels:
            mask = (map_levels == map_level)
            if not mask.any():
                continue
                
            # Get map-specific parameters
            origin = self.origin_list[map_level]
            traversability_hashmap = self.traversability_hashmap_list[map_level]
            map_level_poses = poses[mask]
            
            # Calculate coordinates
            xs = map_level_poses[:, 0] - origin[0]
            ys = map_level_poses[:, 1] - origin[1]
            
            # Get indices
            x_idx, y_idx = self.get_map_idx(xs, ys, map_level)
            
            # Store results
            traversability[mask] = traversability_hashmap[y_idx, x_idx].clone().detach().to(dtype=torch.bool)
        
        return traversability

    def get_map_idx(self, x: torch.Tensor, y: torch.Tensor, map_level: int):
        """
        Get map indices for coordinates, handling multiple map levels.
        
        Args:
            x: Tensor of x coordinates
            y: Tensor of y coordinates
            map_levels: Integer map level (all coordinates must be from same map)
            
        Returns:
            Tuple of (x_idx, y_idx) tensors
        """
        # Calculate indices
        x_idx = ((x + self.width_list[map_level]/2 + self.row_spacing_list[map_level]/2) / 
                self.row_spacing_list[map_level]).long()
        y_idx = ((y + self.height_list[map_level]/2 + self.col_spacing_list[map_level]/2) / 
                self.col_spacing_list[map_level]).long()
        
        # Clamp to valid range
        x_idx = torch.clamp(x_idx, 0, self.num_rows_list[map_level]-1)
        y_idx = torch.clamp(y_idx, 0, self.num_cols_list[map_level]-1)
        
        return x_idx, y_idx
