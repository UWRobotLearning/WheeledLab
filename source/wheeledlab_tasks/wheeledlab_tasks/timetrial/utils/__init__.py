
import os
import numpy as np
from .traversability_utils import *
from .maps_utils import *

### hard coded for now

def create_maps_from_waypoints(maps_folder_path, map_name_list, origin_list, stage_path, resolution):
    """
    Create a USD file and traversability hashmap from a PNG + YAML pair.
    
    Args:
        png_path: Path to the PNG image (black=obstacle, white=drivable).
        yaml_path: Path to the YAML file with metadata (resolution, origin).
        output_usd_path: Output USD file path (e.g., "/path/to/track.usd").
    """

    # Initialize all lists
    waypoints_list = []
    outer_list = []
    inner_list = []
    trackbounds_list = []
    
    d_lat_list = []
    psi_rad_list = []
    kappa_radpm_list = []
    vx_mps_list = []

    hashmap_list = []
    map_size_meters_list = []
    map_size_pixels_list = []
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    resolution_list = []
    spacing_meters_list = []

    # Set stage
    stage_name = stage_path
    stage = set_stage_usd(stage_name)
    # origin_list = [[0, 0], [12.5, 0], [0, 20]]
    
    min_common_square_size = 0

    for i, map_name in enumerate(map_name_list):
        map_path = os.path.join(maps_folder_path, map_name)
        waypoints_path = os.path.join(map_path, 'global_waypoints.json')
        waypoints, trackbounds, d_lat, psi_rad, kappa_radpm, vx_mps = load_waypoints(waypoints_path)
        outer, inner = separate_and_align_bounds(waypoints, trackbounds)
        outer, inner = match_by_projection(waypoints, outer, inner)
        min_map_size = get_min_map_size(outer, resolution)
        if min_map_size > min_common_square_size:
            min_common_square_size = min_map_size
        

    for i, map_name in enumerate(map_name_list):
        map_path = os.path.join(maps_folder_path, map_name)
        waypoints_path = os.path.join(map_path, 'global_waypoints.json')
        
        # Load waypoints coordinates and append to lists
        waypoints, trackbounds, d_lat, psi_rad, kappa_radpm, vx_mps = load_waypoints(waypoints_path)
        trackbounds_list.append(trackbounds)
        d_lat_list.append(d_lat.tolist())
        psi_rad_list.append(psi_rad.tolist())
        kappa_radpm_list.append(kappa_radpm.tolist())
        vx_mps_list.append(vx_mps.tolist())
        
        # Load yaml file info
        # original_resolution, origin_yaml, free_thresh = load_yaml(yaml_path)
        
        # Separate bounds and append
        outer, inner = separate_and_align_bounds(waypoints, trackbounds)
        outer, inner = match_by_projection(waypoints, outer, inner)

        # outer_list.append(outer)
        # inner_list.append(inner)

        # Create drivable map and append all results
        hashmap, map_size_meters, map_size_pixels, (x_min, x_max), (y_min, y_max), resolution = create_square_drivable_map_v2(min_common_square_size, outer, inner, resolution)
        hashmap_list.append(hashmap.tolist())
        map_size_meters_list.append(map_size_meters)
        map_size_pixels_list.append(map_size_pixels)
        x_min_list.append(x_min)
        x_max_list.append(x_max)
        y_min_list.append(y_min)
        y_max_list.append(y_max)
        resolution_list.append(resolution)
        spacing_meters = [resolution, resolution]
        spacing_meters_list.append(spacing_meters)
        set_hashmap_usd(map_name, hashmap, origin_list[i], map_size_pixels, map_size_meters, stage, x_min, x_max, y_min, y_max, resolution)

        waypoints_usd = set_points_usd(waypoints, map_name, 'waypoints', origin_list[i], map_size_meters, stage, [(1.0, 0.0, 0.0)], x_min, y_min)
        waypoints = [[p[0], p[1], p[2]] for p in waypoints_usd]
        outer_usd = set_points_usd(outer, map_name, 'outer', origin_list[i], map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min)
        outer = [[p[0], p[1], p[2]] for p in outer_usd]
        inner_usd = set_points_usd(inner, map_name, 'inner', origin_list[i], map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min)
        inner = [[p[0], p[1], p[2]] for p in inner_usd]

        waypoints_list.append(waypoints)
        outer_list.append(outer)
        inner_list.append(inner)

        TraversabilityHashmapUtil().add_traversability_hashmap(i, hashmap.tolist(), map_size_pixels, (resolution, resolution), origin_list[i])

    # Save stage
    stage.GetRootLayer().Save()

    return hashmap_list, waypoints_list, outer_list, inner_list, d_lat_list, psi_rad_list, kappa_radpm_list, vx_mps_list, spacing_meters_list, map_size_pixels_list


def generate_random_poses_from_list(env_ids, num_poses, map_levels, env_origins, map_origin_list, row_spacing_list, col_spacing_list, traversability_hashmap_list, waypoints_usd_list, outer_usd_list, inner_usd_list, margin=0.1):
    """
    Generate random poses with vectorized operations, supporting multiple hashmaps based on map_level.
    Only generates poses for environments specified in env_ids.
    """
    # Convert inputs to numpy/torch as needed
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)
    
    # Get map levels only for the requested environments
    requested_map_levels = map_levels_np[env_ids_np]
    
    # Initialize output containers
    all_xs_shifted = np.zeros(len(env_ids))
    all_ys_shifted = np.zeros(len(env_ids))
    current_wps_idx = np.zeros(len(env_ids))
    all_angles = np.zeros(len(env_ids))
    
    # Process each unique map_level separately among the requested environments
    unique_map_levels = np.unique(requested_map_levels)
    
    for map_level in unique_map_levels:
        # Get indices (within env_ids) of environments with this map_level
        env_mask = (requested_map_levels == map_level)
        current_env_ids = env_ids_np[env_mask]
        
        # Skip if no environments use this map_level (shouldn't happen due to unique)
        if len(current_env_ids) == 0:
            continue
            
        # Get the hashmap and parameters for this map_level
        traversability_array = np.array(traversability_hashmap_list[map_level])
        H, W = traversability_array.shape
        
        # Get all valid positions in one operation
        valid_y, valid_x = traversability_array.nonzero()
        
        # Sample indices without replacement (one per environment)
        idxs = np.random.choice(len(valid_x), size=len(current_env_ids), replace=len(current_env_ids) > len(valid_x))
        
        # Convert all positions at once (vectorized)
        xs = (valid_x[idxs] - W // 2) * row_spacing_list[map_level]
        ys = (valid_y[idxs] - H // 2) * col_spacing_list[map_level]
        
        # Pre-compute waypoints tensor shifted by the origins
        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32) 
        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)

        # Vectorized angle computation
        positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1) + torch.tensor(map_origin_list[map_level][:2])
        current_indices, _ = find_nearest_waypoint(inner_xy, positions)
        current_wps_idx[env_mask] = current_indices

        # Compute angles for all positions at once
        lookahead = 5
        next_indices = (current_indices + lookahead) % len(inner_xy)
        
        current_wps = inner_xy[current_indices] 
        next_wps = inner_xy[next_indices]
        
        deltas = next_wps - current_wps
        angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15, 15, size=len(current_env_ids))
        
        # Shift the coordinates according to env_origins for the reset
        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy() + np.array(map_origin_list)[map_level, 0]
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy() + np.array(map_origin_list)[map_level, 1]
        
        # Store results in the output arrays at the correct positions
        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles.numpy() if torch.is_tensor(angles) else angles
    
    # Combine results while maintaining original order
    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    
    return poses, current_wps_idx

def generate_random_poses(env_origins, env_ids, num_poses, row_spacing, col_spacing, traversability_hashmap, waypoints_usd, outer_usd, inner_usd, margin=0.1):
    """
    Generate random poses with vectorized operations.
    """
    # Convert to numpy array once
    traversability_array = np.array(traversability_hashmap)
    H, W = traversability_array.shape
    
    # Get all valid positions in one operation
    valid_y, valid_x = traversability_array.nonzero()
    
    # Sample indices without replacement
    idxs = np.random.choice(len(valid_x), size=num_poses, replace=False)
    
    # Convert all positions at once (vectorized)
    xs = (valid_x[idxs] - W // 2) * row_spacing
    ys = (valid_y[idxs] - H // 2) * col_spacing 
    
    # Pre-compute waypoints tensor shifted by the origins
    waypoints_xy = torch.tensor(waypoints_usd)[:, :2].to(torch.float32) 
    inner_xy = torch.tensor(inner_usd)[:, :2].to(torch.float32)

    # Vectorized angle computation
    positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)
    current_indices, _ = find_nearest_waypoint(inner_xy, positions)
    
    # Compute angles for all positions at once
    lookahead = 5
    next_indices = (current_indices + lookahead) % len(inner_xy)
    
    # we take the inner bound instead of the closest waypoint to avoid case when the centerline from the other side of the wall is closer
    current_wps = inner_xy[current_indices] 
    next_wps = inner_xy[next_indices]
    
    deltas = next_wps - current_wps
    angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15,15)
    
    # Now shift the coordinates according to env_origins for the reset
    xs_shifted = (valid_x[idxs] - W // 2) * row_spacing + env_origins[env_ids, 0].cpu().numpy()
    ys_shifted = (valid_y[idxs] - H // 2) * col_spacing + env_origins[env_ids, 1].cpu().numpy()
    poses = list(zip(xs_shifted.tolist(), ys_shifted.tolist(), angles.tolist()))

    # Combine results
    # poses = list(zip(xs.tolist(), ys.tolist(), angles.tolist()))
    
    return poses

# def find_nearest_waypoint(waypoints: torch.Tensor, positions: torch.Tensor):
#     """Optimized nearest waypoint finder."""
#     # Squared distances are faster to compute and preserve order
#     deltas = positions.unsqueeze(1) - waypoints.unsqueeze(0)
#     sq_dists = torch.sum(deltas**2, dim=2)
#     min_dist, closest_idx = torch.min(sq_dists, dim=1)
#     return closest_idx, torch.sqrt(min_dist)  # Only sqrt the min distances


def find_nearest_waypoint(waypoints: torch.Tensor,  # Shape: [M, 2] - M waypoints
                         positions: torch.Tensor, # [N,2] - N environements
                         ) -> tuple[int, torch.Tensor]:
    """
    Finds closest waypoints for all cars, handling circular track wrapping.
    If lookahead is None, searches all waypoints (accurate but slower).
    With lookahead, only checks next K waypoints from current closest (faster).
    """
    M = waypoints.shape[0]
    N = positions.shape[0]
    
    # First find rough closest without wrapping [N]
    diffs = positions.unsqueeze(1) - waypoints.unsqueeze(0)  # [N,M,2]
    dists = torch.norm(diffs, p=2, dim=2)  # [N,M]
    closest_idx = torch.argmin(dists, dim=1)  # [N]
    
    return closest_idx, dists[torch.arange(N), closest_idx]

if __name__ == "__main__":
    create_maps_from_png('test.usd', 100, 100, 0.3, 0.3, 0.3)

