
import os
import numpy as np
from .traversability_utils import *
from .maps_utils import *

### hard coded for now

def create_maps_from_waypoints(maps_folder_path, map_name, num_envs, file_path, resolution):
    """
    Create a USD file and traversability hashmap from a PNG + YAML pair.
    
    Args:
        png_path: Path to the PNG image (black=obstacle, white=drivable).
        yaml_path: Path to the YAML file with metadata (resolution, origin).
        output_usd_path: Output USD file path (e.g., "/path/to/track.usd").
    """
    # Load YAML metadata
    map_path = os.path.join(maps_folder_path, map_name)
    yaml_file = map_name+'.yaml'
    png_file = map_name+'.png'
    yaml_path = os.path.join(map_path, yaml_file)
    png_path = os.path.join(map_path, png_file)
    waypoints_path = os.path.join(map_path, 'global_waypoints.json')

    #load waypoints coordinates
    waypoints, trackbounds, d_lat, psi_rad, kappa_radpm, vx_mps = load_waypoints(waypoints_path)
    
    # load yaml file info
    original_resolution, origin, free_thresh = load_yaml(yaml_path)
    
    # separate inner and outer bounds into two separate arrays
    outer, inner = separate_and_align_bounds(waypoints, trackbounds)

    # Create drivable map
    hashmap, map_size_meters, map_size_pixels, (x_min, x_max), (y_min, y_max), resolution = create_square_drivable_map(outer, inner, resolution)
    # env_origins = compute_origins(num_envs=num_envs, env_size=map_size_meters)

    # Set stage
    stage_name = file_path
    stage = set_stage_usd(stage_name)
    

    # Generate plane
    set_plane_usd(hashmap, map_size_pixels, map_size_meters, stage, x_min, x_max, y_min, y_max, resolution)
    # set_plane_usd_v2(hashmap, env_origins, map_size_pixels, map_size_meters, stage)

    # Convert waypoints to USD coordinates (adjust for origin and flip Y if needed)
    waypoints_usd = set_points_usd(waypoints, 'waypoints', origin, map_size_meters, stage, [(1.0, 0.0, 0.0)], x_min, y_min)

    # Project boundaries to centerline
    outer, inner = match_by_projection(waypoints, outer, inner)
    outer_usd = set_points_usd(outer, 'outer', origin, map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min)
    inner_usd = set_points_usd(inner, 'inner', origin, map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min)

    # Save stage
    stage.GetRootLayer().Save()

    # Return hashmap and metadata
    spacing_meters = (resolution, resolution)  # Equal spacing in x/y
    traversability_hashmap = hashmap.tolist()
    TraversabilityHashmapUtil().set_traversability_hashmap(
        traversability_hashmap, map_size_pixels, spacing_meters
    )
    return traversability_hashmap, waypoints_usd, outer_usd, inner_usd, d_lat.tolist(), psi_rad.tolist(), kappa_radpm.tolist(), vx_mps.tolist(), spacing_meters, map_size_pixels

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
    
    # Pre-compute waypoints tensor
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
    
    # Get the origins for the environments that need reset
    selected_origins = env_origins[env_ids]
    
    # Shift positions by their environment origins (only x and y)
    xs_shifted = xs + selected_origins[:, 0].cpu().numpy()
    ys_shifted = ys + selected_origins[:, 1].cpu().numpy()
    
    # Combine results
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

