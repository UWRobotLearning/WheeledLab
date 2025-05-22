
import os
import numpy as np
from scipy.ndimage import binary_dilation
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from matplotlib.path import Path
import yaml
from PIL import Image
import json
from .traversability_utils import *
from .waypoints_utils import *
def create_maps_from_png(maps_folder_path, map_name, file_path, visualization_scale):
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
    resolution = 0.1  # meters per pixel
    hashmap, map_size_meters, map_size_pixels, (x_min, x_max), (y_min, y_max), resolution = create_square_drivable_map(outer, inner, resolution)
    
    # Set stage
    stage = set_stage_usd(file_path)
    
    # Generate plane
    set_plane_usd(hashmap, map_size_pixels, map_size_meters, stage, x_min, x_max, y_min, y_max, resolution)
    
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


def separate_bounds(points):
    """Separate outer and inner boundaries based on maximum distance between points."""
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    split_idx = np.argmax(distances) + 1  # Index where bounds switch
    outer_bound = points[:split_idx]
    inner_bound = points[split_idx:]
    return outer_bound, inner_bound

def create_square_drivable_map(outer, inner, resolution=0.5):
    """Create square boolean hashmap of drivable areas."""
    # Create grid that fully contains outer boundary
    x_min, y_min = np.min(outer, axis=0)
    x_max, y_max = np.max(outer, axis=0)
    
    # Add small buffer to ensure outer boundary is fully contained
    buffer = resolution * 2
    x_min -= buffer
    y_min -= buffer
    x_max += buffer
    y_max += buffer
    
    # Calculate initial dimensions
    length_meters = x_max - x_min
    height_meters = y_max - y_min
    
    # Make dimensions square by expanding the smaller dimension
    max_dim = max(length_meters, height_meters)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Adjust bounds to create square
    x_min = center_x - max_dim/2
    x_max = center_x + max_dim/2
    y_min = center_y - max_dim/2
    y_max = center_y + max_dim/2
    
    # Calculate square dimensions in meters
    map_size_meters = (max_dim, max_dim)  # (height, length) - now equal
    
    # Calculate grid dimensions in pixels (now square)
    n_pixels = int(np.ceil(max_dim / resolution)) + 1
    map_size_pixels = (n_pixels, n_pixels)  # (n_rows, n_cols) - now equal
    
    # Create grid coordinates
    x = np.linspace(x_min, x_max, n_pixels)
    y = np.linspace(y_min, y_max, n_pixels)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Create paths for polygon checks
    outer_path = Path(outer)
    inner_path = Path(inner)
    
    # Check each grid point
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    in_outer = outer_path.contains_points(points)
    in_inner = inner_path.contains_points(points)
    
    # Drivable = inside outer AND outside inner
    drivable = (in_outer & ~in_inner).reshape(grid_x.shape)
    
    return (drivable, 
            map_size_meters, 
            map_size_pixels, 
            (x_min, x_max), 
            (y_min, y_max), 
            resolution)


# def create_drivable_map(outer, inner, resolution=0.5):
#     """Create boolean hashmap of drivable areas."""
#     # Create grid that fully contains outer boundary
#     x_min, y_min = np.min(outer, axis=0)
#     x_max, y_max = np.max(outer, axis=0)
    
#     # Add small buffer to ensure outer boundary is fully contained
#     buffer = resolution * 2
#     x_min -= buffer
#     y_min -= buffer
#     x_max += buffer
#     y_max += buffer
    
#     # Calculate map dimensions in meters
#     length_meters = x_max - x_min
#     height_meters = y_max - y_min
#     map_size_meters = (height_meters, length_meters)  # (height, length)
#     # Calculate grid dimensions in pixels
#     n_cols = int(np.ceil(length_meters / resolution)) + 1
#     n_rows = int(np.ceil(height_meters / resolution)) + 1
#     map_size_pixels = (n_rows, n_cols)  # (n_rows, n_cols)
#     # Create grid coordinates
#     x = np.arange(x_min, x_max, resolution)
#     y = np.arange(y_min, y_max, resolution)
#     grid_x, grid_y = np.meshgrid(x, y)
    
#     # Create paths for polygon checks
#     outer_path = Path(outer)
#     inner_path = Path(inner)
    
#     # Check each grid point
#     points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
#     in_outer = outer_path.contains_points(points)
#     in_inner = inner_path.contains_points(points)
    
#     # Drivable = inside outer AND outside inner
#     drivable = (in_outer & ~in_inner).reshape(grid_x.shape)
    
#     return drivable, map_size_meters, map_size_pixels, (x_min, x_max), (y_min, y_max), resolution

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    original_resolution = yaml_data['resolution']  # meters/pixel
    origin = yaml_data['origin']         # [x, y, z] offset in meters
    negate = yaml_data.get('negate', 0)  # Invert colors if needed
    # Threshold to binary (white=drivable, black=obstacle)
    occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
    free_thresh = yaml_data.get('free_thresh', 0.196)
    return original_resolution,origin,free_thresh

def load_image_downsampled(visualization_scale, original_resolution, png_path):
    image = Image.open(png_path).convert('L')  # Grayscale
    image_array = np.array(image)
    
    # --- NEW: Downsample image for visualization ---
    if visualization_scale != 1.0:
        new_height = int(image_array.shape[0] * visualization_scale)
        new_width = int(image_array.shape[1] * visualization_scale)
        image_array = np.array(image.resize((new_width, new_height), Image.Resampling.LANCZOS))
        
    # Calculate effective resolution after scaling
    resolution = original_resolution / visualization_scale
    return image_array, resolution

def set_stage_usd(file_path):
    try:
        stage = Usd.Stage.Open(file_path)
        print('[INFO]: Opening existing map')
    except:
        stage = Usd.Stage.CreateNew(file_path)
        print('[INFO]: Creating new map')
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        xform = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(xform.GetPrim())
    return stage

def set_plane_usd(hashmap, map_size_pixels, map_size_meters, stage, x_min, x_max, y_min, y_max, resolution):
    plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')
    # Create vertices 
    xs = np.linspace(
        -map_size_meters[1]/2,
        map_size_meters[1]/2,
        map_size_pixels[1]
    )
    ys = np.linspace(
        -map_size_meters[0]/2,
        map_size_meters[0]/2,
        map_size_pixels[0]
    )
    xx, yy = np.meshgrid(xs, ys)
    vertices = [(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())]
    # Create faces (same as your original code)
    faces = []
    face_counts = []
    for row in range(map_size_pixels[0] - 1):
        for col in range(map_size_pixels[1] - 1):
            v0 = row * map_size_pixels[1] + col
            v1 = v0 + 1
            v2 = v0 + map_size_pixels[1]
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]
    # Assign colors (white=drivable, black=obstacle)
    colors = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]  # Black, White
    face_colors = []
    for row in range(map_size_pixels[0] - 1):
        for col in range(map_size_pixels[1] - 1):
            face_colors.append(colors[int(hashmap[row, col])])
    
    # Double colors for triangles
    face_colors_triangle = [c for color_pair in zip(face_colors, face_colors) for c in color_pair]
    # Set mesh attributes
    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors_triangle)

def set_points_usd(points, name, origin, map_size_meters, stage, color, x_min, y_min):
    points_usd = [
        Gf.Vec3f(
            wp[0]+(-map_size_meters[1]/2-x_min),  # X: Apply resolution and shift
            wp[1]+(-map_size_meters[0]/2-y_min),  # Y
            0.0                          # Z offset
        ) 
        for wp in points
    ]
    points_prim = UsdGeom.Points.Define(stage, '/World/' + name)
    points_prim.GetPointsAttr().Set(points_usd)
    # Visual stylin
    points_prim.CreateWidthsAttr().Set([0.1] * len(points_usd))  # 20cm diameter points
    points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(color)  # Red
    points_prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    return points_usd

def ensure_consistent_direction(centerline, outer, inner):
    """
    Ensure outer and inner boundaries follow the same direction as centerline.
    Returns corrected outer and inner boundaries.
    """
    def calculate_direction(points):
        """Calculate cumulative direction change (positive for CCW, negative for CW)"""
        vectors = np.diff(points, axis=0)
        cross_product = np.cross(vectors[:-1], vectors[1:])
        return np.sum(cross_product)  # Sum of all z-components of cross products
    
    # Calculate direction for each path
    center_dir = calculate_direction(centerline)
    outer_dir = calculate_direction(outer)
    inner_dir = calculate_direction(inner)
    
    # Flip if directions don't match centerline
    if np.sign(center_dir) != np.sign(outer_dir):
        outer = np.flipud(outer)
    if np.sign(center_dir) != np.sign(inner_dir):
        inner = np.flipud(inner)
    
    return outer, inner

def separate_and_align_bounds(centerline, trackbounds):
    """Separate and align boundaries with centerline direction"""
    # First separate the bounds
    outer, inner = separate_bounds(trackbounds)
    
    # Then ensure consistent direction
    outer, inner = ensure_consistent_direction(centerline, outer, inner)
    
    return outer, inner

def match_by_projection(centerline, outer, inner):
    """
    Improved projection-based matching with proper handling of edge cases
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import interp1d
    
    # 1. Create a continuous parameterized centerline
    cum_dist = np.zeros(len(centerline))
    cum_dist[1:] = np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    total_length = cum_dist[-1]
    
    # 2. Create dense centerline representation
    dense_steps = np.linspace(0, total_length, len(centerline)*10)
    dense_center = np.column_stack([
        interp1d(cum_dist, centerline[:,0], kind='linear')(dense_steps),
        interp1d(cum_dist, centerline[:,1], kind='linear')(dense_steps)
    ])
    
    # 3. Find nearest dense center point for each boundary point
    center_tree = cKDTree(dense_center)
    _, outer_assignments = center_tree.query(outer)
    _, inner_assignments = center_tree.query(inner)
    
    # 4. Group boundary points by their nearest centerline segment
    outer_matched = np.full_like(centerline, np.nan)
    inner_matched = np.full_like(centerline, np.nan)
    
    segment_size = len(dense_center) // len(centerline)
    for i in range(len(centerline)):
        start = i * segment_size
        end = (i + 1) * segment_size
        
        # Outer boundary points for this segment
        outer_segment = outer[(outer_assignments >= start) & (outer_assignments < end)]
        if len(outer_segment) > 0:
            outer_matched[i] = np.median(outer_segment, axis=0)
        
        # Inner boundary points for this segment
        inner_segment = inner[(inner_assignments >= start) & (inner_assignments < end)]
        if len(inner_segment) > 0:
            inner_matched[i] = np.median(inner_segment, axis=0)
    
    # 5. Fill any remaining NaN values using interpolation
    def fill_nans(points):
        valid = ~np.isnan(points[:,0])
        if np.all(valid):
            return points
        
        f = interp1d(cum_dist[valid], points[valid], axis=0, 
                     kind='linear', fill_value='extrapolate')
        return f(cum_dist)
    
    outer_matched = fill_nans(outer_matched)
    inner_matched = fill_nans(inner_matched)
    
    return outer_matched, inner_matched

def load_waypoints(waypoints_path):
    try:
        with open(waypoints_path) as f:
            waypoint_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load waypoints from {waypoints_path}: {str(e)}")
        # Extract waypoints from JSON - fixed the extra bracket, shift to match origin
        
    waypoints = np.array([
        [marker['pose']['position']['x'], 
            marker['pose']['position']['y']] 
        for marker in waypoint_data['centerline_markers']['markers']
    ])
    trackbounds = np.array([
        [markers['pose']['position']['x'], 
            markers['pose']['position']['y']] 
        for markers in waypoint_data['trackbounds_markers']['markers']
    ])
    d_lat = np.array([
        [wpnts['d_left'], 
            wpnts['d_right']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])
    psi_rad = np.array([
        [wpnts['psi_rad']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])
    kappa_radpm = np.array([
        [wpnts['kappa_radpm']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])
    vx_mps = np.array([
        [wpnts['vx_mps']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])
    return waypoints, trackbounds, d_lat, psi_rad, kappa_radpm, vx_mps

def generate_random_poses(num_poses, row_spacing, col_spacing, traversability_hashmap, waypoints_usd, outer_usd, inner_usd, margin=0.1):
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
    lookahead = 4
    next_indices = (current_indices + lookahead) % len(inner_xy)
    
    # we take the inner bound instead of the closest waypoint to avoid case when the centerline from the other side of the wall is closer
    current_wps = inner_xy[current_indices]
    next_wps = inner_xy[next_indices]
    
    deltas = next_wps - current_wps
    angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15,15)
    
    # Combine results
    poses = list(zip(xs.tolist(), ys.tolist(), angles.tolist()))
    
    return poses

def find_nearest_waypoint(waypoints: torch.Tensor, positions: torch.Tensor):
    """Optimized nearest waypoint finder."""
    # Squared distances are faster to compute and preserve order
    deltas = positions.unsqueeze(1) - waypoints.unsqueeze(0)
    sq_dists = torch.sum(deltas**2, dim=2)
    min_dist, closest_idx = torch.min(sq_dists, dim=1)
    return closest_idx, torch.sqrt(min_dist)  # Only sqrt the min distances

if __name__ == "__main__":
    create_maps_from_png('test.usd', 100, 100, 0.3, 0.3, 0.3)

# def create_hashmap(free_thresh, image_array, resolution):
#     hashmap = (image_array / 255.0 > free_thresh).astype(bool)
#     # Get map dimensions in pixels and meters
#     map_size_pixels = image_array.shape  # (height, width)
#     map_size_meters = (
#         map_size_pixels[0] * resolution,
#         map_size_pixels[1] * resolution
#     )
    
#     return hashmap,map_size_pixels,map_size_meters
# def generate_random_poses(num_poses, row_spacing, col_spacing, traversability_hashmap, envs_boundaries, num_rows, num_cols, margin=0.1):
#     """Generate poses guaranteed to be on distinct tracks
    
#     Args:
#         num_poses: Number of poses to generate
#         row_spacing: Distance between rows in meters
#         col_spacing: Distance between columns in meters
#         traversability_hashmap: 2D boolean array of drivable areas
#         track_boundaries: List of (min_row, max_row, min_col, max_col) tuples
#         num_rows: Total rows in the full grid
#         num_cols: Total columns in the full grid
#         margin: Safety margin from track edges (unused in this version)
#     """
#     poses = []
#     envs_available = list(range(len(envs_boundaries)))
    
#     # Ensure we don't request more poses than available tracks
#     num_poses = min(num_poses, len(envs_available))
    
#     for track_idx in envs_available[:num_poses]:
#         min_row, max_row, min_col, max_col = envs_boundaries[track_idx]
        
#         # Find all valid positions within this track
#         # TO ADJUST
#         rows, cols = np.where(
#             traversability_hashmap[min_row:max_row+1][min_col:max_col+1] == True
#         )
#         if len(rows) == 0 or len(cols) == 0:
#             continue
            
#         # Select random position within track bounds
#         idx_rows = np.random.choice(len(rows))
#         idx_cols = np.random.choice(len(cols))
#         row = min_row + rows[idx_rows]
#         col = min_col + cols[idx_cols]
        
#         # Convert to world coordinates (center at (0,0))
#         x = (col - num_cols//2) * col_spacing
#         y = (row - num_rows//2) * row_spacing
#         angle = np.random.uniform(0, 360)  # Full 360 degrees
        
#         poses.append((x, y, angle))
    
#     return poses
# def generated_colored_plane(map_size, spacing, env_size, sub_group_size, num_walkers, color_sampling):
#     """
#     Generate a colored plane with a custom number of rows and columns.
#     """
#     num_rows, num_cols = map_size
#     row_spacing, col_spacing = spacing
#     env_num_rows, env_num_cols = env_size
#     # Spacing between points
#     width = num_rows * row_spacing
#     height = num_cols * col_spacing
#     if num_rows % env_num_rows != 0 or num_cols % env_num_cols != 0:
#         raise ValueError("Map size must be a multiple of the sub environment size.")
#     num_env_rows = num_rows // env_num_rows
#     num_env_cols = num_cols // env_num_cols
#     xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing/2
#     ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing/2
#     xx, yy = np.meshgrid(xs, ys)
#     vertices = []
#     for x, y in zip(xx.ravel(), yy.ravel()):
#         vertices.append((x, y, 0))
#     def color_sampler(r, g, b, range):
#         r = np.random.uniform(r - range // 2, r + range // 2) / 255.
#         g = np.random.uniform(g - range // 2, g + range // 2) / 255.
#         b = np.random.uniform(b - range // 2, b + range // 2) / 255.
#         return Gf.Vec3f(r, g, b)
#     # Create a face color primvar
#     if color_sampling:
#         colors = [
#             color_sampler(30, 30, 30, 30),  # black for non-traversable area
#             color_sampler(220.0, 220.0, 220.0, 30),  # white for traversable area
#         ]
#     else:
#         colors = [
#             Gf.Vec3f(0.0, 0.0, 0.0),  # black for non-traversable area
#             Gf.Vec3f(1.0, 1.0, 1.0),  # white for traversable area 
#         ]
#     # Define faces using the indices of the grid points
#     faces = []
#     face_counts = []
#     face_colors = []
#     # Define traversability hash map that contains
#     # 0 for non-traversable area and 1 for traversable area
#     # make triangle mesh
#     traversability_hashmap = np.zeros((num_rows, num_cols)).astype(bool)
#     for row_index in range(num_rows - 1):
#         for col_index in range(num_cols - 1):
#             # Calculate the indices of the corners of the cell
#             # left-top
#             v0 = row_index * num_cols + col_index
#             # right-top
#             v1 = v0 + 1
#             # left-bottom
#             v2 = v0 + num_cols
#             # right-bottom
#             v3 = v2 + 1
#             faces += [v0, v1, v2, v2, v1, v3]
#             face_counts += [3, 3]
#     for i in range(num_env_rows):
#         for j in range(num_env_cols):
#             start_row = i * env_num_rows
#             end_row = (i + 1) * env_num_rows
#             start_col = j * env_num_cols
#             end_col = (j + 1) * env_num_cols
#             traversability_hashmap[start_row:end_row, start_col:end_col]  =\
#                   generate_env_map(env_size, sub_group_size, num_walkers)
#     # dilate the path using asymmetric L1 structure
#     dilate_structure = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]).astype(bool)
#     traversability_hashmap = binary_dilation(traversability_hashmap, structure=dilate_structure, iterations=1)
#     face_colors = [colors[int(traversability_hashmap[row_index, col_index])] for row_index in range(num_rows - 1) for col_index in range(num_cols - 1)]
#     face_colors_triangle = []
#     for color in face_colors:
#         face_colors_triangle += [color, color]
#     return vertices, faces, face_counts, face_colors_triangle, traversability_hashmap
# def generate_env_map(env_size, sub_group_size, num_walkers):
#     """
#     Generate a map for the environment.
#     """
#     env_num_rows, env_num_cols = env_size
#     group_num_rows, group_num_cols = sub_group_size
#     traversability_hashmap = np.zeros((env_num_rows, env_num_cols)).astype(bool)
#     start_points = []
#     for i in range(env_num_rows // group_num_rows):
#         for j in range(env_num_cols // group_num_cols):
#             start_row = np.random.randint(0, group_num_rows) + i * group_num_rows
#             start_col = np.random.randint(0, group_num_cols) + j * group_num_cols
#             start_points.append((start_row, start_col))
#     for start_row, start_col in start_points:
#         for _ in range(num_walkers):
#             end_row = np.random.randint(0, env_num_rows)
#             end_col = np.random.randint(0, env_num_cols)
#             while traversability_hashmap[end_row, end_col] == 1:
#                 end_row = np.random.randint(0, env_num_rows)
#                 end_col = np.random.randint(0, env_num_cols)
#             generate_path(start_row, start_col, end_row, end_col, traversability_hashmap)
    
#     return traversability_hashmap
# def generate_path(start_row, start_col, end_row, end_col, traversability_hashmap):
#     actions = ['up', 'down', 'left', 'right']
#     current_row, current_col = start_row, start_col
#     traversability_hashmap[current_row, current_col] = 1
#     row_diff = end_row - current_row
#     row_action = 'up' if row_diff < 0 else 'down'
#     col_diff = end_col - current_col
#     col_action = 'left' if col_diff < 0 else 'right'
#     action_sequences = [row_action for i in range(abs(row_diff))]
#     action_sequences += [col_action for i in range(abs(col_diff))]
#     random_path = np.random.permutation(action_sequences)
#     for action in random_path:
#         traversability_hashmap[current_row, current_col] = 1
#         if action == 'up':
#             current_row -= 1
#         elif action == 'down':
#             current_row += 1
#         elif action == 'left':
#             current_col -= 1
#         elif action == 'right':
#             current_col += 1
#         traversability_hashmap[current_row, current_col] = 1
# def create_geometry(file_path, map_size, spacing, env_size, sub_group_size, num_walkers=16, color_sampling=False):
#     """
#     Create a USD file with a colored plane geometry. 
#     """
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     # Create a new stage
#     stage = Usd.Stage.CreateNew(file_path)
#     UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
#     UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
#     # Define a new Xform (transform) at the root of the stage
#     xform = UsdGeom.Xform.Define(stage, '/World')
#     stage.SetDefaultPrim(xform.GetPrim())
#     # Define a plane under the Xform
#     plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')
#     _ = generated_colored_plane(map_size, spacing, env_size, sub_group_size, num_walkers, color_sampling)
#     vertices, faces, face_counts, face_colors, traversability_hashmap = _
#     plane.GetPointsAttr().Set(vertices)
#     plane.GetFaceVertexCountsAttr().Set(face_counts)
#     plane.GetFaceVertexIndicesAttr().Set(faces)
#     plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors)
#     # Apply the CollisionGroup schema
#     collisionAPI = UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
#     collisionAPI2 = UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
#     collisionGroup = UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")
#     # Save the stage to the file
#     stage.GetRootLayer().Save()
#     traversability_hashmap = traversability_hashmap.tolist()
#     TraversabilityHashmapUtil().set_traversability_hashmap(
#         traversability_hashmap, map_size, spacing)
#     return traversability_hashmap
# def create_oval(file_path, map_size, spacing, env_size):
#     """
#     Create a USD file with a colored plane geometry representing a race track.
#     """
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     # Create a new stage
#     stage = Usd.Stage.CreateNew(file_path)
#     UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
#     UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
#     # Define a new Xform (transform) at the root of the stage
#     xform = UsdGeom.Xform.Define(stage, '/World')
#     stage.SetDefaultPrim(xform.GetPrim())
#     # Define a plane under the Xform
#     plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')
#     # Generate mesh components (vertices, faces, etc.)
#     num_rows, num_cols = map_size
#     env_num_rows, env_num_cols = env_size
#     nx_env = num_cols//env_num_cols
#     ny_env = num_rows//env_num_rows
#     row_spacing, col_spacing = spacing
#     width = num_rows * row_spacing
#     height = num_cols * col_spacing
    
#     # Create local coordinate grids
#     rows = np.arange(env_num_rows)
#     cols = np.arange(env_num_cols)
#     local_row, local_col = np.meshgrid(rows - env_num_rows/2, 
#                                       cols - env_num_cols/2, 
#                                       indexing='ij')
    
#     xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing/2
#     ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing/2
#     xx, yy = np.meshgrid(xs, ys)
    
#     # 
#     vertices = []
#     for x, y in zip(xx.ravel(), yy.ravel()):
#         vertices.append((x, y, 0))
#     # Define faces using the indices of the grid points
#     faces = []
#     face_counts = []
#     for row_index in range(num_rows - 1):
#         for col_index in range(num_cols - 1):
#             # Calculate the indices of the corners of the cell
#             v0 = row_index * num_cols + col_index
#             v1 = v0 + 1
#             v2 = v0 + num_cols
#             v3 = v2 + 1
#             faces += [v0, v1, v2, v2, v1, v3]
#             face_counts += [3, 3]
#     # Generate race track pattern
#     envs_boundaries = []
#     traversability_hashmap = np.zeros((num_rows, num_cols), dtype=bool)
    
#     # Oval equations
#     track_width = 10
#     outer_major = env_num_rows/4 - 2
#     outer_minor = env_num_cols/2 - 2
#     inner_major = outer_major - track_width
#     inner_minor = outer_minor - track_width
    
#     outer_dist = (local_row/outer_major)**2 + (local_col/outer_minor)**2
#     inner_dist = (local_row/inner_major)**2 + (local_col/inner_minor)**2
#     track_mask = (inner_dist >= 1) & (outer_dist <= 1)
    
#     # Stamp tracks into global map
#     for nx in range(nx_env):
#         for ny in range(ny_env):
#             start_row = nx * env_num_rows
#             start_col = ny * env_num_cols
#             end_row = start_row + env_num_rows
#             end_col = start_col + env_num_cols
            
#             if end_row <= num_rows and end_col <= num_cols:
#                 traversability_hashmap[start_row:end_row, start_col:end_col] |= track_mask
                
#                 # Store center in grid coordinates
#                 envs_boundaries.append((
#                     start_row, end_row-1,  # Convert to inclusive indices
#                     start_col, end_col-1
#                 ))
#     # Create colors based on track
#     colors = [
#             Gf.Vec3f(0.0, 0.0, 0.0),  # black for off-track
#             Gf.Vec3f(1.0, 1.0, 1.0),   # white for track
#     ]
#     face_colors = []
#     for row_index in range(num_rows - 1):
#         for col_index in range(num_cols - 1):
#             # Use the color based on whether the cell is on track
#             face_colors.append(colors[int(traversability_hashmap[row_index, col_index])])
    
#     # Double colors for triangles
#     face_colors_triangle = []
#     for color in face_colors:
#         face_colors_triangle += [color, color]
#     # Set mesh attributes
#     plane.GetPointsAttr().Set(vertices)
#     plane.GetFaceVertexCountsAttr().Set(face_counts)
#     plane.GetFaceVertexIndicesAttr().Set(faces)
#     plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors_triangle)
#     # Apply the CollisionGroup schema
#     collisionAPI = UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
#     collisionAPI2 = UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
#     collisionGroup = UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")
#     # --- Add Point-Based Centerline ---
#     points_prim = UsdGeom.Points.Define(stage, '/World/Centerline_Points')
#     # Generate points (same as before)
#     center_major = (outer_major + inner_major) / 2
#     center_minor = (outer_minor + inner_minor) / 2
#     num_points = 100
#     theta = np.linspace(0, 2 * np.pi, num_points)
#     y_center = center_major * np.cos(theta) * row_spacing
#     x_center = center_minor * np.sin(theta) * col_spacing
#     z_center = np.full_like(x_center, 0.1)  # 1m above ground
#     # Set point positions
#     points = [Gf.Vec3f(x, y, z) for x, y, z in zip(x_center, y_center, z_center)]
#     points_prim.GetPointsAttr().Set(points)
#     # Visual styling
#     points_prim.CreateWidthsAttr().Set([0.2] * len(points))  # 20cm diameter points
#     points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(1.0, 0.0, 0.0)])  # Red
#     points_prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
#     # Optional: Add instanced spheres for better visibility
#     sphere_prim = UsdGeom.Sphere.Define(stage, '/World/Centerline_Prototype')
#     sphere_prim.CreateRadiusAttr().Set(0.1)  # 10cm radius
#     sphere_prim.CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
#     point_instancer = UsdGeom.PointInstancer.Define(stage, '/World/Centerline_Instancer')
#     point_instancer.CreatePrototypesRel().SetTargets([sphere_prim.GetPath()])
#     point_instancer.CreatePositionsAttr().Set(points)
#     point_instancer.CreateOrientationsAttr().Set([Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(points))
#     # --- Save Stage (existing code) ---
#     stage.GetRootLayer().Save()
#     traversability_hashmap = traversability_hashmap.tolist()
#     TraversabilityHashmapUtil().set_traversability_hashmap(
#         traversability_hashmap, map_size, spacing)
#     return traversability_hashmap, envs_boundaries
