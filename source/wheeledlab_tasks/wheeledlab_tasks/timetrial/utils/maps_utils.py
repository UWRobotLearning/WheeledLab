
import os
import numpy as np
import math
from matplotlib.path import Path

from scipy.ndimage import binary_dilation
from pxr import Usd, UsdGeom, UsdPhysics, Gf

import json
import yaml
from PIL import Image

import time

def compute_origins(num_envs, env_size):
    # If no environments are needed, return an empty list
    env_length = env_size[0]

    if num_envs <= 0:
        return []
    
    # Calculate the smallest 'k' such that the grid can hold all environments.
    # The grid has (2k + 1) positions per side, centered around 0.
    # Example: For num_envs=5, math.sqrt(5)≈2.236 → k=ceil((2.236-1)/2)=1 → grid_size=3 (positions: -1,0,1)
    k = math.ceil((math.sqrt(num_envs) - 1) / 2)
    
    # Total positions per side of the grid (always odd to center around 0)
    grid_size = 2*k + 1
    
    # Create an array of indices [0, 1, 2, ..., num_envs-1] to map to grid coordinates
    indices = np.arange(num_envs)
    
    # Compute 'j' (x-coordinate in grid):
    # indices % grid_size → Position in the current row (0 to grid_size-1)
    # Subtract 'k' to center around 0 (e.g., 0 → -k, 1 → -k+1, ..., grid_size-1 → k)
    j = (indices % grid_size) - k
    
    # Compute 'i' (y-coordinate in grid):
    # indices // grid_size → Row number (0 to grid_size-1)
    # Subtract 'k' to center around 0 (e.g., row 0 → -k, row 1 → -k+1, etc.)
    i = (indices // grid_size) - k
    
    # Multiply coordinates by side_length to get actual origins, then zip into (x,y) tuples
    origins = list(zip(j * env_length, i * env_length))
    
    return origins

def get_min_map_size(outer, resolution):
    """Create square boolean hashmap of drivable areas."""
    # Create grid that fully contains outer boundary
    x_min, y_min = np.min(outer, axis=0)
    x_max, y_max = np.max(outer, axis=0)
    
    # Add small buffer to ensure outer boundary is fully contained
    buffer = resolution * 5
    x_min -= buffer
    y_min -= buffer
    x_max += buffer
    y_max += buffer
    
    # Calculate initial dimensions
    length_meters = x_max - x_min
    height_meters = y_max - y_min
    min_map_size = max(length_meters, height_meters)
    
    return min_map_size

def create_square_drivable_map_v2(min_map_size, outer, inner, resolution=0.5):
    """Create square boolean hashmap of drivable areas."""
    # Create grid that fully contains outer boundary
    x_min, y_min = np.min(outer, axis=0)
    x_max, y_max = np.max(outer, axis=0)
    
    # # Add small buffer to ensure outer boundary is fully contained
    buffer = resolution * 5
    x_min -= buffer
    y_min -= buffer
    x_max += buffer
    y_max += buffer
    
    # # Calculate initial dimensions
    # length_meters = x_max - x_min
    # height_meters = y_max - y_min
    
    # Make dimensions square by expanding the smaller dimension
    min_map_size = min_map_size
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Adjust bounds to create square
    x_min = center_x - min_map_size/2
    x_max = center_x + min_map_size/2
    y_min = center_y - min_map_size/2
    y_max = center_y + min_map_size/2
    
    # Calculate square dimensions in meters
    map_size_meters = (min_map_size, min_map_size)  # (height, length) - now equal
    
    # Calculate grid dimensions in pixels (now square)
    n_pixels = int(np.ceil(min_map_size / resolution)) + 1
    map_size_pixels = [n_pixels, n_pixels]  # (n_rows, n_cols) - now equal
    
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

def create_square_drivable_map(outer, inner, resolution=0.5):
    """Create square boolean hashmap of drivable areas."""
    # Create grid that fully contains outer boundary
    x_min, y_min = np.min(outer, axis=0)
    x_max, y_max = np.max(outer, axis=0)
    
    # Add small buffer to ensure outer boundary is fully contained
    buffer = resolution * 5
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
    map_size_pixels = [n_pixels, n_pixels]  # (n_rows, n_cols) - now equal
    
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

def set_stage_usd(file_path):
    # try:
    #     stage = Usd.Stage.Open(file_path)
    #     print('[INFO]: Opening existing map')
    # except:
    stage = Usd.Stage.CreateNew(file_path)
    print('[INFO]: Creating new map')
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())
    return stage

def set_hashmap_usd(map_name, hashmap, origin, map_size_pixels, map_size_meters, stage, x_min, x_max, y_min, y_max, resolution):
    plane = UsdGeom.Mesh.Define(stage, '/World/'+ map_name+'/hashmap')
    # Create vertices 
    xs = np.linspace(
        -map_size_meters[1]/2 + origin[0],
        map_size_meters[1]/2 + origin[0],
        map_size_pixels[1]
    )
    ys = np.linspace(
        -map_size_meters[0]/2 + origin[1],
        map_size_meters[0]/2 + origin[1],
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

# def set_multiple_planes_usd(hashmap_list, map_size_pixels_list, 
#                           map_size_meters_list, stage, x_min_list, x_max_list, 
#                           y_min_list, y_max_list, resolution_list):
#     """
#     Create multiple planes in USD, placed side by side along the x-axis.
    
#     Args:
#         All arguments are now lists containing data for each map
#     """
#     # Initialize combined data structures
#     all_vertices = []
#     all_faces = []
#     all_face_counts = []
#     all_face_colors = []
    
#     # Track current x offset for placing maps
#     current_offset = 0
    
#     for i, (hashmap, map_size_pixels, map_size_meters) in enumerate(zip(
#         hashmap_list, map_size_pixels_list, map_size_meters_list)):
        
#         # Calculate this map's dimensions
#         map_width = map_size_meters[1]
#         map_height = map_size_meters[0]
        
#         # Create vertices for this map with x offset
#         xs = np.linspace(
#             - map_width/2,
#              map_width/2,
#             map_size_pixels[1]
#         )
#         ys = np.linspace(
#             current_offset - map_height/2,
#             current_offset + map_height/2,
#             map_size_pixels[0]
#         )
#         xx, yy = np.meshgrid(xs, ys)
        
#         # Store current vertex count before adding new ones
#         prev_vertex_count = len(all_vertices)
        
#         # Add vertices for this map
#         all_vertices.extend([(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())])
        
#         # Create faces for this map with adjusted indices
#         for row in range(map_size_pixels[0] - 1):
#             for col in range(map_size_pixels[1] - 1):
#                 v0 = prev_vertex_count + row * map_size_pixels[1] + col
#                 v1 = v0 + 1
#                 v2 = v0 + map_size_pixels[1]
#                 v3 = v2 + 1
#                 all_faces += [v0, v1, v2, v2, v1, v3]
#                 all_face_counts += [3, 3]
        
#         # Create face colors for this map
#         colors = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]  # Black, White
#         for row in range(map_size_pixels[0] - 1):
#             for col in range(map_size_pixels[1] - 1):
#                 all_face_colors.append(colors[int(hashmap[row, col])])
        
#         # Update x offset for next map
#         current_x_offset += map_width
    
#     # Double colors for triangles (same as original)
#     all_face_colors_triangle = [c for color_pair in zip(all_face_colors, all_face_colors) for c in color_pair]
    
#     # Create the combined plane mesh
#     plane = UsdGeom.Mesh.Define(stage, '/World/combined_plane')
#     plane.GetPointsAttr().Set(all_vertices)
#     plane.GetFaceVertexCountsAttr().Set(all_face_counts)
#     plane.GetFaceVertexIndicesAttr().Set(all_faces)
#     plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(all_face_colors_triangle)

# def set_multiple_points_usd(points_list, name_list, map_size_meters_list, 
#                            stage, color_list, x_min_list, y_min_list):
#     """
#     Create multiple point sets in USD, placed with the same x-axis offsets as the planes.
    
#     Args:
#         points_list: List of point arrays for each map
#         name_list: List of names for each point set
#         origin_list: List of origin coordinates for each map
#         map_size_meters_list: List of map sizes in meters for each map
#         stage: USD stage
#         color_list: List of colors for each point set
#         x_min_list: List of x_min values for each map
#         y_min_list: List of y_min values for each map
#     """
#     # Track current x offset for placing points (must match plane placement)
#     current_x_offset = 0
#     all_points_prims = []
    
#     for i, (points, name, map_size_meters, x_min, y_min) in enumerate(zip(
#         points_list, name_list, map_size_meters_list, x_min_list, y_min_list)):
        
#         # Calculate this map's width
#         map_width = map_size_meters[1]
        
#         # Transform points with the same offset logic as planes
#         points_usd = [
#             Gf.Vec3f(
#                 wp[0] + current_x_offset - map_size_meters[1]/2 - x_min,  # X with offset
#                 wp[1] - map_size_meters[0]/2 - y_min,                      # Y (no x-offset needed)
#                 0.1  # Slight Z offset to appear above plane
#             ) 
#             for wp in points
#         ]
        
#         # Create points prim
#         points_prim = UsdGeom.Points.Define(stage, f'/World/{name}_{i}')
#         points_prim.GetPointsAttr().Set(points_usd)
        
#         # Visual styling
#         points_prim.CreateWidthsAttr().Set([0.1] * len(points_usd))
#         points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(color)
#         points_prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        
#         all_points_prims.append(points_prim)
        
#         # Update x offset for next map (must match plane placement)
#         current_x_offset += map_width
    
#     return all_points_prims

# def set_plane_usd_v2(hashmap, env_origins, map_size_pixels, map_size_meters, stage):
#     assert hashmap.shape == (map_size_pixels[0], map_size_pixels[1]), "Hashmap dimensions mismatch"
    
#     for i, (x_offset, y_offset) in enumerate(env_origins):
#         env_path = f"/World/env_{i}/plane"  # More conventional path
#         plane = UsdGeom.Mesh.Define(stage, env_path)
        
#         # Create vertices (offset applied ONLY here)
#         xs = np.linspace(
#             -map_size_meters[1]/2 + x_offset,
#             map_size_meters[1]/2 + x_offset,
#             map_size_pixels[1]
#         )
#         ys = np.linspace(
#             -map_size_meters[0]/2 + y_offset,
#             map_size_meters[0]/2 + y_offset,
#             map_size_pixels[0]
#         )
#         xx, yy = np.meshgrid(xs, ys)
#         vertices = [(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())]


#         faces = []
#         face_counts = []
#         for row in range(map_size_pixels[0] - 1):
#             for col in range(map_size_pixels[1] - 1):
#                 v0 = row * map_size_pixels[1] + col
#                 v1 = v0 + 1
#                 v2 = v0 + map_size_pixels[1]
#                 v3 = v2 + 1
#                 faces += [v0, v1, v2, v2, v1, v3]
#                 face_counts += [3, 3]
        
#         colors = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]
#         face_colors = [
#             colors[int(hashmap[row, col])] 
#             for row in range(map_size_pixels[0] - 1)
#             for col in range(map_size_pixels[1] - 1)
#         ]
#         face_colors_triangle = [c for color_pair in zip(face_colors, face_colors) for c in color_pair]
        
#         plane.GetPointsAttr().Set(vertices)
#         plane.GetFaceVertexCountsAttr().Set(face_counts)
#         plane.GetFaceVertexIndicesAttr().Set(faces)
#         plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors_triangle)


def set_points_usd(points, map_name, points_name, origin, map_size_meters, stage, color, x_min, y_min):
    points_usd = [
        Gf.Vec3f(
            wp[0]+(-map_size_meters[1]/2-x_min) + origin[0],  # X: Apply resolution and shift
            wp[1]+(-map_size_meters[0]/2-y_min) + origin[1],  # Y
            0.0                          # Z offset
        ) 
        for wp in points
    ]
    points_prim = UsdGeom.Points.Define(stage, '/World/'+ map_name +'/'+ points_name)
    points_prim.GetPointsAttr().Set(points_usd)
    # Visual stylin
    points_prim.CreateWidthsAttr().Set([0.1] * len(points_usd))  # 10cm diameter points
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

def separate_bounds(points):
    """Separate outer and inner boundaries based on maximum distance between points."""
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    split_idx = np.argmax(distances) + 1  # Index where bounds switch
    outer_bound = points[:split_idx]
    inner_bound = points[split_idx:]

    return outer_bound, inner_bound

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
