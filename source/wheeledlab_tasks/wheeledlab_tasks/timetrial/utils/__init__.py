import os
import numpy as np
from scipy.ndimage import binary_dilation
from pxr import Usd, UsdGeom, UsdPhysics, Gf

import yaml
from PIL import Image

from .traversability_utils import *
from .waypoints_utils import *

def create_oval(file_path, map_size, spacing, env_size):
    """
    Create a USD file with a colored plane geometry representing a race track.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create a new stage
    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Define a new Xform (transform) at the root of the stage
    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())

    # Define a plane under the Xform
    plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')

    # Generate mesh components (vertices, faces, etc.)
    num_rows, num_cols = map_size
    env_num_rows, env_num_cols = env_size

    nx_env = num_cols//env_num_cols
    ny_env = num_rows//env_num_rows

    row_spacing, col_spacing = spacing
    width = num_rows * row_spacing
    height = num_cols * col_spacing
    
    # Create local coordinate grids
    rows = np.arange(env_num_rows)
    cols = np.arange(env_num_cols)
    local_row, local_col = np.meshgrid(rows - env_num_rows/2, 
                                      cols - env_num_cols/2, 
                                      indexing='ij')
    
    xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing/2
    ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing/2
    xx, yy = np.meshgrid(xs, ys)
    
    # 
    vertices = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        vertices.append((x, y, 0))

    # Define faces using the indices of the grid points
    faces = []
    face_counts = []
    for row_index in range(num_rows - 1):
        for col_index in range(num_cols - 1):
            # Calculate the indices of the corners of the cell
            v0 = row_index * num_cols + col_index
            v1 = v0 + 1
            v2 = v0 + num_cols
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    # Generate race track pattern
    envs_boundaries = []
    traversability_hashmap = np.zeros((num_rows, num_cols), dtype=bool)
    
    # Oval equations
    track_width = 10
    outer_major = env_num_rows/4 - 2
    outer_minor = env_num_cols/2 - 2
    inner_major = outer_major - track_width
    inner_minor = outer_minor - track_width
    
    outer_dist = (local_row/outer_major)**2 + (local_col/outer_minor)**2
    inner_dist = (local_row/inner_major)**2 + (local_col/inner_minor)**2
    track_mask = (inner_dist >= 1) & (outer_dist <= 1)
    
    # Stamp tracks into global map
    for nx in range(nx_env):
        for ny in range(ny_env):
            start_row = nx * env_num_rows
            start_col = ny * env_num_cols
            end_row = start_row + env_num_rows
            end_col = start_col + env_num_cols
            
            if end_row <= num_rows and end_col <= num_cols:
                traversability_hashmap[start_row:end_row, start_col:end_col] |= track_mask
                
                # Store center in grid coordinates
                envs_boundaries.append((
                    start_row, end_row-1,  # Convert to inclusive indices
                    start_col, end_col-1
                ))

    # Create colors based on track
    colors = [
            Gf.Vec3f(0.0, 0.0, 0.0),  # black for off-track
            Gf.Vec3f(1.0, 1.0, 1.0),   # white for track
    ]

    face_colors = []
    for row_index in range(num_rows - 1):
        for col_index in range(num_cols - 1):
            # Use the color based on whether the cell is on track
            face_colors.append(colors[int(traversability_hashmap[row_index, col_index])])
    
    # Double colors for triangles
    face_colors_triangle = []
    for color in face_colors:
        face_colors_triangle += [color, color]

    # Set mesh attributes
    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors_triangle)

    # Apply the CollisionGroup schema
    collisionAPI = UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
    collisionAPI2 = UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    collisionGroup = UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")

    # --- Add Point-Based Centerline ---
    points_prim = UsdGeom.Points.Define(stage, '/World/Centerline_Points')

    # Generate points (same as before)
    center_major = (outer_major + inner_major) / 2
    center_minor = (outer_minor + inner_minor) / 2
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    y_center = center_major * np.cos(theta) * row_spacing
    x_center = center_minor * np.sin(theta) * col_spacing
    z_center = np.full_like(x_center, 0.1)  # 1m above ground

    # Set point positions
    points = [Gf.Vec3f(x, y, z) for x, y, z in zip(x_center, y_center, z_center)]
    points_prim.GetPointsAttr().Set(points)

    # Visual styling
    points_prim.CreateWidthsAttr().Set([0.2] * len(points))  # 20cm diameter points
    points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(1.0, 0.0, 0.0)])  # Red
    points_prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    # Optional: Add instanced spheres for better visibility
    sphere_prim = UsdGeom.Sphere.Define(stage, '/World/Centerline_Prototype')
    sphere_prim.CreateRadiusAttr().Set(0.1)  # 10cm radius
    sphere_prim.CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

    point_instancer = UsdGeom.PointInstancer.Define(stage, '/World/Centerline_Instancer')
    point_instancer.CreatePrototypesRel().SetTargets([sphere_prim.GetPath()])
    point_instancer.CreatePositionsAttr().Set(points)
    point_instancer.CreateOrientationsAttr().Set([Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(points))
    # --- Save Stage (existing code) ---
    stage.GetRootLayer().Save()

    traversability_hashmap = traversability_hashmap.tolist()
    TraversabilityHashmapUtil().set_traversability_hashmap(
        traversability_hashmap, map_size, spacing)
    return traversability_hashmap, envs_boundaries

def create_maps_from_png(map_name, file_path, visualization_scale):
    """
    Create a USD file and traversability hashmap from a PNG + YAML pair.
    
    Args:
        png_path: Path to the PNG image (black=obstacle, white=drivable).
        yaml_path: Path to the YAML file with metadata (resolution, origin).
        output_usd_path: Output USD file path (e.g., "/path/to/track.usd").
    """
    # Load YAML metadata
    maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'
    map_path = os.path.join(maps_folder_path, map_name)
    yaml_file = map_name+'.yaml'
    png_file = map_name+'.png'

    yaml_path = os.path.join(map_path, yaml_file)
    png_path = os.path.join(map_path, png_file)
    waypoints_path = os.path.join(map_path, 'global_waypoints.json')

    try:
        with open(waypoints_path) as f:
            waypoint_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load waypoints from {waypoints_path}: {str(e)}")
        # Extract waypoints from JSON - fixed the extra bracket, shift to match origin

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    original_resolution = yaml_data['resolution']  # meters/pixel
    origin = yaml_data['origin']         # [x, y, z] offset in meters
    negate = yaml_data.get('negate', 0)  # Invert colors if needed

    waypoints = np.array([
        [marker['pose']['position']['x'], 
            marker['pose']['position']['y']] 
        for marker in waypoint_data['centerline_markers']['markers']
    ])

    # Load PNG and convert to binary hashmap
    image = Image.open(png_path).convert('L')  # Grayscale
    image_array = np.array(image)
    
    # --- NEW: Downsample image for visualization ---
    if visualization_scale != 1.0:
        new_height = int(image_array.shape[0] * visualization_scale)
        new_width = int(image_array.shape[1] * visualization_scale)
        image_array = np.array(image.resize((new_width, new_height), Image.Resampling.LANCZOS))
    
    # Calculate effective resolution after scaling
    resolution = original_resolution / visualization_scale
        
    # Apply negate (if specified in YAML)
    # if negate:
    #     image_array = 255 - image_array
    
    # Threshold to binary (white=drivable, black=obstacle)
    occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
    free_thresh = yaml_data.get('free_thresh', 0.196)
    hashmap = (image_array / 255.0 > free_thresh).astype(bool)

    # Get map dimensions in pixels and meters
    map_size_pixels = image_array.shape  # (height, width)
    map_size_meters = (
        map_size_pixels[0] * resolution,
        map_size_pixels[1] * resolution
    )

    # Generate USD file (adapted from your original function)
    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())
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


    # Convert waypoints to USD coordinates (adjust for origin and flip Y if needed)
    waypoints_usd = [
        Gf.Vec3f(
            wp[0] - map_size_meters[0]/2 - origin[0],  # X: Apply resolution and shift
            -wp[1]+ map_size_meters[1]/2 + origin[1],  # Y
            0.1                          # Z offset
        ) 
        for wp in waypoints
    ]

    points_prim = UsdGeom.Points.Define(stage, '/World/waypoints')
    points_prim.GetPointsAttr().Set(waypoints_usd)

    # Visual styling
    points_prim.CreateWidthsAttr().Set([0.2] * len(waypoints_usd))  # 20cm diameter points
    points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(1.0, 0.0, 0.0)])  # Red
    points_prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    # Add collision (optional)
    # UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    stage.GetRootLayer().Save()

    # Return hashmap and metadata
    envs_boundaries = []  # Optional: Define if you need sub-regions
    spacing_meters = (resolution, resolution)  # Equal spacing in x/y
    traversability_hashmap = hashmap.tolist()
    TraversabilityHashmapUtil().set_traversability_hashmap(
        hashmap.tolist(), map_size_pixels, spacing_meters
    )

    return traversability_hashmap, waypoints_usd, spacing_meters, map_size_pixels

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

def generate_random_poses(num_poses, row_spacing, col_spacing, traversability_hashmap, waypoints, margin=0.1):
    """
    Generate random poses within the specified ranges.
    """
    H, W = len(traversability_hashmap), len(traversability_hashmap[0])
    pose_candidates = np.array(traversability_hashmap).nonzero()
    idxs = np.random.choice(len(pose_candidates[0]), num_poses)
    ys, xs = (pose_candidates[0][idxs], pose_candidates[1][idxs])
    poses = []
    for i in range(len(xs)):
        x = (float(xs[i]) - W // 2) * row_spacing
        y = (float(ys[i]) - H // 2) * col_spacing
        angle = np.random.uniform(0, 360.0)
        poses.append((x, y, angle))
    return poses

import json
from typing import Optional, Tuple
import numpy as np
import torch
import os
import yaml
from PIL import Image
from pxr import Usd, UsdGeom, Gf


class WaypointLoader:
    def __init__(self, map_name: str, map_origin_x: float = 0.0, map_origin_y: float = 0.0):
        """Initialize waypoint loader from JSON file.
        
        Args:
            json_path: Path to JSON file containing waypoints
            map_origin_x: X coordinate of map origin to shift waypoints
            map_origin_y: Y coordinate of map origin to shift waypoints
        """

        # Load YAML metadata
        maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'
        map_path = os.path.join(maps_folder_path, map_name)
        global_waypoints_path = os.path.join(map_path, 'global_waypoints.json')

        yaml_file = map_name+'.yaml'
        png_file = map_name+'.png'

        yaml_path = os.path.join(map_path, yaml_file)
        png_path = os.path.join(map_path, png_file)

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        resolution = yaml_data['resolution']  # meters/pixel
        origin = yaml_data['origin']         # [x, y, z] offset in meters
        negate = yaml_data.get('negate', 0)  # Invert colors if needed

        try:
            with open(global_waypoints_path) as f:
                self.waypoint_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load waypoints from {global_waypoints_path}: {str(e)}")

        image = Image.open(png_path).convert('L')  # Grayscale
        image_array = np.array(image)

        map_size_pixels = image_array.shape  # (height, width)
        map_size_meters = (
            map_size_pixels[0] * resolution,
            map_size_pixels[1] * resolution
        )

        # Extract waypoints from JSON - fixed the extra bracket, shift to match origin
        self.waypoints = np.array([
            [marker['pose']['position']['x'] - origin[1], 
             marker['pose']['position']['y'] - origin[0]] 
            for marker in self.waypoint_data['centerline_markers']['markers']
        ])

        self.waypoints = self.waypoints.tolist()

        # Validate waypoints
        if len(self.waypoints) == 0:
            raise ValueError("No waypoints found in JSON file")

        # Convert to tensor with proper shape [N, 2]
        # self.waypoints_tensor = torch.as_tensor(self.waypoints, dtype=torch.float32).tolist()

    def create_waypoints_usd(self, output_path: str) -> str:
        """Create a USD file with waypoint visualization and return its path."""
        stage = Usd.Stage.CreateNew(output_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        
        # Create parent Xform
        xform = UsdGeom.Xform.Define(stage, "/Waypoints")
        
        for i, (x, y) in enumerate(self.waypoints):
            prim_path = f"/Waypoints/wp_{i}"
            sphere = UsdGeom.Sphere.Define(stage, prim_path)
            sphere.CreateRadiusAttr().Set(0.2)
            sphere.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.1))
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])
        
        stage.GetRootLayer().Save()
        return output_path

    # def get_waypoints(self, device: str = 'cpu') -> torch.Tensor:
    #     """Get waypoints as a tensor on specified device.
        
    #     Returns:
    #         Tensor of shape [N, 2] containing (x,y) waypoint coordinates
    #     """
    #     return self.waypoints_tensor.to(device=device)

    def visualize_waypoints(self, stage: Usd.Stage, radius: float = 0.2) -> None:
        """Add waypoints to USD stage for visualization.
        
        Args:
            stage: USD stage to add visualization prims to
            radius: Visual size of waypoint markers
        """
        if not isinstance(stage, Usd.Stage):
            raise TypeError("stage must be a Usd.Stage object")

        for i, (x, y) in enumerate(self.waypoints):
            # Create sphere prim for each waypoint
            prim_path = f"/World/waypoint_{i}"
            sphere = UsdGeom.Sphere.Define(stage, prim_path)
            
            # Set sphere properties
            sphere.CreateRadiusAttr().Set(radius)
            sphere.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.1))
            
            # Set visual properties
            color = sphere.CreateDisplayColorAttr()
            color.Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Blue

    @property
    def num_waypoints(self) -> int:
        """Get number of waypoints."""
        return len(self.waypoints)

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get (min_x, max_x), (min_y, max_y) bounds of waypoints."""
        min_x, max_x = np.min(self.waypoints[:, 0]), np.max(self.waypoints[:, 0])
        min_y, max_y = np.min(self.waypoints[:, 1]), np.max(self.waypoints[:, 1])
        return (min_x, max_x), (min_y, max_y)


if __name__ == "__main__":
    create_tracks('test.usd', 100, 100, 0.3, 0.3, 0.3)


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

