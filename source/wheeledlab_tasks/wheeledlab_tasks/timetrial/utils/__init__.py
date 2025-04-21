import os
import numpy as np
from scipy.ndimage import binary_dilation
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from .traversability_utils import *


    
def generated_colored_plane(map_size, spacing, env_size, sub_group_size, num_walkers, color_sampling):
    """
    Generate a colored plane with a custom number of rows and columns.
    """
    num_rows, num_cols = map_size
    row_spacing, col_spacing = spacing
    env_num_rows, env_num_cols = env_size

    # Spacing between points
    width = num_rows * row_spacing
    height = num_cols * col_spacing

    if num_rows % env_num_rows != 0 or num_cols % env_num_cols != 0:
        raise ValueError("Map size must be a multiple of the sub environment size.")

    num_env_rows = num_rows // env_num_rows
    num_env_cols = num_cols // env_num_cols

    xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing/2
    ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing/2
    xx, yy = np.meshgrid(xs, ys)

    vertices = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        vertices.append((x, y, 0))

    def color_sampler(r, g, b, range):
        r = np.random.uniform(r - range // 2, r + range // 2) / 255.
        g = np.random.uniform(g - range // 2, g + range // 2) / 255.
        b = np.random.uniform(b - range // 2, b + range // 2) / 255.
        return Gf.Vec3f(r, g, b)

    # Create a face color primvar
    if color_sampling:
        colors = [
            color_sampler(30, 30, 30, 30),  # black for non-traversable area
            color_sampler(220.0, 220.0, 220.0, 30),  # white for traversable area
        ]
    else:
        colors = [
            Gf.Vec3f(0.0, 0.0, 0.0),  # black for non-traversable area
            Gf.Vec3f(1.0, 1.0, 1.0),  # white for traversable area 
        ]

    # Define faces using the indices of the grid points
    faces = []
    face_counts = []
    face_colors = []

    # Define traversability hash map that contains
    # 0 for non-traversable area and 1 for traversable area
    # make triangle mesh
    traversability_hashmap = np.zeros((num_rows, num_cols)).astype(bool)
    for row_index in range(num_rows - 1):
        for col_index in range(num_cols - 1):
            # Calculate the indices of the corners of the cell
            # left-top
            v0 = row_index * num_cols + col_index
            # right-top
            v1 = v0 + 1
            # left-bottom
            v2 = v0 + num_cols
            # right-bottom
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    for i in range(num_env_rows):
        for j in range(num_env_cols):
            start_row = i * env_num_rows
            end_row = (i + 1) * env_num_rows
            start_col = j * env_num_cols
            end_col = (j + 1) * env_num_cols
            traversability_hashmap[start_row:end_row, start_col:end_col]  =\
                  generate_env_map(env_size, sub_group_size, num_walkers)

    # dilate the path using asymmetric L1 structure
    dilate_structure = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]).astype(bool)
    traversability_hashmap = binary_dilation(traversability_hashmap, structure=dilate_structure, iterations=1)

    face_colors = [colors[int(traversability_hashmap[row_index, col_index])] for row_index in range(num_rows - 1) for col_index in range(num_cols - 1)]
    face_colors_triangle = []
    for color in face_colors:
        face_colors_triangle += [color, color]

    return vertices, faces, face_counts, face_colors_triangle, traversability_hashmap

def generate_env_map(env_size, sub_group_size, num_walkers):
    """
    Generate a map for the environment.
    """
    env_num_rows, env_num_cols = env_size
    group_num_rows, group_num_cols = sub_group_size

    traversability_hashmap = np.zeros((env_num_rows, env_num_cols)).astype(bool)
    start_points = []
    for i in range(env_num_rows // group_num_rows):
        for j in range(env_num_cols // group_num_cols):
            start_row = np.random.randint(0, group_num_rows) + i * group_num_rows
            start_col = np.random.randint(0, group_num_cols) + j * group_num_cols
            start_points.append((start_row, start_col))

    for start_row, start_col in start_points:
        for _ in range(num_walkers):
            end_row = np.random.randint(0, env_num_rows)
            end_col = np.random.randint(0, env_num_cols)
            while traversability_hashmap[end_row, end_col] == 1:
                end_row = np.random.randint(0, env_num_rows)
                end_col = np.random.randint(0, env_num_cols)

            generate_path(start_row, start_col, end_row, end_col, traversability_hashmap)
    
    return traversability_hashmap

def generate_path(start_row, start_col, end_row, end_col, traversability_hashmap):
    actions = ['up', 'down', 'left', 'right']
    current_row, current_col = start_row, start_col
    traversability_hashmap[current_row, current_col] = 1

    row_diff = end_row - current_row
    row_action = 'up' if row_diff < 0 else 'down'
    col_diff = end_col - current_col
    col_action = 'left' if col_diff < 0 else 'right'

    action_sequences = [row_action for i in range(abs(row_diff))]
    action_sequences += [col_action for i in range(abs(col_diff))]

    random_path = np.random.permutation(action_sequences)

    for action in random_path:
        traversability_hashmap[current_row, current_col] = 1
        if action == 'up':
            current_row -= 1
        elif action == 'down':
            current_row += 1
        elif action == 'left':
            current_col -= 1
        elif action == 'right':
            current_col += 1
        traversability_hashmap[current_row, current_col] = 1

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

def create_track(file_path, map_size, spacing, env_size, sub_group_size, num_walkers=16, color_sampling=False):
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
    traversability_hashmap = np.zeros((num_rows, num_cols), dtype=bool)
    
    # Oval equations
    track_width = 5
    outer_major = env_num_rows/2 - 2
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
    
    # Create colors based on track
    if color_sampling:
        colors = [
            Gf.Vec3f(0.1, 0.1, 0.1),  # dark for off-track
            Gf.Vec3f(0.9, 0.9, 0.9),  # bright for track
        ]
    else:
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

    # Save the stage to the file
    stage.GetRootLayer().Save()

    traversability_hashmap = traversability_hashmap.tolist()
    TraversabilityHashmapUtil().set_traversability_hashmap(
        traversability_hashmap, map_size, spacing)
    return traversability_hashmap


def generate_random_poses(num_poses, row_spacing, col_spacing, traversability_hashmap, margin=0.1):
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

if __name__ == "__main__":
    create_track('test.usd', 100, 100, 0.3, 0.3, 0.3)
