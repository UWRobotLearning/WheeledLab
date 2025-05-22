import matplotlib.pyplot as plt
import json
import os
import numpy as np
from matplotlib.path import Path

def load_waypoints(waypoints_path):
    try:
        with open(waypoints_path) as f:
            waypoint_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load waypoints from {waypoints_path}: {str(e)}")
    
    waypoints = np.array([
        [wpnts['x_m'], wpnts['y_m']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])

    trackbounds = np.array([
        [markers['pose']['position']['x'], markers['pose']['position']['y']] 
        for markers in waypoint_data['trackbounds_markers']['markers']
    ])

    d_lat = np.array([
        [wpnts['d_left'], wpnts['d_right']] 
        for wpnts in waypoint_data['centerline_waypoints']['wpnts']
    ])

    return waypoints, trackbounds, d_lat

def separate_bounds(points):
    """Separate outer and inner boundaries based on maximum distance between points."""
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    split_idx = np.argmax(distances) + 1  # Index where bounds switch
    outer_bound = points[:split_idx]
    inner_bound = points[split_idx:]
    return outer_bound, inner_bound

def create_drivable_map(outer, inner, resolution=0.5):
    """Create boolean hashmap of drivable areas."""
    # Create grid that fully contains outer boundary
    x_min, y_min = np.min(outer, axis=0)
    x_max, y_max = np.max(outer, axis=0)
    
    # Add small buffer to ensure outer boundary is fully contained
    buffer = resolution * 2
    x_min -= buffer
    y_min -= buffer
    x_max += buffer
    y_max += buffer
    
    # Create grid coordinates
    x = np.arange(x_min, x_max, resolution)
    y = np.arange(y_min, y_max, resolution)
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
    
    return drivable, (x_min, x_max, y_min, y_max), grid_x, grid_y

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

def plot_lateral_distances(waypoints, outer, inner, d_lat, sample_every=10):
    """
    Plot the lateral distances (d_left and d_right) against the actual track boundaries.
    
    Args:
        waypoints: Centerline coordinates (Nx2)
        outer: Outer boundary coordinates (Mx2)
        inner: Inner boundary coordinates (Kx2)
        d_lat: Lateral distances (Nx2) where [:,0]=left, [:,1]=right
        sample_every: Plot every nth point for clarity
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot centerline and boundaries
    ax.plot(waypoints[::sample_every, 0], waypoints[::sample_every, 1], 'r-', label='Centerline')
    ax.plot(outer[::sample_every, 0], outer[::sample_every, 1], 'b-', label='Outer Boundary')
    ax.plot(inner[::sample_every, 0], inner[::sample_every, 1], 'g-', label='Inner Boundary')
    
    # Calculate and plot normal vectors at sample points
    for i in range(0, len(waypoints), sample_every):
        if i >= len(d_lat):  # Ensure we don't exceed d_lat bounds
            continue
            
        # Get current waypoint and normal vector
        if i < len(waypoints) - 1:
            tangent = waypoints[i+1] - waypoints[i]
        else:
            tangent = waypoints[i] - waypoints[i-1]
        
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Plot left distance (should reach inner boundary)
        left_point = waypoints[i] + normal * d_lat[i, 0]
        ax.plot([waypoints[i, 0], left_point[0]], 
                [waypoints[i, 1], left_point[1]], 
                'm-', alpha=0.3)
        
        # Plot right distance (should reach outer boundary)
        right_point = waypoints[i] - normal * d_lat[i, 1]
        ax.plot([waypoints[i, 0], right_point[0]], 
                [waypoints[i, 1], right_point[1]], 
                'c-', alpha=0.3)
    
    ax.set_title('Lateral Distance Verification')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def match_boundaries_to_centerline(centerline, outer, inner):
    """
    Create new boundary arrays with same length as centerline by finding nearest points.
    
    Args:
        centerline: (Nx2) array of centerline points
        outer: (Mx2) array of outer boundary points
        inner: (Kx2) array of inner boundary points
        
    Returns:
        matched_outer: (Nx2) outer boundary points matched to centerline
        matched_inner: (Nx2) inner boundary points matched to centerline
    """
    from scipy.spatial import cKDTree
    
    # Create KDTree for fast nearest neighbor search

    inner = inner + np.diff(inner)/2
    outer = outer + np.diff(outer)/2

    outer_tree = cKDTree(outer)
    inner_tree = cKDTree(inner)
    
    # Find nearest boundary points for each centerline point
    _, outer_indices = outer_tree.query(centerline)
    _, inner_indices = inner_tree.query(centerline)
    
    # Get the matched points
    matched_outer = outer[outer_indices]
    matched_inner = inner[inner_indices]
    
    return matched_outer, matched_inner

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

def plot_matched_boundaries(centerline, matched_outer, matched_inner, sample_every=5):
    """Visualize the matched boundaries"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original data
    ax.plot(centerline[::sample_every, 0], centerline[::sample_every, 1], 
            'r-', label='Centerline')
    ax.plot(matched_outer[::sample_every, 0], matched_outer[::sample_every, 1], 
            'b-', label='Matched Outer')
    ax.plot(matched_inner[::sample_every, 0], matched_inner[::sample_every, 1], 
            'g-', label='Matched Inner')
    
    # Draw connecting lines
    for i in range(0, len(centerline), sample_every):
        ax.plot([centerline[i, 0], matched_outer[i, 0]],
                [centerline[i, 1], matched_outer[i, 1]], 'b--', alpha=0.2)
        ax.plot([centerline[i, 0], matched_inner[i, 0]],
                [centerline[i, 1], matched_inner[i, 1]], 'g--', alpha=0.2)
    
    ax.set_title('Matched Boundaries Visualization')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'    
    map_name = 'THETRACK'
    map_path = os.path.join(maps_folder_path, map_name)
    waypoints_path = os.path.join(map_path, 'global_waypoints.json')
    
    # Load data
    waypoints, trackbounds, d_lat = load_waypoints(waypoints_path)
    
    # Separate boundaries
    outer, inner = separate_and_align_bounds(waypoints, trackbounds)
    
    # Create drivable map
    resolution = 0.05  # meters per pixel
    drivable_map, bounds, grid_x, grid_y = create_drivable_map(outer, inner, resolution)
    
    # Visualization
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original track plot
    # ax1.scatter(waypoints[:100,0], waypoints[:100,1], s=1, color='red', label="Centerline")
    # ax1.scatter(outer[:350,0], outer[:350,1], s=1, color='cyan', label="Outer Bound")
    # ax1.scatter(inner[200:,0], inner[200:,1], s=1, color='green', label="Inner Bound")
    # ax1.set_title("Original Track Boundaries")
    # ax1.legend()
    # ax1.axis('equal')
    
    # # Drivable area plot
    # ax2.imshow(drivable_map, 
    #           extent=(bounds[0], bounds[1], bounds[2], bounds[3]),
    #           origin='lower',
    #           cmap='binary')
    # ax2.set_title(f"Drivable Area Map (Resolution: {resolution}m/pixel)")
    # ax2.set_xlabel("X (m)")
    # ax2.set_ylabel("Y (m)")
    
    # plt.tight_layout()
    # plt.show()
    
    # # Plot lateral distances verification
    # plot_lateral_distances(waypoints, outer, inner, d_lat, sample_every=20)

    # Create matched boundaries
    matched_outer, matched_inner = match_by_projection(waypoints, outer, inner)
    
    # Verify lengths match
    print(f"Centerline points: {len(waypoints)}")
    print(f"Matched outer points: {len(matched_outer)}")
    print(f"Matched inner points: {len(matched_inner)}")
    
    # Visualize the matching
    plot_matched_boundaries(waypoints, matched_outer, matched_inner)
    
    # Now you can use these matched boundaries for your drivable map:
    drivable_map, bounds, grid_x, grid_y = create_drivable_map(matched_outer, matched_inner, resolution=0.1)