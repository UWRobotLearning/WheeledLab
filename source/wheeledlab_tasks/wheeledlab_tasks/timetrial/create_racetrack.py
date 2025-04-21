# Example: Create a circular/oval track
def create_track_mask(num_rows, num_cols):
    mask = np.zeros((num_rows, num_cols), dtype=np.float32)
    center_x, center_y = num_rows // 2, num_cols // 2
    radius_inner, radius_outer = 3, 5  # Adjust for track width
    
    for i in range(num_rows):
        for j in range(num_cols):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if radius_inner <= dist <= radius_outer:
                mask[i, j] = 1.0  # Traversable
    return mask