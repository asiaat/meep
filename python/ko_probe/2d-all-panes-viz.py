import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def get_permittivity(material):
    """Helper function to get a scalar permittivity for plotting."""
    if material == mp.metal:
        # Use a large value for metal to make it distinct in the plot
        return 1e4
    if isinstance(material, mp.Medium):
        if hasattr(material, 'index') and material.index != 1.0:
            return material.index**2
        if hasattr(material, 'epsilon_diag'):
            # Assuming isotropic medium for visualization
            return material.epsilon_diag[0]
    # Default to air (permittivity of 1.0)
    return 1.0

def plot_meep_cross_section(ax, geometry, plane, slice_pos, range1, range2, resolution=500):
    """
    Calculates and plots a 2D cross-section of a Meep geometry list onto a given Matplotlib axis.

    ax: Matplotlib axis object to plot on.
    geometry: List of Meep geometry objects.
    plane: The plane to slice ('xy', 'yz', or 'xz').
    slice_pos: The position of the slice along the axis not in the plane.
    range1, range2: The (min, max) tuples for the two axes in the plane.
    resolution: The resolution of the grid for calculation.
    """
    # Define grid based on the specified plane
    axis_labels = [plane[0], plane[1]]
    min1, max1 = range1
    min2, max2 = range2
    
    v1 = np.linspace(min1, max1, resolution)
    v2 = np.linspace(min2, max2, resolution)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    
    eps = np.ones_like(V1)

    for obj in geometry:
        eps_val = get_permittivity(obj.material)
        cx, cy, cz = obj.center.x, obj.center.y, obj.center.z
        
        # Initialize an empty mask for the object
        mask = np.zeros_like(eps, dtype=bool)

        # --- XY Plane ---
        if plane == 'xy':
            X, Y = V1, V2
            sz = getattr(obj, 'height', getattr(obj, 'size', [0,0,0])[2])
            if abs(slice_pos - cz) > sz / 2 and not isinstance(obj, mp.Sphere):
                continue
            
            if isinstance(obj, mp.Cone):
                t = (slice_pos - (cz - obj.height/2)) / obj.height
                r_at_z = obj.radius + t * (obj.radius2 - obj.radius)
                mask = (X - cx)**2 + (Y - cy)**2 <= r_at_z**2
            elif isinstance(obj, mp.Cylinder):
                mask = (X - cx)**2 + (Y - cy)**2 <= obj.radius**2
            elif isinstance(obj, mp.Block):
                mask = (np.abs(X - cx) <= obj.size.x/2) & (np.abs(Y - cy) <= obj.size.y/2)
            elif isinstance(obj, mp.Sphere):
                d_sq = (slice_pos - cz)**2
                if d_sq <= obj.radius**2:
                    r_slice_sq = obj.radius**2 - d_sq
                    mask = (X - cx)**2 + (Y - cy)**2 <= r_slice_sq

        # --- YZ or XZ Plane (similar logic) ---
        else: # yz or xz
            Z = V2
            if plane == 'yz':
                Y = V1
                slice_axis_pos, obj_axis_pos = slice_pos, cx
                obj_axis_size = getattr(obj, 'size', [0,0,0])[0]
            else: # xz
                X = V1
                slice_axis_pos, obj_axis_pos = slice_pos, cy
                obj_axis_size = getattr(obj, 'size', [0,0,0])[1]

            if isinstance(obj, mp.Cone):
                mask_z = np.abs(Z - cz) <= obj.height/2
                t = (Z[mask_z] - (cz - obj.height/2)) / obj.height
                r_at_z = obj.radius + t * (obj.radius2 - obj.radius)
                
                # Check if the slice plane intersects the cone at this Z
                dist_sq = (slice_axis_pos - obj_axis_pos)**2
                width_sq = r_at_z**2 - dist_sq
                
                valid_z = width_sq >= 0
                half_width = np.sqrt(width_sq[valid_z])
                
                # Apply the mask for the other axis (X or Y)
                if plane == 'yz':
                    mask_y = np.abs(Y[mask_z][valid_z] - cy) <= half_width
                    full_mask = np.zeros_like(Z, dtype=bool)
                    full_mask[mask_z] = valid_z
                    full_mask[full_mask] = mask_y
                else: # xz
                    mask_x = np.abs(X[mask_z][valid_z] - cx) <= half_width
                    full_mask = np.zeros_like(Z, dtype=bool)
                    full_mask[mask_z] = valid_z
                    full_mask[full_mask] = mask_x
                mask = full_mask

            elif isinstance(obj, mp.Cylinder):
                 dist_sq = (slice_axis_pos - obj_axis_pos)**2
                 if dist_sq <= obj.radius**2:
                     half_width = np.sqrt(obj.radius**2 - dist_sq)
                     mask_z = np.abs(Z - cz) <= obj.height/2
                     if plane == 'yz': mask_v1 = np.abs(Y - cy) <= half_width
                     else: mask_v1 = np.abs(X - cx) <= half_width
                     mask = mask_v1 & mask_z
            
            elif isinstance(obj, mp.Block):
                if abs(slice_axis_pos - obj_axis_pos) <= obj_axis_size / 2:
                    mask_z = np.abs(Z - cz) <= obj.size.z/2
                    if plane == 'yz': mask_v1 = np.abs(Y - cy) <= obj.size.y/2
                    else: mask_v1 = np.abs(X - cx) <= obj.size.x/2
                    mask = mask_v1 & mask_z

            elif isinstance(obj, mp.Sphere):
                d_sq = (slice_axis_pos - obj_axis_pos)**2
                if d_sq <= obj.radius**2:
                    r_slice_sq = obj.radius**2 - d_sq
                    if plane == 'yz': mask = (Y - cy)**2 + (Z - cz)**2 <= r_slice_sq
                    else: mask = (X - cx)**2 + (Z - cz)**2 <= r_slice_sq
        
        eps[mask] = eps_val

    # --- Plotting ---
    extent = [min2, max2, min1, max1]
    im = ax.imshow(np.rot90(eps), extent=extent, cmap='viridis', aspect='auto', interpolation='spline16')
    ax.set_xlabel(f"{axis_labels[1]}-axis")
    ax.set_ylabel(f"{axis_labels[0]}-axis")
    ax.set_title(f"{plane.upper()} Plane at {plane.replace(axis_labels[0],'').replace(axis_labels[1],'')}={slice_pos}")
    return im

def visualize_all_panes(geometry, x_range, y_range, z_range, slice_positions):
    """
    Creates a 3-panel plot showing the XY, YZ, and XZ cross-sections of the geometry.

    geometry: List of Meep geometry objects.
    x_range, y_range, z_range: The (min, max) tuples for each axis.
    slice_positions: A tuple (x_slice, y_slice, z_slice) indicating where to
                     slice for the yz, xz, and xy planes respectively.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.style.use('default')

    x_slice, y_slice, z_slice = slice_positions

    # Plot XY, YZ, and XZ planes
    im_xy = plot_meep_cross_section(axes[0], geometry, 'xy', slice_pos=z_slice, range1=x_range, range2=y_range)
    im_yz = plot_meep_cross_section(axes[1], geometry, 'yz', slice_pos=x_slice, range1=y_range, range2=z_range)
    im_xz = plot_meep_cross_section(axes[2], geometry, 'xz', slice_pos=y_slice, range1=x_range, range2=z_range)

    fig.suptitle("Orthogonal Cross-Sections of Meep Geometry", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Adjust layout to make space for colorbar

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(im_xz, ax=axes.ravel().tolist(), shrink=0.8, location='bottom')
    cbar.set_label("Relative Permittivity (εᵣ)")
    
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    geometry = []

    # A hollow, tapered horn (last object is air, which carves out the first)
    geometry.append(mp.Cone(
        center=mp.Vector3(0, 0, 2.5),
        height=5.0,
        radius=1.5,
        radius2=18.0,
        material=mp.Medium(index=3.4)
    ))
    geometry.append(mp.Cone(
        center=mp.Vector3(0, 0, 2.5),
        height=5.0,
        radius=1.0,
        radius2=3.5,
        material=mp.air  # This creates the hollow part
    ))

    # An off-axis cylinder
    geometry.append(mp.Cylinder(
        center=mp.Vector3(5, 0, 2.5),
        radius=1.5,
        height=3,
        material=mp.Medium(index=2.0)
    ))

    # An off-axis block
    geometry.append(mp.Block(
        center=mp.Vector3(-8, 0, 3.5),
        size=mp.Vector3(4, 2, 3),
        material=mp.Medium(index=2.5)
    ))
    
    # A metal sphere
    geometry.append(mp.Sphere(
        center=mp.Vector3(-15, 0, 2.0),
        radius=1.5,
        material=mp.metal
    ))

    # Define ranges and slice positions for the plot
    x_range = (-20, 20)
    y_range = (-20, 20)
    z_range = (0, 6)
    # (slice_pos for YZ plane, slice_pos for XZ plane, slice_pos for XY plane)
    slice_positions = (0, 0, 2.5) 

    # Visualize the 3 orthogonal cross-sections with the new single function
    visualize_all_panes(geometry, x_range, y_range, z_range, slice_positions)

