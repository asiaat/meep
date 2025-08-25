import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def plot_cylindrical_geometry(cell_size, geometry, resolution):
    """
    Visualizes the 2D cross-section of a Meep geometry defined in
    cylindrical coordinates with a 1D computational cell.

    Args:
        cell_size (mp.Vector3): The size of the simulation cell (radial dimension).
        geometry (list): A list of Meep geometry objects.
        resolution (int): The resolution of the simulation.
    """
    # --- Step 1: Get the 1D refractive index profile along the radius ---
    
    # Create a temporary 1D simulation object to extract epsilon
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        dimensions=mp.CYLINDRICAL
    )
    sim.init_sim()
    
    # Get the 1D array of epsilon values along the radial direction
    eps_data_1d = sim.get_epsilon()
    n_data_1d = np.sqrt(eps_data_1d)
    
    # Manually create the radial coordinates for the 1D data
    sr = cell_size.x  # Radial size is in the x-component
    r_coords_1d = np.linspace(0, sr, len(n_data_1d))

    # --- Step 2: Reconstruct the 2D cross-section ---
    
    # Create a 2D Cartesian grid for plotting
    plot_size = sr
    num_pts_2d = int(2 * plot_size * resolution)
    xy_coords = np.linspace(-plot_size, plot_size, num_pts_2d)
    x_grid, y_grid = np.meshgrid(xy_coords, xy_coords)
    
    # Calculate the radial distance for each point in the 2D grid
    r_grid_2d = np.sqrt(x_grid**2 + y_grid**2)
    
    # Use NumPy's interpolation to map the 1D refractive index profile
    # onto the 2D radial grid. This is the core of the reconstruction.
    n_data_2d = np.interp(r_grid_2d, r_coords_1d, n_data_1d)

    # --- Step 3: Plot the 2D reconstructed geometry ---
    
    plt.figure(figsize=(8, 8), dpi=100)
    # Use pcolormesh for accurate pixel-centered plotting
    plt.pcolormesh(x_grid, y_grid, n_data_2d, shading='gouraud', cmap='viridis')
    
    # Formatting
    plt.title("2D Cross-Section of Cylindrical Geometry")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.gca().set_aspect('equal', adjustable='box')
    cbar = plt.colorbar()
    cbar.set_label("Refractive Index (n)")
    plt.show()

def plot_3d_cylindrical_geometry(r, w, n, height=1.0):
    """
    Creates a 3D plot for a ring resonator defined for a cylindrical simulation.

    Args:
        r (float): Inner radius of the ring.
        w (float): Width of the ring waveguide.
        n (float): Refractive index of the ring.
        height (float): The height of the ring in the z-direction.
    """
    # --- Step 1: Translate cylindrical definition to 3D Cartesian geometry ---
    
    # In a 3D Cartesian grid, a ring is a Cylinder object.
    # The Block object from the cylindrical simulation defines a ring with
    # inner radius 'r' and outer radius 'r + w'.
    ring_geo = mp.Cylinder(
        radius=r + w,
        height=height,
        material=mp.Medium(index=n),
        center=mp.Vector3(0, 0, 0)
    )
    
    # We "cut out" the center to form the ring
    inner_air_hole = mp.Cylinder(
        radius=r,
        height=height,
        material=mp.air, # material is air
        center=mp.Vector3(0, 0, 0)
    )
    
    # --- Step 2: Create a 3D simulation object just for plotting ---
    
    # Define a 3D cell large enough to contain the ring
    cell_size_3d = mp.Vector3(2 * (r + w + 1), 2 * (r + w + 1), height + 2)
    
    sim_3d = mp.Simulation(
        cell_size=cell_size_3d,
        geometry=[ring_geo, inner_air_hole],
        resolution=20,
        dimensions=3
    )
    
    # --- Step 3: Call the 3D plotting function ---
    print("Generating 3D plot... (this may take a moment)")
    sim_3d.plot3D()


# ==============================================================================
# Example using your specific simulation parameters
# ==============================================================================
if __name__ == '__main__':
    # Define parameters
    n = 3.4
    w = 1
    r = 1
    pad = 4
    dpml = 2 # Reduced for faster visualization setup
    
    sr = r + w + pad + dpml
    cell = mp.Vector3(sr, 0, 0)
    resolution = 20 # Increased resolution for a smoother plot

    # Define geometry for the 1D CYLINDRICAL simulation
    geometry = [
        mp.Block(
            center=mp.Vector3(r + (w / 2)),
            size=mp.Vector3(w, 1e20, 1e20),
            material=mp.Medium(index=n)
        )
    ]

    # Call the 2D plotting function
    print("--- Plotting 2D Cross-Section ---")
    plot_cylindrical_geometry(cell, geometry, resolution)

    # Call the new 3D plotting function
    print("\n--- Plotting 3D Geometry ---")
    # You might need to install dependencies: pip install vispy scikit-image jupyter_rfb
    plot_3d_cylindrical_geometry(r=r, w=w, n=n, height=3.0)
