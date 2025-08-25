import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_cylindrical_geometry_2d(cell_size, geometry, resolution):
    """
    Visualizes the 2D cross-section of a Meep geometry defined in
    cylindrical coordinates with a 1D computational cell.
    """
    print("--- Reconstructing 2D geometry from 1D cylindrical definition... ---")
    
    # --- Step 1: Get the 1D refractive index profile along the radius ---
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        dimensions=mp.CYLINDRICAL
    )
    sim.init_sim()
    
    eps_data_1d = sim.get_epsilon()
    n_data_1d = np.sqrt(eps_data_1d)
    
    sr = cell_size.x
    r_coords_1d = np.linspace(0, sr, len(n_data_1d))

    # --- Step 2: Reconstruct the 2D cross-section ---
    plot_size = sr
    num_pts_2d = int(2 * plot_size * (resolution * 2)) 
    xy_coords = np.linspace(-plot_size, plot_size, num_pts_2d)
    x_grid, y_grid = np.meshgrid(xy_coords, xy_coords)
    
    r_grid_2d = np.sqrt(x_grid**2 + y_grid**2)
    n_data_2d = np.interp(r_grid_2d, r_coords_1d, n_data_1d)

    # --- Step 3: Plot the 2D reconstructed geometry ---
    plt.figure(figsize=(8, 8), dpi=100)
    plt.pcolormesh(x_grid, y_grid, n_data_2d, shading='gouraud', cmap='viridis')
    
    plt.title("2D Cross-Section of Cylindrical Geometry (r-φ plane)")
    plt.xlabel("x position (μm)")
    plt.ylabel("y position (μm)")
    plt.gca().set_aspect('equal', adjustable='box')
    cbar = plt.colorbar()
    cbar.set_label("Refractive Index (n)")
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()
    print("--- 2D Plot displayed. ---")

def plot_cylindrical_geometry_3d(r, w, n, height=2.0):
    """
    Creates a 3D plot for a ring resonator defined for a cylindrical simulation.

    Args:
        r (float): Inner radius of the ring.
        w (float): Width of the ring waveguide.
        n (float): Refractive index of the ring.
        height (float): The height of the ring in the z-direction.
    """
    print("\n--- Creating 3D geometry for visualization... ---")
    
    # --- Step 1: Translate cylindrical definition to 3D Cartesian geometry ---
    # A Block in cylindrical coordinates becomes a Cylinder (ring) in Cartesian.
    ring_outer = mp.Cylinder(
        radius=r + w,
        height=height,
        material=mp.Medium(index=n),
        center=mp.Vector3(0, 0, 0)
    )
    
    # Create an inner cylinder of air to "cut out" the center, forming a ring.
    ring_inner_air = mp.Cylinder(
        radius=r,
        height=height,
        material=mp.air,
        center=mp.Vector3(0, 0, 0)
    )
    
    # --- Step 2: Create a temporary 3D simulation object just for plotting ---
    cell_size_3d = mp.Vector3(2 * (r + w + 1), 2 * (r + w + 1), height + 2)
    
    sim_3d = mp.Simulation(
        cell_size=cell_size_3d,
        geometry=[ring_outer, ring_inner_air],
        resolution=25, # A moderate resolution is fine for plotting
        dimensions=3
    )
    
    # --- Step 3: Call the 3D plotting function ---
    print("--- Launching interactive 3D plot... ---")
    sim_3d.plot3D()


# ==============================================================================
# Main script execution: Define parameters and call the plotting functions
# ==============================================================================
if __name__ == '__main__':
    # Define parameters (should match your simulation script)
    n = 3.4
    w = 1
    r = 1
    pad = 4
    dpml = 2
    
    sr = r + w + pad + dpml
    # This cell is for the 1D calculation needed for the 2D plot
    cell_1d = mp.Vector3(sr, 0, 0)
    resolution = 20

    # Define geometry for the 1D calculation
    geometry_1d = [
        mp.Block(
            center=mp.Vector3(r + (w / 2)),
            size=mp.Vector3(w, 1e20, 1e20),
            material=mp.Medium(index=n)
        )
    ]

    # Call the 2D plotting function
    plot_cylindrical_geometry_2d(cell_1d, geometry_1d, resolution)

    # Call the new 3D plotting function
    plot_cylindrical_geometry_3d(r=r, w=w, n=n, height=1.0)
