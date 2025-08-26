import meep as mp
import matplotlib.pyplot as plt
import math

def plot_hollow_taper_3d():
    """
    Creates and displays an interactive 3D plot of the
    hollow tapered waveguide geometry.
    """
    # ==============================================================================
    # 1. Define the geometry parameters
    # ==============================================================================
    resolution = 25  # pixels/μm

    w1 = 1.0  # width of waveguide 1
    w2 = 2.0  # width of waveguide 2
    Lw = 10.0 # length of straight waveguide sections
    wall_thickness = 0.2 # Thickness of the hollow waveguide walls
    
    # Use a fixed taper length for visualization
    Lt = 8.0 
    
    # Define a finite height for the prisms to make them 3D objects
    prism_height = 1.5

    # --- MODIFIED GEOMETRY ---
    # First, define the outer solid tapered waveguide
    outer_vertices = [
        mp.Vector3(-0.5 * Lw - 0.5 * Lt, 0.5 * w1),
        mp.Vector3(-0.5 * Lt, 0.5 * w1),
        mp.Vector3(0.5 * Lt, 0.5 * w2),
        mp.Vector3(0.5 * Lw + 0.5 * Lt, 0.5 * w2),
        mp.Vector3(0.5 * Lw + 0.5 * Lt, -0.5 * w2),
        mp.Vector3(0.5 * Lt, -0.5 * w2),
        mp.Vector3(-0.5 * Lt, -0.5 * w1),
        mp.Vector3(-0.5 * Lw - 0.5 * Lt, -0.5 * w1),
    ]
    solid_taper = mp.Prism(
        outer_vertices, 
        height=prism_height, 
        material=mp.Medium(epsilon=12.0)
    )

    # Second, define a slightly smaller inner prism of air to create the hollow core
    w1_inner = w1 - 2 * wall_thickness
    w2_inner = w2 - 2 * wall_thickness
    inner_vertices = [
        mp.Vector3(-0.5 * Lw - 0.5 * Lt, 0.5 * w1_inner),
        mp.Vector3(-0.5 * Lt, 0.5 * w1_inner),
        mp.Vector3(0.5 * Lt, 0.5 * w2_inner),
        mp.Vector3(0.5 * Lw + 0.5 * Lt, 0.5 * w2_inner),
        mp.Vector3(0.5 * Lw + 0.5 * Lt, -0.5 * w2_inner),
        mp.Vector3(0.5 * Lt, -0.5 * w2_inner),
        mp.Vector3(-0.5 * Lt, -0.5 * w1_inner),
        mp.Vector3(-0.5 * Lw - 0.5 * Lt, -0.5 * w1_inner),
    ]
    hollow_core = mp.Prism(
        inner_vertices, 
        height=prism_height, 
        material=mp.air
    )

    geometry = [solid_taper, hollow_core]

    # ==============================================================================
    # 2. Create a 3D Simulation object just for plotting
    # ==============================================================================
    
    # Define a 3D cell large enough to contain the geometry
    sx = Lw + Lt + 4
    sy = w2 + 4
    sz = prism_height + 2
    cell_size = mp.Vector3(sx, sy, sz)

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        geometry=geometry,
        dimensions=3  # Ensure simulation is 3D
    )

    # ==============================================================================
    # 3. Generate and display the plot
    # ==============================================================================
    print("--- Launching interactive 3D plot... ---")
    print("You can rotate the object with your mouse.")
    sim.plot3D()

if __name__ == '__main__':
    plot_hollow_taper_3d()
