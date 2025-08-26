import meep as mp
import matplotlib.pyplot as plt
import math

def plot_hollow_taper_3d():
    """
    Creates and displays an interactive 3D plot of a hollow,
    cylindrical, and conically tapered waveguide geometry.
    """
    # ==============================================================================
    # 1. Define the geometry parameters
    # ==============================================================================
    resolution = 25  # pixels/μm

    d1 = 1.0  # diameter of input waveguide
    d2 = 2.0  # diameter of output waveguide
    Lw = 10.0 # length of straight waveguide sections
    wall_thickness = 0.2 # Thickness of the hollow waveguide walls
    
    # Use a fixed taper length for visualization
    Lt = 8.0 
    
    # Define a finite height for the cell in the z-direction for plotting
    z_height = 1.5

    # Define the dielectric material
    Si = mp.Medium(epsilon=12.0)

    # --- NEW CYLINDRICAL GEOMETRY ---
    
    # --- Outer Solid Structure ---
    # 1. Input cylinder
    input_cyl_outer = mp.Cylinder(
        center=mp.Vector3(-0.5 * Lt - 0.5 * Lw, 0, 0),
        radius=d1 / 2,
        height=Lw,  # Corrected: Length of the cylinder
        axis=mp.Vector3(x=1),
        material=Si
    )

    # 2. Conical taper
    taper_cone_outer = mp.Cone(
        center=mp.Vector3(0, 0, 0),
        radius=d1 / 2,
        radius2=d2 / 2,
        height=Lt,  # Correct: Length of the cone
        axis=mp.Vector3(x=1),
        material=Si
    )

    # 3. Output cylinder
    output_cyl_outer = mp.Cylinder(
        center=mp.Vector3(0.5 * Lt + 0.5 * Lw, 0, 0),
        radius=d2 / 2,
        height=Lw,  # Corrected: Length of the cylinder
        axis=mp.Vector3(x=1),
        material=Si
    )

    # --- Inner Air Structure to create the hollow core ---
    r1_inner = (d1 / 2) - wall_thickness
    r2_inner = (d2 / 2) - wall_thickness

    # 4. Input cylinder (air)
    input_cyl_inner = mp.Cylinder(
        center=mp.Vector3(-0.5 * Lt - 0.5 * Lw, 0, 0),
        radius=r1_inner,
        height=Lw, # Corrected: Length of the cylinder
        axis=mp.Vector3(x=1),
        material=mp.air
    )

    # 5. Conical taper (air)
    taper_cone_inner = mp.Cone(
        center=mp.Vector3(0, 0, 0),
        radius=r1_inner,
        radius2=r2_inner,
        height=Lt, # Correct: Length of the cone
        axis=mp.Vector3(x=1),
        material=mp.air
    )

    # 6. Output cylinder (air)
    output_cyl_inner = mp.Cylinder(
        center=mp.Vector3(0.5 * Lt + 0.5 * Lw, 0, 0),
        radius=r2_inner,
        height=Lw, # Corrected: Length of the cylinder
        axis=mp.Vector3(x=1),
        material=mp.air
    )

    geometry = [
        input_cyl_outer,
        taper_cone_outer,
        output_cyl_outer,
        input_cyl_inner,
        taper_cone_inner,
        output_cyl_inner,
    ]

    # ==============================================================================
    # 2. Create a 3D Simulation object just for plotting
    # ==============================================================================
    
    # Define a 3D cell large enough to contain the geometry
    sx = Lw + Lt + 4
    sy = d2 + 4
    sz = z_height + 2
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
