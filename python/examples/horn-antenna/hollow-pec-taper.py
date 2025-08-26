import meep as mp
import matplotlib.pyplot as plt
import math

def plot_hollow_taper_2d():
    """
    Creates and displays a 2D cross-section of the hollow, cylindrical,
    and conically tapered PEC waveguide using cylindrical coordinates.
    """
    # ==============================================================================
    # 1. Define the geometry parameters (units in mm)
    # ==============================================================================
    resolution = 30  # pixels/mm

    d1 = 1.0  # diameter of input waveguide
    d2 = 2.0  # diameter of output waveguide
    Lw = 10.0 # length of straight waveguide sections
    wall_thickness = 0.2
    Lt = 8.0 
    
    pec_material = mp.metal

    # --- Define Simulation Cell in Cylindrical Coordinates (r, z) ---
    # The cell is a 2D plane. The r-dimension goes from 0 to r_size.
    # The z-dimension is the length along the waveguide axis.
    r_size = d2 / 2 + 2
    z_size = Lw + Lt + Lw
    cell_size = mp.Vector3(r_size, 0, z_size)

    # ==============================================================================
    # 2. Create the 2D Geometry Objects (Cross-sections)
    # ==============================================================================
    
    # Set z=0 at the center of the conical taper for easier reference.
    # The geometry is built by defining the outer solid shape and then
    # subtracting a smaller inner shape made of air.
    
    # --- Outer Solid Structure ---
    # 1. Input cylinder (a rectangle in the r-z plane)
    input_cyl_outer = mp.Block(
        center=mp.Vector3(d1 / 4, 0, -Lt / 2 - Lw / 2),
        size=mp.Vector3(d1 / 2, mp.inf, Lw),
        material=pec_material
    )

    # 2. Conical taper (a trapezoid/Prism in the r-z plane)
    taper_vertices_outer = [
        mp.Vector3(d1 / 2, -Lt / 2),
        mp.Vector3(d2 / 2, Lt / 2),
        mp.Vector3(0, Lt / 2),
        mp.Vector3(0, -Lt / 2)
    ]
    taper_cone_outer = mp.Prism(vertices=taper_vertices_outer, height=mp.inf, material=pec_material)

    # 3. Output cylinder (a rectangle in the r-z plane)
    output_cyl_outer = mp.Block(
        center=mp.Vector3(d2 / 4, 0, Lt / 2 + Lw / 2),
        size=mp.Vector3(d2 / 2, mp.inf, Lw),
        material=pec_material
    )

    # --- Inner Air Structure to create the hollow core ---
    r1_inner = (d1 / 2) - wall_thickness
    r2_inner = (d2 / 2) - wall_thickness

    input_cyl_inner = mp.Block(
        center=mp.Vector3(r1_inner / 2, 0, -Lt / 2 - Lw / 2),
        size=mp.Vector3(r1_inner, mp.inf, Lw),
        material=mp.air
    )

    inner_taper_vertices = [
        mp.Vector3(r1_inner, -Lt / 2),
        mp.Vector3(r2_inner, Lt / 2),
        mp.Vector3(0, Lt / 2),
        mp.Vector3(0, -Lt / 2)
    ]
    taper_cone_inner = mp.Prism(vertices=inner_taper_vertices, height=mp.inf, material=mp.air)

    output_cyl_inner = mp.Block(
        center=mp.Vector3(r2_inner / 2, 0, Lt / 2 + Lw / 2),
        size=mp.Vector3(r2_inner, mp.inf, Lw),
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
    # 3. Create a 2D Simulation object for plotting
    # ==============================================================================
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        geometry=geometry,
        dimensions=mp.CYLINDRICAL
    )

    # ==============================================================================
    # 4. Generate and display the plot
    # ==============================================================================
    plt.figure(dpi=150)
    sim.plot2D()
    plt.title("2D Cross-Section of Hollow Taper (r-z plane)")
    plt.xlabel("Radial Position r (mm)")
    plt.ylabel("Axial Position z (mm)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    plot_hollow_taper_2d()
