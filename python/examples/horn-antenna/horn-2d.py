import meep as mp
import math
import matplotlib.pyplot as plt

def plot_horn_antenna_2d():
    """
    Creates and displays a 2D plot of the conical horn antenna's cross-section
    using cylindrical coordinates for an efficient simulation setup.
    """
    # ==============================================================================
    # 1. Define the geometry parameters based on the diagram (all units in mm)
    # ==============================================================================
    resolution = 25  # pixels/mm. Higher resolution for a clearer plot.

    # --- Materials ---
    pec_material = mp.metal
    dielectric_material = mp.Medium(epsilon=1.3)

    # --- Key Dimensions from Diagram ---
    horn_outer_radius = 250 / 2
    dielectric_height = 17
    cone_top_radius = horn_outer_radius
    cone_bottom_radius = 86.3 / 2
    cone_height = 22.5
    horn_base_height = 88.7 - cone_height
    coax_outer_radius = 4.1 / 2
    coax_inner_radius = 1.27 / 2
    coax_height = 10.0
    inner_cone_top_radius = 65 / 2
    inner_cone_bottom_radius = 27.3 / 2
    inner_cone_height = 119 - 10.0 - 6.0 # From diagram (b) and (b3)
    
    # --- Define Simulation Cell in Cylindrical Coordinates (r, z) ---
    r_size = horn_outer_radius + 10
    z_size = 119 + dielectric_height + 20
    cell_size = mp.Vector3(r_size, 0, z_size)

    # ==============================================================================
    # 2. Create the 2D Geometry Objects (Cross-sections)
    # ==============================================================================
    
    # Set z=0 at the top surface of the dielectric slab for easier reference
    z_top_dielectric = 0
    
    # --- Main Horn Body (PEC) ---
    # Correctly modeled as a cone (trapezoid) on top of a cylinder (rectangle)
    z_top_cone = z_top_dielectric - dielectric_height
    z_bottom_cone = z_top_cone - cone_height
    
    horn_cone_vertices = [
        mp.Vector3(cone_bottom_radius, z_bottom_cone),
        mp.Vector3(cone_top_radius, z_top_cone),
        mp.Vector3(0, z_top_cone),
        mp.Vector3(0, z_bottom_cone)
    ]
    horn_cone_shape = mp.Prism(vertices=horn_cone_vertices, height=mp.inf, material=pec_material)
    
    horn_base_center_z = z_bottom_cone - horn_base_height / 2
    horn_base_shape = mp.Block(
        center=mp.Vector3(cone_bottom_radius / 2, 0, horn_base_center_z),
        size=mp.Vector3(cone_bottom_radius, mp.inf, horn_base_height),
        material=pec_material
    )

    # --- Inner Hollow Structure (Air) ---
    z_top_inner_cone = z_top_cone - 3.8 - 7.0 # From diagram (b) and (b3)
    z_bottom_inner_cone = z_top_inner_cone - inner_cone_height
    
    inner_cone_vertices = [
        mp.Vector3(inner_cone_bottom_radius, z_bottom_inner_cone),
        mp.Vector3(inner_cone_top_radius, z_top_inner_cone),
        mp.Vector3(0, z_top_inner_cone),
        mp.Vector3(0, z_bottom_inner_cone)
    ]
    inner_cone_air = mp.Prism(vertices=inner_cone_vertices, height=mp.inf, material=mp.air)

    # --- Coaxial Feed (PEC + Air) ---
    coax_center_z = z_bottom_inner_cone - 6.5 - coax_height / 2 # From diagram (b3)
    coax_outer = mp.Block(
        center=mp.Vector3(coax_outer_radius / 2, 0, coax_center_z),
        size=mp.Vector3(coax_outer_radius, mp.inf, coax_height),
        material=pec_material
    )
    coax_inner = mp.Block(
        center=mp.Vector3(coax_inner_radius / 2, 0, coax_center_z),
        size=mp.Vector3(coax_inner_radius, mp.inf, coax_height),
        material=mp.air
    )

    # --- Dielectric Support Slab ---
    dielectric_center_z = z_top_dielectric - dielectric_height / 2
    dielectric_slab = mp.Block(
        center=mp.Vector3(horn_outer_radius / 2, 0, dielectric_center_z),
        size=mp.Vector3(horn_outer_radius, mp.inf, dielectric_height),
        material=dielectric_material
    )
    
    # --- Mounting Hole (one is enough to show in the cross-section) ---
    hole_radius = 1.6
    hole_r_position = 65.2 # From diagram (a3), 163/2 - 49/2
    mounting_hole = mp.Block(
        center=mp.Vector3(hole_r_position, 0, dielectric_center_z),
        size=mp.Vector3(2 * hole_radius, mp.inf, dielectric_height),
        material=mp.air
    )

    geometry = [
        horn_cone_shape,
        horn_base_shape,
        inner_cone_air,
        coax_outer,
        coax_inner,
        dielectric_slab,
        mounting_hole,
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
    plt.title("2D Cross-Section of Horn Antenna (r-z plane)")
    plt.xlabel("Radial Position r (mm)")
    plt.ylabel("Axial Position z (mm)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    plot_horn_antenna_2d()
