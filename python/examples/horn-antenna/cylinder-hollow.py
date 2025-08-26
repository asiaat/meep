import meep as mp
import matplotlib.pyplot as plt
import math

def plot_hollow_cylinder_and_cone_3d():
    """
    Creates and displays an interactive 3D plot of a hollow metal cylinder
    joined with a hollow metal cone.
    """
    # ==============================================================================
    # 1. Define the geometry parameters (units in mm)
    # ==============================================================================
    resolution = 25  # pixels/mm

    # Cylinder parameters
    cyl_height = 10.0
    cyl_outer_diameter = 4.0
    cyl_inner_diameter = 1.3
    
    # Cone parameters
    cone_height = 6.0
    cone_bottom_diameter = 4.0
    cone_top_diameter = 8.0
    cone_hollow_bottom_diameter = 2.0
    cone_hollow_top_diameter = 3.0

    # Use a medium with a large negative epsilon to represent PEC for visualization
    pec_material = mp.Medium(epsilon=-1000)

    # --- GEOMETRY DEFINITION ---
    # Set z=0 as the junction between the cylinder and the cone.
    
    # --- CYLINDER (z < 0) ---
    cyl_outer_radius = cyl_outer_diameter / 2
    cyl_inner_radius = cyl_inner_diameter / 2
    
    outer_cylinder = mp.Cylinder(
        center=mp.Vector3(0, 0, -cyl_height / 2),
        radius=cyl_outer_radius,
        height=cyl_height,
        axis=mp.Vector3(0, 0, 1),
        material=pec_material
    )
    
    inner_cylinder = mp.Cylinder(
        center=mp.Vector3(0, 0, -cyl_height / 2),
        radius=cyl_inner_radius,
        height=cyl_height,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    # --- CONE (z > 0) ---
    cone_bottom_radius = cone_bottom_diameter / 2
    cone_top_radius = cone_top_diameter / 2
    
    outer_cone = mp.Cone(
        center=mp.Vector3(0, 0, cone_height / 2),
        radius=cone_bottom_radius,
        radius2=cone_top_radius,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        material=pec_material
    )

    hollow_bottom_radius = cone_hollow_bottom_diameter / 2
    hollow_top_radius = cone_hollow_top_diameter / 2
    
    inner_cone = mp.Cone(
        center=mp.Vector3(0, 0, cone_height / 2),
        radius=hollow_bottom_radius,
        radius2=hollow_top_radius,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    geometry = [outer_cylinder, inner_cylinder, outer_cone, inner_cone]

    # ==============================================================================
    # 2. Create a 3D Simulation object for plotting
    # ==============================================================================
    
    # Define a 3D cell large enough to contain the entire object
    sx = cone_top_diameter + 4
    sy = cone_top_diameter + 4
    sz = cyl_height + cone_height + 4
    cell_size = mp.Vector3(sx, sy, sz)

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        dimensions=3
    )

    # ==============================================================================
    # 3. Generate and display the plot
    # ==============================================================================
    print("--- Launching 3D geometry plot of hollow cylinder and cone ---")
    sim.plot3D()

if __name__ == '__main__':
    plot_hollow_cylinder_and_cone_3d()
