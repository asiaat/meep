import meep as mp
import math

def plot_horn_feed_3d():
    """
    Creates and displays an interactive 3D plot of the coaxial feed
    and its junction with the conical horn, based on detail (b2).
    """
    # ==============================================================================
    # 1. Define the geometry parameters from the diagram (units in mm)
    # ==============================================================================
    resolution = 30  # pixels/mm

    # --- Materials ---
    pec_material = mp.metal
    air_material = mp.air

    # --- Coaxial Feed Dimensions (from detail b2) ---
    coax_height = 10.0
    coax_outer_radius = 4.1 / 2
    coax_inner_radius = 1.27 / 2

    # --- Horn Base Dimensions (from detail b) ---
    # We will model a small section of the horn base where the coax connects.
    horn_base_height = 15.0
    horn_base_outer_radius = 27.3 / 2 # The radius of the solid part at the junction
    
    # The inner conical hole that flares outwards
    hole_bottom_radius = 2.0 # From detail b1, the hole starts small
    hole_top_radius = 6.0 # An estimated radius after a short distance

    # ==============================================================================
    # 2. Create the Geometry Objects
    # ==============================================================================
    
    # Set z=0 as the junction point between the coax and the horn base.
    
    # --- Coaxial Feed (Cylinder) ---
    # Positioned below the horn base (from z=-10 to z=0)
    coax_outer = mp.Cylinder(
        center=mp.Vector3(0, 0, -coax_height / 2),
        radius=coax_outer_radius,
        height=coax_height,
        material=pec_material
    )
    
    coax_inner = mp.Cylinder(
        center=mp.Vector3(0, 0, -coax_height / 2),
        radius=coax_inner_radius,
        height=coax_height,
        material=air_material
    )

    # --- Horn Base (Cone-Horn Joint) ---
    # Positioned above the coax (from z=0 to z=15)
    horn_base_solid = mp.Cylinder(
        center=mp.Vector3(0, 0, horn_base_height / 2),
        radius=horn_base_outer_radius,
        height=horn_base_height,
        material=pec_material
    )

    # The conical hole inside the horn base
    horn_hole_cone = mp.Cone(
        center=mp.Vector3(0, 0, horn_base_height / 2),
        radius=hole_bottom_radius,
        radius2=hole_top_radius,
        height=horn_base_height,
        material=air_material
    )

    geometry = [
        coax_outer,
        coax_inner,
        horn_base_solid,
        horn_hole_cone
    ]

    # ==============================================================================
    # 3. Create a 3D Simulation object just for plotting
    # ==============================================================================
    
    # Define a 3D cell large enough to contain the geometry
    cell_size = mp.Vector3(2 * horn_base_outer_radius + 10, 
                           2 * horn_base_outer_radius + 10, 
                           coax_height + horn_base_height + 10)

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        geometry=geometry,
        dimensions=3
    )

    # ==============================================================================
    # 4. Generate and display the plot
    # ==============================================================================
    print("--- Launching interactive 3D plot of the feed junction... ---")
    print("You can rotate the object with your mouse.")
    sim.plot3D()

if __name__ == '__main__':
    plot_horn_feed_3d()
