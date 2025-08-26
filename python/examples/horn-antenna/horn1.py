import meep as mp
import math

def plot_horn_antenna_3d():
    """
    Creates and displays an interactive 3D plot of the
    conical coaxial horn antenna from the provided diagram.
    """
    # ==============================================================================
    # 1. Define the geometry parameters based on the diagram (all units in mm)
    # ==============================================================================
    resolution = 15  # pixels/mm. A lower value runs faster for visualization.

    # --- Materials ---
    pec_material = mp.metal  # Perfect Electric Conductor
    dielectric_material = mp.Medium(epsilon=1.3)

    # --- Overall Dimensions ---
    horn_outer_radius = 250 / 2
    horn_height = 88.7
    dielectric_height = 17
    
    # --- Conical Horn Dimensions (a2) ---
    cone_top_radius = horn_outer_radius
    cone_bottom_radius = 86.3 / 2
    cone_height = 22.5

    # --- Coaxial Feed Dimensions (b2) ---
    coax_outer_radius = 4.1 / 2
    coax_inner_radius = 1.27 / 2
    coax_height = 10.0

    # --- Inner Conical Section (b) ---
    inner_cone_top_radius = 65 / 2
    inner_cone_bottom_radius = 27.3 / 2
    inner_cone_height = 119 - 36.2 - 10 # Approximate height from diagram

    # ==============================================================================
    # 2. Create the Geometry Objects
    # ==============================================================================
    
    # --- Main Horn Body (PEC) ---
    # 1. The main outer cone
    main_horn = mp.Cone(
        center=mp.Vector3(0, 0, 0),
        radius=cone_bottom_radius,
        radius2=cone_top_radius,
        height=cone_height,
        material=pec_material
    )
    
    # 2. The lower cylindrical part of the horn
    lower_cyl = mp.Cylinder(
        center=mp.Vector3(0, 0, -cone_height/2 - (horn_height-cone_height)/2),
        radius=cone_bottom_radius,
        height=horn_height-cone_height,
        material=pec_material
    )

    # --- Inner Hollow Structure (Air) ---
    # 3. The inner cone that is hollowed out
    inner_cone = mp.Cone(
        center=mp.Vector3(0, 0, -3.8), # Approximate center from diagram
        radius=inner_cone_bottom_radius,
        radius2=inner_cone_top_radius,
        height=inner_cone_height,
        material=mp.air
    )

    # --- Coaxial Feed (PEC) ---
    # 4. The outer part of the coaxial feed
    coax_outer = mp.Cylinder(
        center=mp.Vector3(0, 0, -119 + coax_height/2),
        radius=coax_outer_radius,
        height=coax_height,
        material=pec_material
    )
    
    # 5. The inner part of the coaxial feed (hollowed out)
    coax_inner = mp.Cylinder(
        center=mp.Vector3(0, 0, -119 + coax_height/2),
        radius=coax_inner_radius,
        height=coax_height,
        material=mp.air
    )

    # --- Dielectric Support Slab (Green part) ---
    # 6. The main dielectric disk
    dielectric_slab = mp.Cylinder(
        center=mp.Vector3(0, 0, cone_height/2 + dielectric_height/2),
        radius=horn_outer_radius,
        height=dielectric_height,
        material=dielectric_material
    )
    
    # 7. Create holes in the dielectric slab
    # The diagram shows 8 holes, let's place them in a circle
    hole_radius = 1.5 # M3 screw clearance hole is ~3.2mm, so radius ~1.6
    holes_circle_radius = (163 / 2) * 0.8 # Approximate radius from diagram
    holes = []
    for i in range(8):
        angle = 2 * math.pi * i / 8
        hole_center = mp.Vector3(
            holes_circle_radius * math.cos(angle),
            holes_circle_radius * math.sin(angle),
            dielectric_slab.center.z
        )
        holes.append(
            mp.Cylinder(
                center=hole_center,
                radius=hole_radius,
                height=dielectric_height,
                material=mp.air
            )
        )

    geometry = [main_horn, lower_cyl, inner_cone, coax_outer, coax_inner, dielectric_slab] + holes

    # ==============================================================================
    # 3. Create a 3D Simulation object just for plotting
    # ==============================================================================
    
    # Define a 3D cell large enough to contain the geometry
    cell_size = mp.Vector3(2 * horn_outer_radius + 20, 
                           2 * horn_outer_radius + 20, 
                           cone_height + dielectric_height + 120)

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        geometry=geometry,
        dimensions=3
    )

    # ==============================================================================
    # 4. Generate and display the plot
    # ==============================================================================
    print("--- Launching interactive 3D plot... ---")
    print("You can rotate the object with your mouse.")
    sim.plot3D()

if __name__ == '__main__':
    plot_horn_antenna_3d()
