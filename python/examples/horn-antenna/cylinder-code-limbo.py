import meep as mp

def plot_hollow_metal_cylinder_with_cone():
    """
    Creates and displays a 3D plot of a hollow metal cylinder
    with a hollow conical extension on top.
    """
    # Cylinder parameters (mm)
    outer_radius = 2.0    # outer radius of cylinder
    inner_radius = 0.65   # hollow radius of cylinder
    height_mm = 10.0      # height of cylinder

    # Cone parameters (mm)
    cone_height = 160.0
    cone_outer_r1 = 2.0   # bottom outer radius (matches cylinder outer)
    cone_outer_r2 = 80.0   # top outer radius
    cone_inner_r1 = 1.0   # bottom inner radius (hollow)
    cone_inner_r2 = 40.0   # top inner radius (hollow)

    resolution = 5  # pixels/mm

    # ------------------ Geometry ------------------
    # Cylinder outer (metal replaced with finite epsilon for visualization)
    outer_cyl = mp.Cylinder(
        radius=outer_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.Medium(epsilon=10.0)   # use mp.metal for real sim
    )

    # Cylinder inner (air hollow)
    inner_cyl = mp.Cylinder(
        radius=inner_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    # Cone outer (metal)
    outer_cone = mp.Cone(
        radius=cone_outer_r1,
        radius2=cone_outer_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.Medium(epsilon=10.0)   # use mp.metal for real sim
    )

    # Cone inner (air hollow)
    inner_cone = mp.Cone(
        radius=cone_inner_r1,
        radius2=cone_inner_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.air
    )

    geometry = [outer_cyl, inner_cyl, outer_cone, inner_cone]

    # ------------------ Simulation cell ------------------
    sx = 2 * cone_outer_r2 + 4
    sy = 2 * cone_outer_r2 + 4
    sz = height_mm + cone_height + 6
    cell_size = mp.Vector3(sx, sy, sz)

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        dimensions=3
    )

    # ------------------ Plot ------------------
    print("--- Launching 3D geometry plot of hollow cylinder + cone ---")
    sim.plot3D(isosurface=5.0)  # pick an isosurface between air (1) and metal (10)

if __name__ == '__main__':
    plot_hollow_metal_cylinder_with_cone()
