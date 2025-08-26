import meep as mp

def plot_hollow_metal_cylinder():
    """
    Creates and displays a 3D plot of a hollow metal cylinder geometry
    (outer diameter 4.0 mm, inner hollow diameter 1.3 mm, height 10 mm).
    """
    # Parameters (mm)
    outer_radius = 2.0    # mm (outer radius)
    inner_radius = 0.65   # mm (hollow radius)
    height_mm = 10.0      # mm (height)
    resolution = 25       # pixels/mm for plotting

    # Outer metal cylinder
    outer_cyl = mp.Cylinder(
        radius=outer_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.metal
    )

    # Inner air cylinder (to hollow it out)
    inner_cyl = mp.Cylinder(
        radius=inner_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    geometry = [outer_cyl, inner_cyl]

    # Define simulation cell (just large enough to contain cylinder + padding)
    sx = 2 * outer_radius + 4
    sy = 2 * outer_radius + 4
    sz = height_mm + 4
    cell_size = mp.Vector3(sx, sy, sz)

    
    
    # Use perfect metal for simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=[
        mp.Cylinder(radius=outer_radius, height=height_mm, material=mp.Medium(epsilon=10)),
        mp.Cylinder(radius=inner_radius, height=height_mm, material=mp.air)
        ],
        resolution=resolution,
        dimensions=3
    )

    # Plot
    print("--- Launching 3D geometry plot of hollow cylinder ---")
    sim.plot3D()

if __name__ == '__main__':
    plot_hollow_metal_cylinder()
