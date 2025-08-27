import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# This script models the 2D cross-section of a coaxial feed horn antenna.
# The geometry is based on the user-provided image and is created using
# cylindrical coordinates (r, z) in a 2D simulation.

# --- 1. Set up Simulation Parameters ---

# Resolution of the simulation grid (pixels per distance unit)
resolution = 20

# Dimensions of the computational cell
# We need enough space for the antenna and the PML layers.
cell_z = 12  # Height of the cell
cell_r = 6   # Radius of the cell (since it's cylindrical)
cell_size = mp.Vector3(z=cell_z, y=cell_r) # In Meep 2D cylindrical, y corresponds to r

# Perfect Matched Layers (PML) to absorb outgoing waves
pml_layers = [mp.PML(thickness=1.0)]

# --- 2. Define Geometric Parameters ---
# These dimensions are chosen to create a structure similar to the image.

# Coaxial Cable Base
coax_base_z = -4.0   # Starting z-position of the coaxial base
coax_length = 4.0    # Length of the coaxial section
outer_cyl_r = 1.5    # Radius of the outer metal cylinder
inner_cyl_r = 0.75   # Radius of the inner hollow cylinder
dielectric_r = 1.45  # Radius of the dielectric material between cylinders

# Horn Section
horn_start_z = coax_base_z + coax_length # z-position where the horn starts
horn_length = 5.0    # Length of the horn section
horn_r1_outer = outer_cyl_r # Starting outer radius of the horn
horn_r2_outer = 4.0         # Ending outer radius of the horn aperture
horn_wall_thickness = 0.1   # Thickness of the metal horn wall

# Calculate inner radii for the hollow cone
horn_r1_inner = horn_r1_outer - horn_wall_thickness
horn_r2_inner = horn_r2_outer - horn_wall_thickness


# Material Definitions
metal = mp.Medium(epsilon=mp.inf) # Perfect Electric Conductor (PEC) for metal parts
dielectric = mp.Medium(epsilon=2.25) # Example dielectric (e.g., Teflon)
air = mp.Medium(epsilon=1) # Explicitly define air for carving

# --- 3. Create the Geometry Objects ---

# The geometry is a list of Meep objects.
# We are using a 2D simulation with cylindrical coordinates, so the
# y-dimension is interpreted as the radial coordinate 'r'.
# We define the coaxial section by creating three overlapping solid cylinders.
# The last object in the list takes precedence in overlapping regions.

geometry = [
    # 1. Outer Metal Cylinder (Coaxial Base) - Solid
    # This fills the entire coaxial region with metal initially.
    mp.Block(
        center=mp.Vector3(z=coax_base_z + coax_length / 2),
        size=mp.Vector3(z=coax_length, y=outer_cyl_r, x=mp.inf),
        material=metal,
    ),

    # 2. Dielectric inside the Coaxial Cable - Solid
    # This overwrites the metal in the region r < dielectric_r
    mp.Block(
        center=mp.Vector3(z=coax_base_z + coax_length / 2),
        size=mp.Vector3(z=coax_length, y=dielectric_r, x=mp.inf),
        material=dielectric,
    ),

    # 3. Inner Conductor (PEC) - Solid
    # This overwrites the dielectric in the region r < inner_cyl_r
    mp.Block(
        center=mp.Vector3(z=coax_base_z + coax_length / 2),
        size=mp.Vector3(z=coax_length, y=inner_cyl_r, x=mp.inf),
        material=metal,
    ),

    # 4. Outer Cone (Horn) - Solid Metal
    # This defines the outer boundary of the horn.
    mp.Cone(
        center=mp.Vector3(z=horn_start_z + horn_length / 2),
        height=horn_length,
        radius=horn_r1_outer,
        radius2=horn_r2_outer,
        axis=mp.Vector3(z=1),
        material=metal,
    ),
    
    # 5. Inner Cone (Hollow part) - Solid Air
    # This overwrites the metal cone, creating the hollow interior.
    mp.Cone(
        center=mp.Vector3(z=horn_start_z + horn_length / 2),
        height=horn_length,
        radius=horn_r1_inner,
        radius2=horn_r2_inner,
        axis=mp.Vector3(z=1),
        material=air, # Use air to carve out the metal
    )
]

# --- 4. Set up and Run the Simulation for Visualization ---

# We don't need sources or a long run time just to see the geometry.
# We create the simulation object and run it for 0 time steps.
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    resolution=resolution,
    # For older Meep versions, 'dimensions' must be passed as an argument here.
    dimensions=mp.CYLINDRICAL
)

# Run the simulation for 0 time to initialize the grid and geometry
sim.init_sim()

# --- 5. Plot the Geometry ---

# Use Meep's built-in plot2D function for visualization.
# This is a more robust method for plotting the geometry, especially
# for cylindrical coordinates in older Meep versions.
plt.figure(dpi=150)
sim.plot2D()
plt.show()

