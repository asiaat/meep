import meep as mp
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Define parameters for a single case
# These should match the parameters in your main simulation script.
# ==============================================================================
resolution = 25  # pixels/μm

w1 = 1.0  # width of waveguide 1
w2 = 2.0  # width of waveguide 2
Lw = 10.0 # length of straight waveguide sections

# We'll visualize the setup for a longer taper length to see the geometry clearly
Lt = 8.0 

dair = 3.0    # length of air region
dpml_x = 6.0  # length of PML in x direction
dpml_y = 2.0  # length of PML in y direction

sy = dpml_y + dair + w2 + dair + dpml_y
sx = dpml_x + Lw + Lt + Lw + dpml_x
cell_size = mp.Vector3(sx, sy, 0)

Si = mp.Medium(epsilon=12.0)
lcen = 6.67  # mode wavelength
fcen = 1 / lcen  # mode frequency

# ==============================================================================
# 2. Define the simulation components: boundaries, sources, and geometry
# ==============================================================================
boundary_layers = [mp.PML(dpml_x, direction=mp.X), mp.PML(dpml_y, direction=mp.Y)]
symmetries = [mp.Mirror(mp.Y)]

# Define the source
src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=src_pt,
        size=mp.Vector3(y=sy - 2 * dpml_y),
        eig_match_freq=True,
        eig_parity=mp.ODD_Z + mp.EVEN_Y,
    )
]

# --- MODIFIED GEOMETRY ---
# First, define the outer solid tapered waveguide
outer_vertices = [
    mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
    mp.Vector3(-0.5 * Lt, 0.5 * w1),
    mp.Vector3(0.5 * Lt, 0.5 * w2),
    mp.Vector3(0.5 * sx + 1, 0.5 * w2),
    mp.Vector3(0.5 * sx + 1, -0.5 * w2),
    mp.Vector3(0.5 * Lt, -0.5 * w2),
    mp.Vector3(-0.5 * Lt, -0.5 * w1),
    mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
]
solid_taper = mp.Prism(outer_vertices, height=mp.inf, material=Si)

# Second, define a slightly smaller inner prism of air to create the hollow core
wall_thickness = 0.2
w1_inner = w1 - 2 * wall_thickness
w2_inner = w2 - 2 * wall_thickness

inner_vertices = [
    mp.Vector3(-0.5 * sx - 1, 0.5 * w1_inner),
    mp.Vector3(-0.5 * Lt, 0.5 * w1_inner),
    mp.Vector3(0.5 * Lt, 0.5 * w2_inner),
    mp.Vector3(0.5 * sx + 1, 0.5 * w2_inner),
    mp.Vector3(0.5 * sx + 1, -0.5 * w2_inner),
    mp.Vector3(0.5 * Lt, -0.5 * w2_inner),
    mp.Vector3(-0.5 * Lt, -0.5 * w1_inner),
    mp.Vector3(-0.5 * sx - 1, -0.5 * w1_inner),
]
hollow_core = mp.Prism(inner_vertices, height=mp.inf, material=mp.air)


# The final geometry is the combination of the solid part and the air hole
geometry = [solid_taper, hollow_core]

# ==============================================================================
# 3. Create the Simulation object
# ==============================================================================
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    geometry=geometry,
    sources=sources,
    symmetries=symmetries,
)

# ==============================================================================
# 4. Add the flux monitor so it will be visible on the plot
# ==============================================================================
mon_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.7 * Lw)
flux_region = mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
flux_obj = sim.add_flux(fcen, 0, 1, flux_region)

# ==============================================================================
# 5. Generate and display the plot
# ==============================================================================
if __name__ == '__main__':
    plt.figure(dpi=150)
    sim.plot2D()
    plt.title(f"Setup with Hollow Tapered Waveguide (Lt = {Lt} μm)")
    plt.show()
