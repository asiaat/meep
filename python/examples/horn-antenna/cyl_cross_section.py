import matplotlib.pyplot as plt
import numpy as np

import meep as mp

def plot_ring_schematic(r, w, sr, height=1):
    """
    Plot both a 2D polar (top-down) and longitudinal cross-section view
    of the ring resonator.
    
    r      : inner radius of ring
    w      : width of the ring
    sr     : outer boundary of computational cell
    height : arbitrary vertical thickness for schematic
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Polar (top-down) view ---
    theta = np.linspace(0, 2*np.pi, 500)
    axs[0] = plt.subplot(1, 2, 1, projection='polar')
    axs[0].set_theta_zero_location("N")
    axs[0].set_theta_direction(-1)
    axs[0].fill_between(theta, r, r + w, color='skyblue', alpha=0.7, label='Ring waveguide')
    axs[0].plot(theta, sr*np.ones_like(theta), 'k--', label='Computational boundary')
    axs[0].set_rmax(sr + 1)
    axs[0].set_rticks([0, r, r + w, sr])
    axs[0].set_rlabel_position(-22.5)
    axs[0].set_title("Top-down (polar) view")
    axs[0].legend(loc='upper right')

    # --- Longitudinal (cross-section) view ---
    axs[1].set_aspect('equal')
    axs[1].set_xlim(0, sr + 5)
    axs[1].set_ylim(0, height + 0.5)
    axs[1].add_patch(plt.Rectangle((r, 0), w, height, color='skyblue', alpha=0.7))
    axs[1].axvline(sr, color='k', linestyle='--', label='Computational boundary')
    axs[1].set_xlabel("Radius r")
    axs[1].set_ylabel("Height z")
    axs[1].set_title("Longitudinal cross-section")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

r = 0.7  # radius of cylinder
h = 2.3  # height of cylinder

#KO
w = 1

wvl_min = 2 * np.pi * r / 10
wvl_max = 2 * np.pi * r / 2

frq_min = 1 / wvl_max
frq_max = 1 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)
dfrq = frq_max - frq_min
nfrq = 100

## at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
resolution = 25

dpml = 0.5 * wvl_max
dair = 1.0 * wvl_max

pml_layers = [mp.PML(thickness=dpml)]

sr = r + dair + dpml
sz = dpml + dair + h + dair + dpml
cell_size = mp.Vector3(sr, 0, sz)

sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        component=mp.Er,
        center=mp.Vector3(0.5 * sr, 0, -0.5 * sz + dpml),
        size=mp.Vector3(sr),
    ),
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        component=mp.Ep,
        center=mp.Vector3(0.5 * sr, 0, -0.5 * sz + dpml),
        size=mp.Vector3(sr),
        amplitude=-1j,
    ),
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

box_z1 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, -0.5 * h), size=mp.Vector3(r)),
)
box_z2 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, +0.5 * h), size=mp.Vector3(r)),
)
box_r = sim.add_flux(
    frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h))
)

sim.run(until_after_sources=10)

freqs = mp.get_flux_freqs(box_z1)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)
box_r_data = sim.get_flux_data(box_r)

box_z1_flux0 = mp.get_fluxes(box_z1)

sim.reset_meep()

n_cyl = 2.0
geometry = [
    mp.Block(
        material=mp.Medium(index=n_cyl),
        center=mp.Vector3(0.5 * r),
        size=mp.Vector3(r, 0, h),
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

box_z1 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, -0.5 * h), size=mp.Vector3(r)),
)
box_z2 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, +0.5 * h), size=mp.Vector3(r)),
)
box_r = sim.add_flux(
    frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h))
)

sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
sim.load_minus_flux_data(box_r, box_r_data)

sim.run(until_after_sources=100)

box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)
box_r_flux = mp.get_fluxes(box_r)

scatt_flux = np.asarray(box_z1_flux) - np.asarray(box_z2_flux) - np.asarray(box_r_flux)
intensity = np.asarray(box_z1_flux0) / (np.pi * r**2)
scatt_cross_section = np.divide(-scatt_flux, intensity)

#plot_ring_schematic(r,w,sr)


if mp.am_master():
    plt.figure(dpi=150)
    plt.loglog(2 * np.pi * r * np.asarray(freqs), scatt_cross_section, "bo-")
    plt.grid(True, which="both", ls="-")
    plt.xlabel("(cylinder circumference)/wavelength, 2πr/λ")
    plt.ylabel("scattering cross section, σ")
    plt.title("Scattering Cross Section of a Lossless Dielectric Cylinder")
    plt.tight_layout()
    plt.savefig("cylinder_cross_section.png")
