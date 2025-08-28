import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def make_geometry(r, h, cthik, metal=True, n_cyl=2.0):
    """Return geometry list for cylinder."""
    if metal:
        material = mp.metal
    else:
        material = mp.Medium(index=n_cyl)

    geometry = [
        mp.Block(
            material=material,
            center=mp.Vector3(cthik * r),
            size=mp.Vector3(r, 0, h),
        )
    ]
    return geometry


def make_simulation(r, h, cthik, resolution=25, metal=False):
    """Build simulation object (with optional scatterer)."""
    wvl_min = 2 * np.pi * r / 10
    wvl_max = 2 * np.pi * r / 2
    frq_min = 1 / wvl_max
    frq_max = 1 / wvl_min
    frq_cen = 0.5 * (frq_min + frq_max)
    dfrq = frq_max - frq_min
    nfrq = 100

    dpml = cthik * wvl_max
    dair = 1.0 * wvl_max

    pml_layers = [mp.PML(thickness=dpml)]
    sr = r + dair + dpml
    sz = dpml + dair + h + dair + dpml
    cell_size = mp.Vector3(sr, 0, sz)

    # circularly polarized source
    sources = [
        mp.Source(
            mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
            component=mp.Er,
            center=mp.Vector3(cthik * sr, 0, -0.5 * sz + dpml),
            size=mp.Vector3(sr),
        ),
        mp.Source(
            mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
            component=mp.Ep,
            center=mp.Vector3(cthik * sr, 0, -0.5 * sz + dpml),
            size=mp.Vector3(sr),
            amplitude=-1j,
        ),
    ]

    geometry = make_geometry(r, h, cthik, metal=metal) if metal else None

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        force_complex_fields=True,  # correct keyword
    )

    return sim, frq_cen, dfrq, nfrq


def add_flux_monitors(sim, r, h, cthik, frq_cen, dfrq, nfrq):
    """Attach flux regions and return handles."""
    box_z1 = sim.add_flux(
        frq_cen, dfrq, nfrq,
        mp.FluxRegion(center=mp.Vector3(cthik * r, 0, -0.5 * h), size=mp.Vector3(r))
    )
    box_z2 = sim.add_flux(
        frq_cen, dfrq, nfrq,
        mp.FluxRegion(center=mp.Vector3(cthik * r, 0, +0.5 * h), size=mp.Vector3(r))
    )
    box_r = sim.add_flux(
        frq_cen, dfrq, nfrq,
        mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h))
    )
    return box_z1, box_z2, box_r


def add_flux_region(sim, h, frq_cen, dfrq, nfrq):
    """Add a single flux region at the bottom for reflection measurement."""
    flux_region = sim.add_flux(
        frq_cen, dfrq, nfrq,
        mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*h), size=mp.Vector3(sim.cell_size.x, 0, 0))
    )
    return flux_region


def run_reference(r, h, cthik):
    """Run reference simulation without scatterer and return flux data."""
    sim, frq_cen, dfrq, nfrq = make_simulation(r, h, cthik, metal=False)
    box_z1, box_z2, box_r = add_flux_monitors(sim, r, h, cthik, frq_cen, dfrq, nfrq)

    sim.run(until_after_sources=10)
    freqs = mp.get_flux_freqs(box_z1)
    box_z1_data = sim.get_flux_data(box_z1)
    box_z2_data = sim.get_flux_data(box_z2)
    box_r_data = sim.get_flux_data(box_r)
    box_z1_flux0 = mp.get_fluxes(box_z1)

    return freqs, box_z1_data, box_z2_data, box_r_data, box_z1_flux0


def run_scatterer(r, h, cthik, box_z1_data, box_z2_data, box_r_data, metal=True):
    """Run scatterer simulation (metal or dielectric) and return fluxes."""
    sim, frq_cen, dfrq, nfrq = make_simulation(r, h, cthik, metal=metal)
    box_z1, box_z2, box_r = add_flux_monitors(sim, r, h, cthik, frq_cen, dfrq, nfrq)

    # load reference fields
    sim.load_minus_flux_data(box_z1, box_z1_data)
    sim.load_minus_flux_data(box_z2, box_z2_data)
    sim.load_minus_flux_data(box_r, box_r_data)

    sim.run(until_after_sources=100)

    return mp.get_fluxes(box_z1), mp.get_fluxes(box_z2), mp.get_fluxes(box_r)


def analyze_fluxes(freqs, refl0, refl, tran, side, r):
    """Compute scattering cross-section and R/T/A coefficients."""
    scatt_flux = np.asarray(refl) - np.asarray(tran) - np.asarray(side)
    intensity = np.asarray(refl0) / (np.pi * r**2)
    scatt_cs = np.divide(-scatt_flux, intensity)

    R = np.asarray(refl) / np.asarray(refl0)
    T = np.asarray(tran) / np.asarray(refl0)
    A = 1 - R - T  # absorption

    return scatt_cs, R, T, A


def plot_scattering(freqs, scatt_cs):
    plt.figure()
    plt.plot(freqs, scatt_cs, label="Scattering σ")
    plt.xlabel("Frequency")
    plt.ylabel("σ_scat")
    plt.legend()
    plt.grid(True)


def plot_rta(freqs, R, T, A):
    plt.figure()
    plt.plot(freqs, R, label="Reflection")
    plt.plot(freqs, T, label="Transmission")
    plt.plot(freqs, A, label="Absorption")
    plt.xlabel("Frequency")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.grid(True)


def run_with_animation(r, h, cthik, metal=False, field_component=mp.Ez,
                       until=200, frames=20):
    """Run simulation and animate field evolution in cylindrical coordinates."""

    # Build simulation
    sim, frq_cen, dfrq, nfrq = make_simulation(r, h, cthik, metal=metal)

    # Flux monitor
    refl_fr = add_flux_region(sim, h, frq_cen, dfrq, nfrq)

    # Run short simulation to initialize fields
    sim.run(until=20)

    # Get initial field snapshot
    field_data = sim.get_array(
        component=field_component,
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(sim.cell_size.x, sim.cell_size.z)
    )
    if field_data.ndim == 1:
        field_data = field_data[:, np.newaxis]

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f"Field evolution: {field_component}")
    ax.set_xlabel("r (a.u.)")
    ax.set_ylabel("z (a.u.)")

    im = ax.imshow(
        np.rot90(np.abs(field_data)),  # magnitude of complex field
        interpolation="spline36",
        cmap="RdBu",
        extent=[0, sim.cell_size.x, -0.5 * sim.cell_size.z, 0.5 * sim.cell_size.z]
    )

    # Update function
    def update(frame):
        sim.run(until=(frame + 1) * until / frames)
        field_data = sim.get_array(
            component=field_component,
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(sim.cell_size.x, sim.cell_size.z)
        )
        if field_data.ndim == 1:
            field_data = field_data[:, np.newaxis]
        im.set_array(np.rot90(np.abs(field_data)))
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, blit=True, repeat=False
    )
    plt.show()

    # Reflection spectrum
    freqs = mp.get_flux_freqs(refl_fr)
    refl = np.array(mp.get_fluxes(refl_fr))

    plt.figure()
    plt.plot(freqs, refl, "b-")
    plt.xlabel("Frequency (a / λ)")
    plt.ylabel("Reflected Flux")
    plt.title("Reflection Spectrum")
    plt.grid(True)
    plt.show()

    return freqs, refl


def main():
    # parameters
    r, h, cthik = .7, 2.3, 0.5

    # reference run
    freqs, box_z1_data, box_z2_data, box_r_data, refl0 = run_reference(r, h, cthik)

    # scatterer run (dielectric)
    refl, tran, side = run_scatterer(r, h, cthik, box_z1_data, box_z2_data, box_r_data, metal=False)

    # analysis
    scatt_cs, R, T, A = analyze_fluxes(freqs, refl0, refl, tran, side, r)

    # plots
    plot_scattering(freqs, scatt_cs)
    # plot_rta(freqs, R, T, A)

    # animation
    run_with_animation(r, h, cthik, metal=False, field_component=mp.Ez)

    plt.show()


if __name__ == "__main__":
    main()
