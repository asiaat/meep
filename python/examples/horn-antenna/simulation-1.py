import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def simulate_conical_coaxial_horn():
    """
    Broadband coaxial horn simulation with impulse-like excitation
    and measurement of reflection coefficient.
    """

    # ------------------ Geometry Parameters ------------------
    # Cylinder (coaxial base) parameters (mm)
    outer_radius = 2.0
    inner_radius = 0.65
    height_mm = 10.0

    # Cone parameters (mm)
    cone_height = 160.0
    cone_outer_r1 = 2.0
    cone_outer_r2 = 80.0
    cone_inner_r1 = 1.0
    cone_inner_r2 = 40.0

    resolution = 5  # pixels/mm

    # ------------------ Geometry ------------------
    outer_cyl = mp.Cylinder(
        radius=outer_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.metal
    )

    inner_cyl = mp.Cylinder(
        radius=inner_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    outer_cone = mp.Cone(
        radius=cone_outer_r1,
        radius2=cone_outer_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.metal
    )

    inner_cone = mp.Cone(
        radius=cone_inner_r1,
        radius2=cone_inner_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.air
    )

    geometry = [outer_cyl, inner_cyl, outer_cone, inner_cone]

    # ------------------ Simulation Cell ------------------
    sx = 2 * cone_outer_r2 + 20
    sy = 2 * cone_outer_r2 + 20
    sz = height_mm + cone_height + 20
    cell_size = mp.Vector3(sx, sy, sz)

    # PML boundaries
    pml_layers = [mp.PML(thickness=10.0)]

    # ------------------ Source Parameters ------------------
    # Frequency range: 0.05–10 GHz
    fmin = 0.05     # GHz
    fmax = 10.0     # GHz
    fc = 0.5*(fmin + fmax)   # center frequency
    df = fmax - fmin

    # Convert GHz to Meep units (1 µm base unit)
    # Assume 1 µm = 1 mm for scaling (adjust if needed)
    fmin /= 1e3
    fmax /= 1e3
    fc /= 1e3
    df /= 1e3

    # Gaussian source ~ broadband impulse
    src = mp.Source(
        src=mp.GaussianSource(frequency=fc, fwidth=df),
        component=mp.Ez,   # excite Ez (radial pol. will appear in near-fields)
        center=mp.Vector3(0, 0, -0.5*sz + 5),  # just inside feed
        size=mp.Vector3(0, 0, 0)
    )

    # ------------------ Simulation ------------------
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=[src],
        resolution=resolution,
        dimensions=3
    )

    # ------------------ Monitors ------------------
    # Reflection monitor (at feed plane)
    refl_pt = mp.Vector3(0, 0, -0.5*sz + 5)
    refl_flux = sim.add_flux(fc, df, 1000, mp.FluxRegion(center=refl_pt, size=mp.Vector3(10, 10, 0)))

    # Near-field slice monitor (to get Er distribution)
    nf_plane = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        fmin=fc-df/2, fmax=fc+df/2, nfreq=200,
        where=mp.Volume(center=mp.Vector3(0, 0, 0.5*cone_height), size=mp.Vector3(40, 40, 0))
    )

    # ------------------ Run ------------------
    sim.run(until=500)

    # ------------------ Reflection Spectrum ------------------
    freqs = mp.get_flux_freqs(refl_flux)
    refl_data = mp.get_fluxes(refl_flux)

    # Convert to dB
    refl_db = 10*np.log10(np.abs(refl_data))

    # ------------------ Plot Reflection ------------------
    plt.figure()
    plt.plot(freqs*1e3, refl_db)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflection (dB)")
    plt.title("Reflection Coefficient S11")
    plt.grid(True)
    plt.show()

    print("Simulation finished. Reflection spectrum plotted.")

if __name__ == "__main__":
    simulate_conical_coaxial_horn()
