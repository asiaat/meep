import meep as mp
import numpy as np

mp.verbosity(0)

def run_reflection_only():
    """
    Minimal axisymmetric coaxial port simulation.
    Measures only reflection (S11).
    """

    resolution = 2   # pixels per mm (low for speed)

    # Simple short cylinder feed (mm)
    cyl_outer_r = 2.0
    cyl_height  = 5.0

    # Computational cell (r × z plane)
    r_extent = cyl_outer_r + 5
    z_extent = cyl_height + 15
    cell_size = mp.Vector3(r_extent, z_extent)

    pml_layers = [mp.PML(thickness=5.0, direction=mp.R)]

    # ------------------ Geometry ------------------
    geometry = [
        mp.Block(size=mp.Vector3(cyl_outer_r, cyl_height),
                 center=mp.Vector3(0.5*cyl_outer_r, 0.5*cyl_height),
                 material=mp.metal)
    ]

    # ------------------ Source ------------------
    fcen = 7.0   # center frequency (GHz)
    df   = 4.0   # span (3–7 GHz)

    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,    # axisymmetric source
        center=mp.Vector3(0.0, 1.0)
    )

    # ------------------ Simulation ------------------
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=[src],
        boundary_layers=pml_layers,
        resolution=resolution,
        dimensions=2,  # r–z plane
        m=0            # azimuthal symmetry
    )

    # ------------------ Reflection Monitor ------------------
    refl_region = mp.FluxRegion(center=mp.Vector3(0, 2.0),
                                size=mp.Vector3(r_extent-1, 0))
    refl = sim.add_flux(fcen, df, 3000, refl_region)

    # ------------------ Run ------------------
    #sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-3))
    sim.run(until=9000)

    # ------------------ Save Data ------------------
    freqs = mp.get_flux_freqs(refl)
    refl_data = mp.get_fluxes(refl)
    np.savetxt("reflection_only.txt", np.column_stack([freqs, refl_data]))

    print("✅ Reflection-only simulation finished.")
    print("Saved: reflection_only.txt")

if __name__ == "__main__":
    run_reflection_only()
