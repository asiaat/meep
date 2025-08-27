import meep as mp
import numpy as np

def run_axisymmetric_horn():
    """
    Axisymmetric simulation of a coaxial horn antenna.
    Uses cylindrical symmetry (r–z plane) -> much faster than full 3D.
    """

    resolution = 2   # pixels per mm (start coarse; increase later)

    # Geometry parameters (mm)
    cyl_outer_r = 2.0
    cyl_inner_r = 0.65
    cyl_height  = 5.0

    cone_height    = 40.0
    cone_outer_r1  = cyl_outer_r
    cone_outer_r2  = 20.0
    cone_inner_r1  = cyl_inner_r
    cone_inner_r2  = 10.0

    # Computational cell size (r × z plane)
    r_extent = cone_outer_r2 + 10
    z_extent = cyl_height + cone_height + 20
    cell_size = mp.Vector3(r_extent, z_extent)

    # PML boundaries (absorbing layers)
    pml_layers = [mp.PML(thickness=5.0, direction=mp.R)]

    # ------------------ Geometry ------------------
    geometry = []

    # Cylinder outer wall (metal)
    geometry.append(mp.Block(
        size=mp.Vector3(cyl_outer_r, cyl_height),
        center=mp.Vector3(0.5*cyl_outer_r, 0.5*cyl_height),
        material=mp.metal
    ))

    # Cone outer wall (metal)
    geometry.append(mp.Prism(
        vertices=[mp.Vector3(cyl_outer_r, cyl_height),
                  mp.Vector3(cone_outer_r2, cyl_height + cone_height),
                  mp.Vector3(cone_outer_r2+0.1, cyl_height + cone_height),
                  mp.Vector3(cyl_outer_r+0.1, cyl_height)],
        height=mp.inf,
        material=mp.metal
    ))

    # Inner hollow (air) -> nothing to add, default is air

    # ------------------ Source ------------------
    fcen = 5.0   # center frequency (GHz)
    df   = 4.0   # frequency width (GHz)

    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,   # axisymmetric source
        center=mp.Vector3(0.0, 1.0),   # just above port
    )

    # ------------------ Simulation ------------------
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=[src],
        boundary_layers=pml_layers,
        resolution=resolution,
        dimensions=2,   # r–z plane
        m=0             # azimuthal symmetry
    )

    # ------------------ Monitors ------------------
    # Reflection monitor near port
    refl_region = mp.FluxRegion(center=mp.Vector3(0, 2.0), size=mp.Vector3(r_extent-2, 0))
    refl = sim.add_flux(fcen, df, 100, refl_region)

    # Transmission monitor near horn aperture
    trans_region = mp.FluxRegion(center=mp.Vector3(0, cyl_height+cone_height-2),
                                 size=mp.Vector3(r_extent-2, 0))
    trans = sim.add_flux(fcen, df, 100, trans_region)

    # ------------------ Run ------------------
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(), 1e-3))

    # ------------------ Data extraction ------------------
    freqs = mp.get_flux_freqs(refl)
    refl_data = mp.get_fluxes(refl)
    trans_data = mp.get_fluxes(trans)

    np.savetxt("axisym_reflection.txt", np.column_stack([freqs, refl_data]))
    np.savetxt("axisym_transmission.txt", np.column_stack([freqs, trans_data]))

    print("✅ Axisymmetric horn simulation finished.")
    print("Saved: axisym_reflection.txt, axisym_transmission.txt")

if __name__ == "__main__":
    run_axisymmetric_horn()
