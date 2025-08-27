import meep as mp
import numpy as np

def run_coaxial_horn_sim():
    """
    Simulation of a hollow coaxial horn antenna with impulse excitation.
    Measures reflection (S11) and prepares field monitors for time-domain reconstruction.
    """

    # Cylinder parameters (mm)
    outer_radius = 2.0
    inner_radius = 0.65
    height_mm = 10.0

    # Cone parameters (mm)
    cone_height = 20.0
    cone_outer_r1 = 2.0
    cone_outer_r2 = 10.0
    cone_inner_r1 = 1.0
    cone_inner_r2 = 40.0
    

    resolution = 5  # pixels/mm

    # ------------------ Geometry ------------------
    outer_cyl = mp.Cylinder(radius=outer_radius,
                            height=height_mm,
                            axis=mp.Vector3(0, 0, 1),
                            material=mp.metal)

    inner_cyl = mp.Cylinder(radius=inner_radius,
                            height=height_mm,
                            axis=mp.Vector3(0, 0, 1),
                            material=mp.air)

    outer_cone = mp.Cone(radius=cone_outer_r1,
                         radius2=cone_outer_r2,
                         height=cone_height,
                         axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(0, 0, 0.5*height_mm + 0.5*cone_height),
                         material=mp.metal)

    inner_cone = mp.Cone(radius=cone_inner_r1,
                         radius2=cone_inner_r2,
                         height=cone_height,
                         axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(0, 0, 0.5*height_mm + 0.5*cone_height),
                         material=mp.air)

    geometry = [outer_cyl, inner_cyl, outer_cone, inner_cone]

    # ------------------ Simulation cell ------------------
    sx = 2 * cone_outer_r2 + 4
    sy = 2 * cone_outer_r2 + 4
    sz = height_mm + cone_height + 20
    cell_size = mp.Vector3(sx, sy, sz)

    pml_layers = [mp.PML(thickness=5.0)]

    # ------------------ Broadband Source ------------------
    fmin = 0.05   # 0.05 GHz → scaled to Meep units later
    fmax = 10.0   # 10 GHz
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    # Use GaussianImpulse (broadband)
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,   # you can choose Er/Ez depending on coax feed mode
        center=mp.Vector3(0, 0, -0.5*height_mm + 1.0),  # near coaxial port
        size=mp.Vector3(0, 0, 0)  # point-like
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

    # ------------------ Reflection/Transmission Monitors ------------------
    refl_region = mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*height_mm+2.0),
                                size=mp.Vector3(2*outer_radius+2, 2*outer_radius+2, 0))
    refl = sim.add_flux(fcen, df, 2, refl_region)

    

    # ------------------ Run ------------------
    #sim.run(until=50)   # time steps, adjust as needed
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0,0,0), 1e-3))
    #sim.run(until=2)


    # ------------------ Data extraction ------------------
    # Reflection spectrum (S11)
    freqs = mp.get_flux_freqs(refl)
    refl_data = mp.get_fluxes(refl)

    
    # Save results
    np.savetxt("reflection_S11.txt", np.column_stack([freqs, refl_data]))
   

    print("Simulation complete. Reflection & transmission data saved.")

if __name__ == '__main__':
    run_coaxial_horn_sim()
