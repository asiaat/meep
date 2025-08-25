import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def main():
    """
    Runs the Meep simulation for a cylindrical ring resonator,
    finds its modes with Harminv, and outputs the fields.
    """
    # --- 1. Define Simulation Parameters ---
    n = 3.4          # index of waveguide
    w = 1            # width of waveguide
    r = 1            # inner radius of ring
    pad = 4          # padding between waveguide and edge of PML
    dpml = 2         # thickness of PML

    sr = r + w + pad + dpml  # radial size (cell is from 0 to sr)
    
    # Simulation cell in cylindrical coordinates (r, phi, z)
    # The computational cell is a 1D line along the radius 'r'.
    cell = mp.Vector3(sr, 0, 0)

    # phi (angular) dependence of the fields is given by exp(i * m * phi)
    m = 3

    # --- 2. Define Geometry ---
    # A Block in cylindrical coordinates with infinite size in y and z
    # corresponds to a ring in the r-phi plane.
    geometry = [
        mp.Block(
            center=mp.Vector3(r + (w / 2)),
            size=mp.Vector3(w, 1e20, 1e20),
            material=mp.Medium(index=n),
        )
    ]

    pml_layers = [mp.PML(dpml)]
    resolution = 20

    # --- 3. Define Source ---
    fcen = 0.15      # pulse center frequency
    df = 0.1         # pulse frequency width
    
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(r + 0.1),
        )
    ]

    # --- 4. Create Simulation Object ---
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=m,
    )

    # --- 5. Run Harminv Analysis ---
    # This run finds the resonant modes of the ring.
    # Harminv analyzes the Ez field at a point inside the ring
    # after the source has finished.
    print("--- Running Harminv to find resonant modes... ---")
    sim.run(
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)),
        until_after_sources=200,
    )
    print("--- Harminv analysis complete. ---")

    # --- 6. Run Field Output ---
    # This second run outputs the fields to create an animation.
    # It runs for one period of the center frequency.
    output_dir = "meep-output-ring"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sim.use_output_directory(output_dir)
    
    print(f"\n--- Running field output for one period... Output will be in '{output_dir}' directory. ---")
    sim.run(
        mp.in_volume(
            # Output a larger volume from -sr to sr, using symmetry for the -r part
            mp.Volume(center=mp.Vector3(), size=mp.Vector3(2 * sr)),
            # At the beginning, output the geometry (epsilon)
            mp.at_beginning(mp.output_epsilon),
            # Append slices of the Ez field to a single HDF5 file over time
            mp.to_appended("ez", mp.at_every(1 / fcen / 20, mp.output_efield_z)),
        ),
        until=1 / fcen,
    )
    print("--- Field output complete. ---")


# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == '__main__':
    main()
