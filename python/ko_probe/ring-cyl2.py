import meep as mp
import numpy as np
import math
import os
import sys
import h5py

def main():
    """
    Runs the Meep simulation for a cylindrical ring resonator and manually
    saves the output fields from the computational cell (0 to sr).
    """
    # --- 1. Define Simulation Parameters ---
    n = 3.4
    w = 1
    r = 1
    pad = 4
    dpml = 2
    sr = r + w + pad + dpml
    cell = mp.Vector3(sr, 0, 0)
    m = 4
    resolution = 20
    fcen = 0.15
    df = 0.1

    # --- 2. Define Geometry and Sources ---
    geometry = [mp.Block(center=mp.Vector3(r + (w / 2)), size=mp.Vector3(w, 1e20, 1e20), material=mp.Medium(index=n))]
    pml_layers = [mp.PML(dpml)]
    sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3(r + 0.1))]

    # --- 3. Set up Simulation Object and Output Directory ---
    basename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_dir = f"{basename}-output"
    
    if mp.am_master():
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
        else:
            os.makedirs(output_dir)

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=m,
        filename_prefix=basename
    )
    
    sim.use_output_directory(output_dir)

    # --- 4. Run Harminv Analysis ---
    print("--- Running Harminv to find resonant modes... ---")
    sim.run(mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)), until_after_sources=200)
    print("--- Harminv analysis complete. ---")

    # --- 5. Manual Field Output ---
    ez_data_list = []

    def save_ez_slice(sim_instance):
        """A custom step function to get and store the Ez field slice."""
        # FIX: Get data only from the computational cell (0 to sr).
        # Do not use a 'where' argument here.
        ez_slice = sim_instance.get_efield_z()
        ez_data_list.append(ez_slice)

    print(f"\n--- Running field output for one period... ---")
    sim.run(
        # Output epsilon for the full symmetric volume
        mp.in_volume(mp.Volume(center=mp.Vector3(), size=mp.Vector3(2 * sr)),
                     mp.at_beginning(mp.output_epsilon)),
        mp.at_every(1 / fcen / 20, save_ez_slice),
        until=1 / fcen,
    )
    print("--- Field data collected. ---")

    # --- 6. Manually save the collected data to HDF5 ---
    if mp.am_master():
        print("--- Saving collected field data to HDF5 file... ---")
        # Stack the collected 1D slices into a single 2D array (r, t)
        ez_animation_data = np.stack(ez_data_list, axis=1)
        
        # Manually create and write to the HDF5 file
        data_filepath = os.path.join(output_dir, f"{basename}-ez.h5")
        with h5py.File(data_filepath, 'w') as f:
            f.create_dataset('ez', data=ez_animation_data)
        print(f"--- Data successfully saved to {data_filepath} ---")

# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == '__main__':
    main()
