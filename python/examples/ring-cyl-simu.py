import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import glob
# imageio is used to compile PNG frames into a GIF
import imageio.v2 as imageio

def main():
    """
    Runs the Meep simulation for a cylindrical ring resonator,
    saves field plots as PNG frames, and compiles them into a GIF.
    """
    # --- 1. Define Simulation Parameters ---
    n = 3.4
    w = 1
    r = 1
    pad = 4
    dpml = 2
    sr = r + w + pad + dpml
    cell = mp.Vector3(sr, 0, 0)
    m = 3
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
    animation_dir = os.path.join(output_dir, "animation_frames")
    
    if mp.am_master():
        if not os.path.exists(animation_dir):
            os.makedirs(animation_dir)
        # Clean up old frames before starting
        for f in glob.glob(os.path.join(animation_dir, "*.png")):
            os.remove(f)

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

    # --- 5. Field Output and Animation Frame Generation ---
    # Manually create the coordinate and material arrays for plotting
    sim.init_sim()
    r_coords_comp = np.linspace(0, sr, int(sr * resolution))
    eps_data_comp = sim.get_epsilon()
    n_data_comp = np.sqrt(eps_data_comp)
    waveguide_indices = np.where(n_data_comp > 1.0)
    wg_min_r = r_coords_comp[waveguide_indices[0][0]]
    wg_max_r = r_coords_comp[waveguide_indices[0][-1]]

    frame_num = 0
    def save_png_frame(sim):
        nonlocal frame_num
        # Get field data for the computational cell (0 to sr)
        ez_data_comp = sim.get_efield_z()
        
        # Manually create the symmetric data for plotting (-sr to sr)
        ez_data_full = np.concatenate((np.flip(ez_data_comp), ez_data_comp))
        r_coords_full = np.linspace(-sr, sr, len(ez_data_full))

        # Create the plot for this frame
        fig, ax = plt.subplots(dpi=100)
        ax.plot(r_coords_full, ez_data_full, 'b-')
        
        # Add waveguide overlay
        ax.axvspan(wg_min_r, wg_max_r, color='gray', alpha=0.3, label='Waveguide')
        ax.axvspan(-wg_max_r, -wg_min_r, color='gray', alpha=0.3)
        
        # Formatting
        ax.set_ylim(-0.2, 0.2) # Fixed ylim for consistent animation
        ax.set_xlim(-sr, sr)
        ax.set_xlabel("Position r")
        ax.set_ylabel("Electric Field (Ez)")
        ax.set_title(f"Ez Field at Time Step {sim.meep_time():.2f}")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        
        # Save the frame as a PNG file
        plt.savefig(os.path.join(animation_dir, f"frame_{frame_num:04d}.png"))
        plt.close(fig)
        frame_num += 1

    print(f"\n--- Running field output and saving frames to '{animation_dir}'... ---")
    sim.run(
        mp.at_every(1 / fcen / 20, save_png_frame),
        until=1 / fcen
    )
    print("--- Frame generation complete. ---")

    # --- 6. Compile frames into a GIF ---
    if mp.am_master():
        print("\n--- Compiling frames into GIF... ---")
        frame_files = sorted(glob.glob(os.path.join(animation_dir, "*.png")))
        images = [imageio.imread(f) for f in frame_files]
        gif_path = os.path.join(output_dir, "ring_animation.gif")
        imageio.mimsave(gif_path, images, duration=50) # duration in ms
        print(f"--- Animation successfully saved to '{gif_path}' ---")

# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == '__main__':
    main()
