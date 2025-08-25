import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import h5py
import sys
import glob

def create_animation(output_dir, basename):
    """
    Reads the HDF5 output from the Meep simulation and creates a GIF animation.

    Args:
        output_dir (str): The directory containing the HDF5 output files.
        basename (str): The base name of the output files (from the script name).
    """
    print(f"\n--- Creating animation from files in '{output_dir}'... ---")
    
    # --- 1. Load Data from HDF5 Files ---
    ez_file = os.path.join(output_dir, f"{basename}-ez.h5")
    
    eps_files = glob.glob(os.path.join(output_dir, f"{basename}-eps-*.h5"))
    if not eps_files:
        print(f"Error: Epsilon file matching pattern '{basename}-eps-*.h5' not found.")
        return
    eps_file = eps_files[0]

    try:
        with h5py.File(ez_file, 'r') as f:
            ez_data = f['ez'][:]
        with h5py.File(eps_file, 'r') as f:
            eps_data = f['epsilon'][:]
            coords = f['epsilon'].attrs['x']
    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"Error: Could not read HDF5 files. Make sure the simulation ran correctly. Details: {e}")
        return

    # --- 2. Prepare Data for Plotting ---
    num_times = ez_data.shape[1]
    n_data = np.sqrt(eps_data)
    waveguide_indices = np.where(n_data > 1.0)
    
    # --- 3. Set up the Animation Plot ---
    fig, ax = plt.subplots(dpi=100)
    line, = ax.plot(coords, ez_data[:, 0], 'b-')
    
    if len(waveguide_indices[0]) > 0:
        wg_min = coords[waveguide_indices[0][0]]
        wg_max = coords[waveguide_indices[0][-1]]
        ax.axvspan(wg_min, wg_max, color='gray', alpha=0.3, label='Waveguide')
        ax.axvspan(-wg_max, -wg_min, color='gray', alpha=0.3)

    ax.set_ylim(-np.max(np.abs(ez_data)) * 1.1, np.max(np.abs(ez_data)) * 1.1)
    ax.set_xlim(np.min(coords), np.max(coords))
    ax.set_xlabel("Position r")
    ax.set_ylabel("Electric Field (Ez)")
    ax.set_title("Ez Field Evolution in Ring Resonator")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # --- 4. Define Animation Logic ---
    def animate(frame):
        line.set_ydata(ez_data[:, frame])
        time_text.set_text(f'Time Step: {frame+1}/{num_times}')
        return line, time_text

    # --- 5. Create and Save the Animation ---
    anim = FuncAnimation(fig, animate, frames=num_times, blit=True, interval=50)
    output_path = os.path.join(output_dir, "ring_animation.gif")
    anim.save(output_path, writer='pillow')
    
    print(f"--- Animation successfully saved to '{output_path}' ---")
    plt.close(fig)

# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == '__main__':
    # The basename should match the simulation script's name
    script_basename = "ring-cyl-simu" 
    output_directory = f"{script_basename}-output"
    create_animation(output_directory, script_basename)
