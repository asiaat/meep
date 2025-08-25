import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import argparse

def h5_to_png(data_file, eps_file, output_filename="plot.png", scale=2):
    """
    Replicates the functionality of Meep's h5topng to create a PNG image
    from HDF5 data files, showing a 2D plot of field vs. radius and time.
    """
    print(f"\n--- Reading data from '{os.path.basename(data_file)}' ---")
    print(f"--- Reading geometry from '{os.path.basename(eps_file)}' ---")

    try:
        with h5py.File(data_file, 'r') as f:
            field_data_complex = f['ez'][:]
        with h5py.File(eps_file, 'r') as f:
            eps_data = f['eps'][:]
            
            # Manually generate coordinates since they are missing
            r = 1
            w = 1
            pad = 4
            dpml = 2
            sr = r + w + pad + dpml
            r_coords = np.linspace(-sr, sr, len(eps_data))

    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"Error: Could not read HDF5 files. Details: {e}")
        return

    # --- FIX: Take the real part of the complex field data for plotting ---
    field_data = np.real(field_data_complex)

    num_time_steps = field_data.shape[1]
    time_coords = np.arange(num_time_steps)

    fig, ax = plt.subplots(figsize=(5 * scale, 4 * scale), dpi=100)
    max_abs_val = np.max(np.abs(field_data))
    
    # Ensure max_abs_val is not zero to avoid errors with empty fields
    if max_abs_val == 0:
        max_abs_val = 1.0

    im = ax.pcolormesh(
        time_coords,
        r_coords,
        field_data,
        cmap='RdBu',
        vmin=-max_abs_val,
        vmax=max_abs_val,
        shading='gouraud'
    )

    eps_grid_2d, _ = np.meshgrid(eps_data, time_coords, indexing='ij')
    ax.contour(
        time_coords,
        r_coords,
        eps_grid_2d,
        levels=np.unique(eps_data),
        colors='k',
        linewidths=1.0,
        linestyles='--'
    )

    ax.set_title("Ez Field vs. Radius and Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Radius (r)")
    
    cbar = fig.colorbar(im)
    cbar.set_label("Electric Field (Ez)")
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"--- Successfully saved plot to '{output_filename}' ---")
    plt.close(fig)


# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basename', type=str, help='The base name of the simulation script and output files (e.g., ring-cyl2)')
    args = parser.parse_args()

    sim_script_basename = args.basename
    output_dir = f"{sim_script_basename}-output"

    data_filepath = os.path.join(output_dir, f"{sim_script_basename}-ez.h5")
    
    eps_files = glob.glob(os.path.join(output_dir, f"{sim_script_basename}-eps-*.h5"))
    
    if not os.path.exists(data_filepath) or not eps_files:
        print(f"Error: HDF5 files not found in '{output_dir}'.")
        print("Please run the simulation script first to generate the output files.")
    else:
        eps_filepath = eps_files[0]
        output_image_path = os.path.join(output_dir, "ez_rt_plot.png")
        
        h5_to_png(data_filepath, eps_filepath, output_image_path, scale=2)
