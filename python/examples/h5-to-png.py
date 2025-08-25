import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

def h5_to_png(data_file, eps_file, output_filename="ez_rt_plot.png", scale=2):
    """
    Replicates the functionality of Meep's h5topng to create a PNG image
    from HDF5 data files.

    Args:
        data_file (str): Path to the HDF5 file containing the field data (e.g., ez.h5).
        eps_file (str): Path to the HDF5 file containing the epsilon data for contours.
        output_filename (str): Name of the output PNG file.
        scale (float): Scaling factor for the output image resolution.
    """
    print(f"--- Reading data from '{os.path.basename(data_file)}' ---")
    print(f"--- Reading geometry from '{os.path.basename(eps_file)}' ---")

    try:
        # --- 1. Read Field and Geometry Data ---
        with h5py.File(data_file, 'r') as f:
            # The dataset is named 'ez' inside the file
            field_data = f['ez'][:]  # This will be an (r, t) array

        with h5py.File(eps_file, 'r') as f:
            eps_data = f['epsilon'][:]
            # The coordinates are stored as an attribute
            r_coords = f['epsilon'].attrs['x']
    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"Error: Could not read HDF5 files. Details: {e}")
        return

    # The time steps are not explicitly saved, so we create a time axis
    num_time_steps = field_data.shape[1]
    time_coords = np.arange(num_time_steps)

    # --- 2. Set up the Plot ---
    # Use figsize and dpi to control the output image size (replicates -S flag)
    fig, ax = plt.subplots(figsize=(5 * scale, 4 * scale), dpi=100)

    # --- 3. Plot the Field Data ---
    # Determine the absolute maximum for a symmetric colormap (replicates -Zc)
    max_abs_val = np.max(np.abs(field_data))
    
    # Use pcolormesh for a 2D plot. Transpose data to have time on the x-axis.
    # Use the 'RdBu' colormap, which is similar to 'dkbluered'.
    im = ax.pcolormesh(
        time_coords,
        r_coords,
        field_data,
        cmap='RdBu',
        vmin=-max_abs_val,
        vmax=max_abs_val,
        shading='gouraud'
    )

    # --- 4. Overlay Geometry Contours ---
    # The epsilon data is 1D, so we create a 2D grid for the contour plot
    # where the epsilon value is constant along the time axis.
    eps_grid_2d, _ = np.meshgrid(eps_data, time_coords, indexing='ij')
    
    # Draw contour lines where the dielectric constant changes
    ax.contour(
        time_coords,
        r_coords,
        eps_grid_2d,
        levels=np.unique(eps_data), # Draw lines at each unique epsilon value
        colors='k',      # Black contours
        linewidths=1.0,
        linestyles='--'
    )

    # --- 5. Formatting and Saving ---
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
    # Define the output directory and filenames based on the simulation script
    sim_script_basename = "ring-cyl-simu"
    output_dir = f"{sim_script_basename}-output"

    data_filepath = os.path.join(output_dir, f"{sim_script_basename}-ez.h5")
    
    # Find the epsilon file using a wildcard, as its name includes a timestamp
    eps_files = glob.glob(os.path.join(output_dir, f"{sim_script_basename}-eps-*.h5"))
    
    if not os.path.exists(data_filepath) or not eps_files:
        print("Error: HDF5 files not found. Please run the simulation script first.")
    else:
        eps_filepath = eps_files[0]
        output_image_path = os.path.join(output_dir, "ez_rt_plot.png")
        
        # Call the function to generate the plot
        h5_to_png(data_filepath, eps_filepath, output_image_path, scale=2)
