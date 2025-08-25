import h5py
import sys
import os
import glob

def inspect_h5_file(filepath):
    """Opens an HDF5 file and prints its structure."""
    print(f"\n--- Inspecting file: {os.path.basename(filepath)} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            print("Keys (datasets) found in the file:")
            if not list(f.keys()):
                print("  (No datasets found)")
            else:
                # Print the name of each dataset found at the top level
                for key in f.keys():
                    print(f"  - '{key}'")

            # Also check for attributes, which is where coordinates are stored
            if list(f.keys()):
                first_key = list(f.keys())[0]
                print(f"\nAttributes found in dataset '{first_key}':")
                if not list(f[first_key].attrs.keys()):
                    print("  (No attributes found)")
                else:
                    for attr_name, attr_val in f[first_key].attrs.items():
                         print(f"  - '{attr_name}'")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# ==============================================================================
# Main script execution
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <simulation_basename>")
        print("Example: python inspect_h5.py ring-cyl2")
        sys.exit(1)

    basename = sys.argv[1]
    output_dir = f"{basename}-output"

    print(f"Looking for output files in directory: '{output_dir}'")

    # --- Inspect the Epsilon File ---
    eps_files = glob.glob(os.path.join(output_dir, f"{basename}-eps-*.h5"))
    if eps_files:
        inspect_h5_file(eps_files[0])
    else:
        print(f"Error: Could not find an epsilon file for basename '{basename}'")
        
    # --- Inspect the Ez File ---
    ez_file = os.path.join(output_dir, f"{basename}-ez.h5")
    if os.path.exists(ez_file):
        inspect_h5_file(ez_file)
    else:
        print(f"Error: Could not find the ez file: {ez_file}")
