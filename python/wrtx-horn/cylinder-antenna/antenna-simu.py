import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# --- 1. GEOMETRY DEFINITION ---
def make_horn_antenna_geometry(
    r_in_start, r_in_end, r_out_start, r_out_end, h,
    taper_profile='linear', n_slices=100, material=mp.metal
):
    """
    Generates a list of MEEP geometry objects for a horn-like antenna.
    This function approximates the horn shape by stacking thin annular slices
    made of mp.Block objects. This is the standard MEEP method for creating
    arbitrary solids of revolution, as mp.Prism is not compatible with
    the cylindrical solver's revolution-based geometry creation.
    """
    geometry = []
    z_coords = np.linspace(0, h, n_slices, endpoint=False)
    slice_height = h / n_slices

    for i, z_start in enumerate(z_coords):
        z_mid = z_start + 0.5 * slice_height
        # Normalized position along the horn length (0 to 1)
        s = z_mid / h

        # Calculate radii based on the selected taper profile
        if taper_profile == 'linear':
            slice_r_outer = r_out_start + (r_out_end - r_out_start) * s
            slice_r_inner = r_in_start + (r_in_end - r_in_start) * s
        elif taper_profile == 'quadratic':
            slice_r_outer = r_out_start + (r_out_end - r_out_start) * s**2
            slice_r_inner = r_in_start + (r_in_end - r_in_start) * s**2
        elif taper_profile == 'exponential':
            r_out_start_safe = max(r_out_start, 1e-9)
            r_in_start_safe = max(r_in_start, 1e-9)
            slice_r_outer = r_out_start_safe * np.exp(np.log(r_out_end / r_out_start_safe) * s)
            slice_r_inner = r_in_start_safe * np.exp(np.log(r_in_end / r_in_start_safe) * s)
        else:
            raise ValueError("Invalid taper_profile. Choose from 'linear', 'quadratic', 'exponential'.")

        geometry.append(mp.Block(
            material=material,
            center=mp.Vector3(x=(slice_r_outer + slice_r_inner) / 2, z=z_mid),
            size=mp.Vector3(x=slice_r_outer - slice_r_inner, z=slice_height)
        ))
    return geometry

# --- 2. SOURCE DEFINITION ---
def create_source_waveform(fcen):
    """
    Creates an interpolated, time-integrated Ricker wavelet function for the source.
    This function encapsulates the analytical pulse definition and its preparation.
    
    Args:
        fcen (float): The center frequency of the Ricker wavelet.

    Returns:
        function: A function of time `t` that MEEP can use as a custom source.
    """
    # Define the analytical pulse for the radial E-field (Er)
    def ricker_wavelet(t, fcen):
        t0 = 2.0 / fcen  # Time shift to center the pulse away from t=0
        arg = (np.pi * fcen * (t - t0))**2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    # Generate the time-integrated feed signal g_f(t) on a fine time grid
    dt_source = 1 / (25 * fcen)
    t_max_source = 8 / fcen
    time_steps = np.arange(0, t_max_source, dt_source)
    Er_at_feed = ricker_wavelet(time_steps, fcen)
    g_f = cumulative_trapezoid(Er_at_feed, time_steps, initial=0)

    # Create an interpolation function that MEEP can call at its own time steps
    def source_function(t):
        return np.interp(t, time_steps, g_f)
        
    return source_function

def define_sources(source_waveform_func, feed_inner_radius, feed_outer_radius, sz, dpml):
    """
    Creates and returns the MEEP source object list based on the waveform and geometry.

    Args:
        source_waveform_func (function): The function defining the source's time-domain signal.
        feed_inner_radius (float): Inner radius of the coaxial feed.
        feed_outer_radius (float): Outer radius of the coaxial feed.
        sz (float): Axial size of the computational cell.
        dpml (float): Thickness of the PML layers.

    Returns:
        list: A list containing the configured mp.Source object.
    """
    source_center_r = (feed_inner_radius + feed_outer_radius) / 2
    source_size_r = feed_outer_radius - feed_inner_radius
    source_center_z = -sz / 2 + dpml + 0.1

    sources = [
        mp.Source(
            src=mp.CustomSource(src_func=source_waveform_func),
            component=mp.Er,
            center=mp.Vector3(x=source_center_r, z=source_center_z),
            size=mp.Vector3(x=source_size_r, z=0)
        )
    ]
    return sources

# --- 3. SIMULATION SETUP & EXECUTION ---
def setup_simulation(cell_size, resolution, pml_layers, geometry, sources):
    """
    Initializes and returns the MEEP Simulation object.
    """
    return mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=0
    )

def plot_simulation_results(sim, sr, sz):
    """
    Fetches the E-field data from the completed simulation and generates a plot.
    """
    plot_vol = mp.Volume(center=mp.Vector3(x=sr/2, z=0), size=mp.Vector3(x=sr, z=sz))
    # Corrected line: Use sim.get_array with a specified component and volume.
    er_data = sim.get_array(component=mp.Er, vol=plot_vol)

    plt.figure(figsize=(10, 8))
    # Handle case where field data might be all zeros to avoid plotting errors
    max_val = np.max(abs(er_data))
    vmin, vmax = (-max_val, max_val) if max_val > 0 else (-1, 1)

    plt.imshow(np.transpose(er_data), interpolation='spline36', cmap='RdBu',
               extent=[-sz/2, sz/2, 0, sr], vmin=vmin, vmax=vmax)
    plt.title("Radial Electric Field (Er) at Simulation End")
    plt.xlabel("z (meters)")
    plt.ylabel("r (meters)")
    plt.colorbar(label="Er (arbitrary units)")
    plt.show()

# --- 4. MAIN WORKFLOW ---
def main():
    """
    Main function to define parameters and orchestrate the simulation workflow.
    """
    # Simulation Parameters
    resolution = 25
    dpml = 1.0
    sr = 0.5
    sz = 3.0
    cell_size = mp.Vector3(sr + dpml, 0, sz + 2 * dpml)
    pml_layers = [mp.PML(thickness=dpml)]

    # Antenna Parameters
    feed_inner_radius = 0.01
    feed_outer_radius = 0.03
    horn_inner_radius = 0.10
    horn_outer_radius = 0.12
    horn_height = 0.30

    # Source Parameters
    fcen = 6e9

    # --- Execute Simulation Steps ---
    print("Step 1: Creating horn antenna geometry...")
    horn_geometry = make_horn_antenna_geometry(
        r_in_start=feed_inner_radius, r_in_end=horn_inner_radius,
        r_out_start=feed_outer_radius, r_out_end=horn_outer_radius,
        h=horn_height, taper_profile='linear'
    )

    print("Step 2: Defining custom source...")
    source_waveform = create_source_waveform(fcen)
    sources = define_sources(source_waveform, feed_inner_radius, feed_outer_radius, sz, dpml)

    print("Step 3: Setting up MEEP simulation...")
    sim = setup_simulation(cell_size, resolution, pml_layers, horn_geometry, sources)

    print("Step 4: Running simulation...")
    sim.run(until=15)

    print("Step 5: Visualizing results...")
    plot_simulation_results(sim, sr, sz)
    
    print("Simulation finished.")

if __name__ == '__main__':
    main()


