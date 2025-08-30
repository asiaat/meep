import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def make_horn_antenna_geometry(
    r_in_start, r_in_end, r_out_start, r_out_end, h,
    taper_profile='linear', n_slices=100, material=mp.metal
):
    """
    Generates a list of MEEP geometry objects for a horn-like antenna.

    For a 'linear' profile, this function defines the horn wall's cross-section
    as a single mp.Polygon, which is the most efficient and accurate method for 2D/RZ.
    For curved profiles ('quadratic', 'exponential'), it approximates the shape
    by stacking thin annular slices made of mp.Block objects.

    Args:
        r_in_start (float): Inner radius at the base (z=0).
        r_in_end (float): Inner radius at the horn opening (z=h).
        r_out_start (float): Outer radius at the base (z=0).
        r_out_end (float): Outer radius at the horn opening (z=h).
        h (float): Height of the horn along the z-axis.
        taper_profile (str): The profile of the flare. Can be 'linear', 'quadratic', 'or 'exponential'.
        n_slices (int): Number of slices to approximate curved geometries. Only used for non-linear profiles.
        material (meep.Material): The material for the antenna walls.

    Returns:
        list: A list of meep.GeometricObject instances.
    """
    if taper_profile == 'linear':
        # For a straight-sided horn in a 2D/cylindrical simulation, we define
        # the cross-section in the r-z plane (which Meep maps to x-z).
        vertices = [
            mp.Vector3(x=r_in_start, z=0),
            mp.Vector3(x=r_out_start, z=0),
            mp.Vector3(x=r_out_end, z=h),
            mp.Vector3(x=r_in_end, z=h)
        ]
        # FIX: A Prism's vertices must lie in a plane perpendicular to its extrusion
        # axis. Since our vertices are in the r-z plane, the axis normal to it is
        # the y-axis. We must specify this axis and provide the mandatory 'height'.
        geometry = [mp.Prism(
            vertices=vertices,
            height=mp.inf,
            axis=mp.Vector3(0, 1, 0),
            material=material
        )]
        return geometry

    # For curved profiles, we fall back to the slicing method
    geometry = []
    z_coords = np.linspace(0, h, n_slices, endpoint=False)
    slice_height = h / n_slices

    for i, z_start in enumerate(z_coords):
        z_mid = z_start + 0.5 * slice_height
        # Normalized position along the horn length (0 to 1)
        s = z_mid / h

        # Calculate radii based on the selected taper profile
        if taper_profile == 'quadratic':
            slice_r_outer = r_out_start + (r_out_end - r_out_start) * s**2
            slice_r_inner = r_in_start + (r_in_end - r_in_start) * s**2
        elif taper_profile == 'exponential':
            # Ensure start radius is not zero for log scale
            r_out_start_safe = max(r_out_start, 1e-9)
            r_in_start_safe = max(r_in_start, 1e-9)
            slice_r_outer = r_out_start_safe * np.exp(np.log(r_out_end / r_out_start_safe) * s)
            slice_r_inner = r_in_start_safe * np.exp(np.log(r_in_end / r_in_start_safe) * s)
        else:
            raise ValueError("Invalid taper_profile. Choose from 'linear', 'quadratic', 'exponential'.")

        # In RZ, a Block centered at x > 0 with a size in x represents an annulus.
        geometry.append(mp.Block(
            material=material,
            center=mp.Vector3(x=(slice_r_outer + slice_r_inner) / 2, z=z_mid),
            size=mp.Vector3(x=slice_r_outer - slice_r_inner, z=slice_height)
        ))

    return geometry

def visualize_geometry_2d(geometry, h, r_max):
    """
    Visualizes the 2D cross-section of the antenna geometry in the (r,z) plane.
    This is how MEEP's cylindrical solver interprets the geometry.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for obj in geometry:
        if isinstance(obj, mp.Prism):
            # For a Prism, extract vertices and draw a polygon
            verts = [(v.x, v.z) for v in obj.vertices]
            poly = plt.Polygon(verts, facecolor='royalblue', edgecolor='darkblue')
            ax.add_patch(poly)
        elif isinstance(obj, mp.Block):
            # For a Block, calculate corners and draw a rectangle
            center = obj.center
            size = obj.size
            r_center, z_center = center.x, center.z
            r_size, z_size = size.x, size.z
            r_min = r_center - r_size / 2
            z_min = z_center - z_size / 2
            rect = plt.Rectangle((r_min, z_min), r_size, z_size, 
                                 facecolor='royalblue', edgecolor='darkblue')
            ax.add_patch(rect)

    ax.set_title("2D Cross-Section of Horn Antenna (r-z plane)")
    ax.set_xlabel("Radius (r)")
    ax.set_ylabel("Height (z)")
    ax.set_xlim(0, r_max * 1.1)
    ax.set_ylim(0, h * 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


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
        m=0  # Azimuthal symmetry
    )

def plot_simulation_results(sim, sr, sz):
    """
    Fetches the E-field data from the completed simulation and generates a plot.
    """
    plot_vol = mp.Volume(center=mp.Vector3(x=sr/2, z=0), size=mp.Vector3(x=sr, z=sz))
    er_data = sim.get_array(component=mp.Er, vol=plot_vol)

    plt.figure(figsize=(10, 8))
    # Handle case where field data might be all zeros to avoid plotting errors
    max_val = np.max(np.abs(er_data))
    vmin, vmax = (-max_val, max_val) if max_val > 1e-9 else (-1, 1)

    plt.imshow(np.transpose(er_data), interpolation='spline36', cmap='RdBu',
               extent=[-sz/2, sz/2, 0, sr], vmin=vmin, vmax=vmax, origin='lower')
    plt.title("Radial Electric Field (Er) at Simulation End")
    plt.xlabel("z (simulation units)")
    plt.ylabel("r (simulation units)")
    plt.colorbar(label="Er (arbitrary units)")
    plt.show()
    

# -----------------------------
# 3.4 MONITORING & COLLECTION
# -----------------------------

def add_time_series_probes(sim, radii, z_list):
    """
    Create point probes for (Er, Ez, Hphi) at selected (r, z) points.
    Returns a dict with locations and recorded arrays (filled later).
    """
    probes = []
    for r in radii:
        for z in z_list:
            probes.append({
                "r": r, "z": z,
                "Er": [], "Ez": [], "Hphi": []
            })
    # Attach a callback to sample at every step
    def sample_fields(sim):
        for p in probes:
            r, z = p["r"], p["z"]
            p["Er"].append(sim.get_field_point(mp.Er, mp.Vector3(x=r, z=z)))
            p["Ez"].append(sim.get_field_point(mp.Ez, mp.Vector3(x=r, z=z)))
            p["Hphi"].append(sim.get_field_point(mp.Hp, mp.Vector3(x=r, z=z)))  # Hp = H_phi in cylindrical
    return probes, sample_fields


def add_field_snapshotter(sr, sz, every_dt, save_prefix="snap"):
    """
    Returns a callback that, when called by mp.at_every(every_dt, ...), pulls grids of Er, Ez, Hphi
    and stores them in memory (and optionally to npy on disk).
    """
    frames = {"t": [], "Er": [], "Ez": [], "Hphi": []}
    def snapshot(sim):
        # Build a volume covering the (r,z) plotting region you already use
        vol = mp.Volume(center=mp.Vector3(x=sr/2, z=0), size=mp.Vector3(x=sr, z=sz))
        er = sim.get_array(component=mp.Er, vol=vol)
        ez = sim.get_array(component=mp.Ez, vol=vol)
        hp = sim.get_array(component=mp.Hp, vol=vol)
        frames["t"].append(sim.meep_time())
        frames["Er"].append(er)
        frames["Ez"].append(ez)
        frames["Hphi"].append(hp)
        # Optional: persist to disk for large runs
        # idx = len(frames["t"]) - 1
        # np.save(f"{save_prefix}-Er-{idx:05d}.npy", er)
        # np.save(f"{save_prefix}-Ez-{idx:05d}.npy", ez)
        # np.save(f"{save_prefix}-Hphi-{idx:05d}.npy", hp)
    return frames, snapshot


def add_flux_monitors(sim, sr, sz, dpml, fcen, df=0.0, nfreq=1):
    """
    Add power/flux monitors at:
      - a plane just after the horn (z = z_plane)
      - a cylindrical 'side' (r = r_side) using two regions to cover top/bottom
    Returns handles to MEEP Flux objects for post-run extraction.
    """
    z_plane = -sz/2 + dpml + 0.5  # adjust to just above feed/horn as needed
    r_side  = sr - dpml - 0.1

    # Through-z plane (integrates normal component of Poynting across the plane)
    flux_z = sim.add_flux(fcen, df, nfreq,
                          mp.FluxRegion(center=mp.Vector3(x=sr/2, z=z_plane),
                                        size=mp.Vector3(x=sr, z=0.0)))
    # Through side wall (two stacked segments to span z fully)
    half = (sz - 2*dpml) / 2
    flux_r_top = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(x=r_side, z= half/2),
                                            size=mp.Vector3(x=0.0, z= half)))
    flux_r_bot = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(x=r_side, z=-half/2),
                                            size=mp.Vector3(x=0.0, z= half)))
    return {"flux_z": flux_z, "flux_r_top": flux_r_top, "flux_r_bot": flux_r_bot}


def add_near_to_far(sim, sr, sz, dpml, fcen, df=0.0, nfreq=1):
    """
    A compact N2F box just inside PMLs to capture radiation leaving the horn.
    In cylindrical coords, MEEP supports near2far; use a rectangle around the aperture/free space.
    """
    r_max = sr - dpml - 0.1
    z_top =  sz/2 - dpml - 0.1
    z_bot = -sz/2 + dpml + 0.1

    n2f = sim.add_near2far(fcen, df, nfreq,
                           mp.Near2FarRegion(center=mp.Vector3(x= r_max/2, z= z_top),
                                              size=mp.Vector3(x=r_max, z=0)),
                           mp.Near2FarRegion(center=mp.Vector3(x= r_max/2, z= z_bot),
                                              size=mp.Vector3(x=r_max, z=0)),
                           mp.Near2FarRegion(center=mp.Vector3(x= r_max,   z= 0),
                                              size=mp.Vector3(x=0,    z=z_top - z_bot)))
    return n2f


# -----------------------------
# 3.5 QUICK ANALYTICS
# -----------------------------

def energy_density(Er, Ez, Hphi, eps_rel=1.0, mu_rel=1.0):
    """
    Instantaneous energy density (MEEP units): u = 0.5*(eps*|E|^2 + mu*|H|^2).
    For TEM-like m=0 case, Ephi≈0 and Hphi is dominant magnetic component.
    """
    E2 = Er**2 + Ez**2
    H2 = Hphi**2
    return 0.5*(eps_rel*E2 + mu_rel*H2)


def ring_radius_and_index(Er, Ez, Hphi, r_coords, z_coords,
                          z_pick=None, r_ring_band=(0.05, 0.25), r_axis_band=(0.0, 0.02)):
    """
    From a single snapshot (2D arrays), estimate:
      - r0 : radius of peak energy at chosen z-slice
      - Torodiality index T = E_ring / (E_ring + E_axis)
    """
    U = energy_density(Er, Ez, Hphi)  # shape (Nr, Nz) with your transpose convention, see below

    # define coordinate vectors
    Nr, Nz = U.shape  # CAREFUL: depends on how you store (r,z) vs (z,r)

    # We'll assume arrays are indexed [r, z] as in your get_array usage (er_data)
    if z_pick is None:
        # Choose a mid-plane where the pulse is likely to appear
        z_pick = 0.0

    # find nearest z-index
    z_idx = np.argmin(np.abs(z_coords - z_pick))

    # energy vs radius at this z
    U_r = U[:, z_idx]  # (Nr,)

    # ring radius = argmax(U_r) restricted to a plausible band
    rmin, rmax = r_ring_band
    valid = (r_coords >= rmin) & (r_coords <= rmax)
    if np.any(valid):
        idx_band = np.where(valid)[0]
        r0_idx = idx_band[np.argmax(U_r[valid])]
    else:
        r0_idx = np.argmax(U_r)
    r0 = r_coords[r0_idx]

    # toroidality index: integrate energy in an annulus vs near-axis core
    def radial_mask(rmin, rmax):
        return (r_coords >= rmin) & (r_coords < rmax)

    # integrate across full z for robustness (weights 2π r in cylindrical area element)
    mask_ring = radial_mask(*r_ring_band)
    mask_axis = radial_mask(*r_axis_band)

    # area-weighted integrals
    U_ring = np.trapz(np.trapz(U[mask_ring, :] * r_coords[mask_ring, None], r_coords[mask_ring], axis=0), z_coords)
    U_axis = np.trapz(np.trapz(U[mask_axis, :] * r_coords[mask_axis, None], r_coords[mask_axis], axis=0), z_coords)

    T = U_ring / (U_ring + U_axis + 1e-30)
    return r0, T


def plot_snapshot_fields(r_coords, z_coords, Er, Ez, Hphi, title_suffix=""):
    """
    Nice colormaps + vector quiver. Matches your plotting style.
    """
    # transpose if your arrays are [r,z] to plot with extent=[zmin,zmax, rmin,rmax]
    ErT, EzT = Er.T, Ez.T

    fig, axs = plt.subplots(1, 3, figsize=(18,5))

    im0 = axs[0].imshow(ErT, extent=[z_coords[0], z_coords[-1], r_coords[0], r_coords[-1]],
                        origin='lower', aspect='auto', cmap='RdBu')
    axs[0].set_title(f"E_r {title_suffix}")
    axs[0].set_xlabel("z"); axs[0].set_ylabel("r"); plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(EzT, extent=[z_coords[0], z_coords[-1], r_coords[0], r_coords[-1]],
                        origin='lower', aspect='auto', cmap='RdBu')
    axs[1].set_title(f"E_z {title_suffix}")
    axs[1].set_xlabel("z"); plt.colorbar(im1, ax=axs[1])

    # Vector field (downsample)
    skip_r = max(1, len(r_coords)//60)
    skip_z = max(1, len(z_coords)//120)
    R, Z = np.meshgrid(r_coords[::skip_r], z_coords[::skip_z], indexing="ij")
    axs[2].quiver(Z, R, Ez[::skip_r, ::skip_z], Er[::skip_r, ::skip_z], scale=None)
    axs[2].set_title(f"Vector (E_z, E_r) {title_suffix}")
    axs[2].set_xlabel("z"); axs[2].set_ylabel("r"); axs[2].set_aspect('auto')

    plt.tight_layout()
    plt.show()


def plot_ring_metrics(times, r0_list, T_list):
    """
    Quick diagnostic plots for ring radius and toroidality over time.
    """
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(times, r0_list)
    ax[0].set_title("Ring radius r0(t)")
    ax[0].set_xlabel("time (sim units)"); ax[0].set_ylabel("r0")

    ax[1].plot(times, T_list)
    ax[1].set_title("Toroidality index T(t)")
    ax[1].set_xlabel("time (sim units)"); ax[1].set_ylabel("T in [0,1]")

    plt.tight_layout(); plt.show()
    

def plot_energy_map(r_coords, z_coords, Er, Ez, Hphi, title="Energy density"):
    U = energy_density(Er, Ez, Hphi).T
    plt.figure(figsize=(7,5))
    plt.imshow(U, extent=[z_coords[0], z_coords[-1], r_coords[0], r_coords[-1]],
               origin='lower', aspect='auto', cmap='magma')
    plt.xlabel("z"); plt.ylabel("r"); plt.title(title); plt.colorbar(label="u")
    plt.show()
    
    
# -----------------------------
# GAUSS-LAW Ez RECONSTRUCTION
# -----------------------------

def reconstruct_Ez_from_Gauss(Er, r_coords, dz, Ez_sim=None):
    """
    Reconstruct Ez(r,z) from Gauss's law in cylindrical coords (m=0):
        (1/r) d(r Er)/dr + dEz/dz = 0   →   Ez ≈ - ∫ (1/r d(rEr)/dr) dz
    Here we approximate dEz/dz = - (1/r) d(rEr)/dr, then integrate along z.
    
    Args:
        Er       : 2D array [Nr, Nz] of Er field
        r_coords : 1D array [Nr] of radial positions
        dz       : grid spacing in z
        Ez_sim   : (optional) simulated Ez array [Nr, Nz] to compare

    Returns:
        Ez_rec : reconstructed Ez array [Nr, Nz]
    """
    Nr, Nz = Er.shape
    Ez_rec = np.zeros_like(Er)

    # compute d(r Er)/dr
    for i in range(1, Nr-1):
        r = r_coords[i]
        d_rEr_dr = (r_coords[i+1]*Er[i+1,:] - r_coords[i-1]*Er[i-1,:]) / (r_coords[i+1]-r_coords[i-1])
        div_term = d_rEr_dr / r
        # integrate over z (cumulative trapezoid)
        Ez_rec[i,:] = - cumulative_trapezoid(div_term, dx=dz, initial=0)

    # at r=0, enforce regularity Ez=0
    Ez_rec[0,:] = 0.0

    # optional: compare
    if Ez_sim is not None:
        diff = np.linalg.norm(Ez_rec - Ez_sim) / (np.linalg.norm(Ez_sim) + 1e-30)
        print(f"Gauss-law Ez reconstruction relative error: {diff:.3e}")

    return Ez_rec


# -----------------------------
# SKYRMION NUMBER CALCULATION
# -----------------------------

def compute_skyrmion_number(Er, Ez, Hphi, r_coords, z_coords):
    """
    Compute a skyrmion number-like topological invariant for the 2D field map.
    Treat the normalized vector field n = (Er, Ez, Hphi) / |...| as a mapping S^2 -> S^2.
    
    Formula (discrete):
        Q = (1/4π) ∫ n · (∂_r n × ∂_z n) dr dz
    
    Returns:
        Q : scalar skyrmion number
    """
    Nr, Nz = Er.shape
    dr = r_coords[1] - r_coords[0]
    dz = z_coords[1] - z_coords[0]

    # normalize vector field
    V = np.stack([Er, Ez, Hphi], axis=0)  # (3, Nr, Nz)
    norm = np.sqrt(np.sum(V**2, axis=0)) + 1e-30
    n = V / norm

    # finite differences
    dnr = (np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)) / (2*dr)
    dnz = (np.roll(n, -1, axis=2) - np.roll(n, 1, axis=2)) / (2*dz)

    # cross product along last two axes
    cross = np.cross(np.moveaxis(dnr,0,-1), np.moveaxis(dnz,0,-1))  # shape (Nr,Nz,3)
    integrand = np.sum(np.moveaxis(n,0,-1) * cross, axis=-1)  # n·(∂r n × ∂z n)

    # integrate
    Q = np.sum(integrand) * dr * dz / (4*np.pi)
    return Q




# --- 4. MAIN WORKFLOW ---
def main():
    """
    Main function to define parameters and orchestrate the simulation workflow.
    """
    # Simulation Parameters
    resolution = 80
    dpml = 1.0
    sr = 0.5  # Radial size
    sz = 3.0  # Axial size
    cell_size = mp.Vector3(sr + dpml, 0, sz + 2 * dpml)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X), 
                  mp.PML(thickness=dpml, direction=mp.Z)]

    # Antenna Parameters
    feed_inner_radius = 0.01
    feed_outer_radius = 0.03
    horn_inner_radius = 0.07
    horn_outer_radius = 0.16
    horn_height = 0.30

    # Source Parameters
    fcen = 6e9 # Note: Meep units are based on c=1. For realism, you'd scale dimensions.
               # Here we treat units as arbitrary but consistent.
    # For a frequency of 6e9, the wavelength is ~0.05. Let's use a normalized frequency.
    # Let 1 unit = 1 cm. Then wavelength is 5 units. fcen = 1/5 = 0.2
    fcen_meep = 0.2


    # --- Execute Simulation Steps ---
    print("Step 1: Creating horn antenna geometry...")
    horn_geometry = make_horn_antenna_geometry(
        r_in_start=feed_inner_radius, r_in_end=horn_inner_radius,
        r_out_start=feed_outer_radius, r_out_end=horn_outer_radius,
        h=horn_height, taper_profile='linear'
    )
    visualize_geometry_2d(horn_geometry, horn_height, horn_outer_radius)

    print("Step 2: Defining custom source...")
    # Use the normalized MEEP frequency
    source_waveform = create_source_waveform(fcen_meep)
    sources = define_sources(source_waveform, feed_inner_radius, feed_outer_radius, sz, dpml)

    print("Step 3: Setting up MEEP simulation...")
    sim = setup_simulation(cell_size, resolution, pml_layers, horn_geometry, sources)
    
        # ---- 3.4 monitors ----
    # (A) time-series probes at a few radii / z's
    probe_radii = [0.0 + 1e-3, 0.05, 0.10, 0.15]           # tweak to your geometry
    probe_zs    = [-sz/2 + dpml + 0.2, 0.0, sz/2 - dpml - 0.2]
    probes, probe_cb = add_time_series_probes(sim, probe_radii, probe_zs)

    # (B) field snapshots every dt
    every_dt = 1.0  # sim-time units; adjust to your fcen_meep; ~lambda/10/c time is a good start
    frames, snap_cb = add_field_snapshotter(sr, sz, every_dt, save_prefix="snap")

    # (C) power flux monitors (use your normalized center frequency)
    fcen_monitor = fcen_meep
    flux_handles = add_flux_monitors(sim, sr, sz, dpml, fcen_monitor)

    # (D) near-to-far box
    n2f = add_near_to_far(sim, sr, sz, dpml, fcen_monitor)

    # ---- run with callbacks ----
    sim.run(
        mp.at_every(every_dt, snap_cb),
        mp.at_every(0.2*every_dt, probe_cb),  # sample probes more often if you like
        until=50
    )

    # ---- post-run: grab flux/N2F if needed ----
    # Example: total power through z-plane at fcen
    # (index 0 since nfreq=1)
    Pz = mp.get_fluxes(flux_handles["flux_z"])[0]
    Pr_side_top = mp.get_fluxes(flux_handles["flux_r_top"])[0]
    Pr_side_bot = mp.get_fluxes(flux_handles["flux_r_bot"])[0]
    print(f"Flux: Pz={Pz:.4g}, Pr(top)={Pr_side_top:.4g}, Pr(bot)={Pr_side_bot:.4g}")

    # Example: far-field at an angle (theta wrt z-axis) and distance R
    # ff = sim.get_farfield(n2f, mp.Vector3(x=0.0, z=10.0))  # E fields in far zone
    # print("Far-field Ez at (0,10):", ff.z)

    # ---- quick analytics on snapshots ----
    # Reconstruct coordinate arrays for your plotting volume
    Nr = frames["Er"][0].shape[0]
    Nz = frames["Er"][0].shape[1]
    r_coords = np.linspace(0, sr, Nr)
    z_coords = np.linspace(-sz/2, +sz/2, Nz)

    # Track ring radius and toroidality over time
    r0_list, T_list = [], []
    for k, (Erk, Ezk, Hphik) in enumerate(zip(frames["Er"], frames["Ez"], frames["Hphi"])):
        r0, T = ring_radius_and_index(Erk, Ezk, Hphik, r_coords, z_coords,
                                      z_pick=0.0, r_ring_band=(0.03, 0.25), r_axis_band=(0.0, 0.01))
        r0_list.append(r0); T_list.append(T)

    plot_ring_metrics(frames["t"], r0_list, T_list)

    # show one nice snapshot near the pulse maximum
    peak_idx = int(np.argmax(T_list))
    plot_snapshot_fields(r_coords, z_coords,
                         frames["Er"][peak_idx], frames["Ez"][peak_idx], frames["Hphi"][peak_idx],
                         title_suffix=f"(t = {frames['t'][peak_idx]:.2f})")
    
    # Plot energy density (donut structure)
    plot_energy_map(r_coords, z_coords, Erk, Ezk, Hphik,
                    title=f"Energy density at t={frames['t'][peak_idx]:.2f}")



    print("Step 4: Running simulation...")
    sim.run(until=100) # Increased runtime to see wave propagate further

    print("Step 5: Visualizing results...")
    plot_simulation_results(sim, sr, sz)
    
    print("Simulation finished.")
    
        # ----- GAUSS LAW CHECK -----
    Erk, Ezk, Hphik = frames["Er"][peak_idx], frames["Ez"][peak_idx], frames["Hphi"][peak_idx]
    dz = z_coords[1] - z_coords[0]
    Ez_rec = reconstruct_Ez_from_Gauss(Erk, r_coords, dz, Ez_sim=Ezk)

    # plot simulated vs reconstructed Ez at mid-radius
    plt.figure(figsize=(6,4))
    r_idx = np.argmin(np.abs(r_coords - 0.1))  # pick r ~ 0.1
    plt.plot(z_coords, Ezk[r_idx,:], label="Ez (sim)")
    plt.plot(z_coords, Ez_rec[r_idx,:], '--', label="Ez (reconstructed)")
    plt.xlabel("z"); plt.ylabel("Ez"); plt.legend(); plt.title("Gauss-law Ez check")
    plt.show()

    # ----- SKYRMION NUMBER -----
    Q = compute_skyrmion_number(Erk, Ezk, Hphik, r_coords, z_coords)
    print(f"Skyrmion number Q ≈ {Q:.3f}")


if __name__ == '__main__':
    main()


