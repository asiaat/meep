"""
Enhanced Meep simulation script for cylindrical-symmetric HN toroidal pulse.

Enhancements:
1. Visualization of the toroidal pulse levitating through the source point.
2. Outputs saved with date-time components for serial test examples.
3. Added analytics for toroidal shape formation over time.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from datetime import datetime

# -------------------------
# Parameters
# -------------------------
f_min_real = 0.5
f_max_real = 10.0
f_ref_real = 10.0
f_min = f_min_real / f_ref_real
f_max = f_max_real / f_ref_real
f_center = 0.5 * (f_min + f_max)
f_width = (f_max - f_min) * 0.55

print(f"Normalized center frequency: {f_center:.3f}, bandwidth fwidth={f_width:.3f}")

param_a = 0.6
resolution = 30
sr = 12.0
sz = 20.0
pml_thickness = 2.0
run_time = 200

# Date component for serial test examples
date_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
outdir = f"meep_hn_cyl_output_{date_tag}"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# HN TM envelope
# -------------------------
def hn_tm_cyl(rho, z, a=param_a):
    denom = (rho**2 + z**2 + a**2)**2.5
    if denom == 0:
        return (0.0, 0.0, 0.0)
    E_rho = 3.0 * rho * z / denom
    E_z   = (2.0*z**2 - rho**2 + a**2) / denom
    H_phi = 3.0 * rho * z / denom
    return (E_rho, E_z, H_phi)

def e_rho_weight(r):
    E_rho, _, _ = hn_tm_cyl(r.x, r.z)
    return float(E_rho)

def e_z_weight(r):
    _, E_z, _ = hn_tm_cyl(r.x, r.z)
    return float(E_z)

# -------------------------
# Simulation setup
# -------------------------
cell = mp.Vector3(sr, 0, sz)
sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    dimensions=mp.CYLINDRICAL,
    m=0,
    boundary_layers=[mp.PML(pml_thickness)],
)

src_time = mp.GaussianSource(frequency=f_center, fwidth=f_width)
sim.sources = [
    mp.Source(src=src_time, component=mp.Er, center=mp.Vector3(0,0,-4), size=mp.Vector3(sr,0,6), amp_func=e_rho_weight),
    mp.Source(src=src_time, component=mp.Ez, center=mp.Vector3(0,0,-4), size=mp.Vector3(sr,0,6), amp_func=e_z_weight),
]

flux_z = 6.0
flux_reg = mp.FluxRegion(center=mp.Vector3(0,0,flux_z), size=mp.Vector3(sr-2,0,0))
flux = sim.add_flux(f_center, f_width, 100, flux_reg)

probe_pt = mp.Vector3(0,0,flux_z)
probe_time, probe_field = [], []

# Toroid analytics storage
toroid_radius_over_time = []
toroid_peak_field_over_time = []

# -------------------------
# Callbacks
# -------------------------
def probe_cb(sim):
    probe_time.append(float(sim.meep_time()))
    Ez_val = float(np.real(sim.get_field_point(mp.Ez, probe_pt)))
    probe_field.append(Ez_val)

    # Toroid analytics: find radius of max Ez around center
    r_vals = np.linspace(0, sr/2, 100)
    Ez_profile = np.array([float(np.real(sim.get_field_point(mp.Ez, mp.Vector3(r,0,0)))) for r in r_vals])
    peak_idx = np.argmax(Ez_profile)
    toroid_radius_over_time.append(r_vals[peak_idx])
    toroid_peak_field_over_time.append(Ez_profile[peak_idx])

frames = []
def snapshot_cb(sim):
    # Levitating visualization: center field slice at source plane (z=0)
    field = sim.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(sr,0,sr), component=mp.Ez)
    if field is not None:
        frames.append(np.array(field, dtype=float))

# -------------------------
# Run simulation
# -------------------------
print("Running cylindrical HN pulse sim with toroidal analytics and levitating visualization...")
sim.run(mp.at_every(0.5, probe_cb), mp.at_every(5, snapshot_cb), until=run_time)

# -------------------------
# Save & plot
# -------------------------
np.save(os.path.join(outdir, f"probe_time_{date_tag}.npy"), np.array(probe_time, dtype=float))
np.save(os.path.join(outdir, f"probe_ez_{date_tag}.npy"), np.array(probe_field, dtype=float))
np.save(os.path.join(outdir, f"toroid_radius_{date_tag}.npy"), np.array(toroid_radius_over_time, dtype=float))
np.save(os.path.join(outdir, f"toroid_peak_field_{date_tag}.npy"), np.array(toroid_peak_field_over_time, dtype=float))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(probe_time, probe_field)
plt.xlabel('time (Meep units)')
plt.ylabel('Ez at probe (axis)')
plt.title('Probe Ez time series')

plt.subplot(1,2,2)
plt.plot(probe_time, toroid_radius_over_time, label='Toroid radius')
plt.plot(probe_time, toroid_peak_field_over_time, label='Peak Ez')
plt.xlabel('time (Meep units)')
plt.ylabel('Toroid analytics')
plt.title('Toroid formation over time')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"probe_and_toroid_analytics_{date_tag}.png"), dpi=150)

# Existing flux and GIF saving logic
freqs = mp.get_flux_freqs(flux)
flux_vals = mp.get_fluxes(flux)
plt.figure()
plt.plot(freqs, flux_vals)
plt.xlabel('frequency (Meep units)')
plt.ylabel('flux')
plt.title('Flux spectrum at z={}'.format(flux_z))
plt.savefig(os.path.join(outdir, f"flux_spectrum_{date_tag}.png"), dpi=150)

if frames:
    images = []
    for f in frames:
        fig, ax = plt.subplots()
        im = ax.imshow(f.T, origin='lower', aspect='auto', extent=[-sr/2,sr/2,-sr/2,sr/2])
        ax.set_xlabel('x (r)')
        ax.set_ylabel('y (z)')
        fig.colorbar(im, ax=ax)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        rgb = buf[:,:,[1,2,3]]
        images.append(rgb)
        plt.close(fig)
    imageio.mimsave(os.path.join(outdir, f"ez_levitating_{date_tag}.gif"), images, fps=5)

print(f"Outputs saved in {outdir}")
