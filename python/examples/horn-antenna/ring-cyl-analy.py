import argparse
import meep as mp
import matplotlib.pyplot as plt
import numpy as np


def plot_ring_schematic(r, w, sr, height=1):
    """Schematic views: top-down polar and longitudinal cross-section."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Polar (top-down) view ---
    theta = np.linspace(0, 2*np.pi, 500)
    axs[0] = plt.subplot(1, 2, 1, projection='polar')
    axs[0].set_theta_zero_location("N")
    axs[0].set_theta_direction(-1)
    axs[0].fill_between(theta, r, r + w, color='skyblue', alpha=0.7, label='Ring waveguide')
    axs[0].plot(theta, sr*np.ones_like(theta), 'k--', label='Computational boundary')
    axs[0].set_rmax(sr + 1)
    axs[0].set_rticks([0, r, r + w, sr])
    axs[0].set_rlabel_position(-22.5)
    axs[0].set_title("Top-down (polar) view")
    axs[0].legend(loc='upper right')

    # --- Longitudinal (cross-section) view ---
    axs[1].set_aspect('equal')
    axs[1].set_xlim(0, sr + 5)
    axs[1].set_ylim(0, height + 0.5)
    axs[1].add_patch(plt.Rectangle((r, 0), w, height, color='skyblue', alpha=0.7))
    axs[1].axvline(sr, color='k', linestyle='--', label='Computational boundary')
    axs[1].set_xlabel("Radius r")
    axs[1].set_ylabel("Height z")
    axs[1].set_title("Longitudinal cross-section")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def run_simulation(fcen, df, m, r, w, pad, dpml, n, resolution, with_ring=True):
    """Run Meep simulation with or without ring geometry."""
    sr = r + w + pad + dpml
    cell = mp.Vector3(sr, 0, 0)

    geometry = []
    if with_ring:
        geometry = [
            mp.Block(
                center=mp.Vector3(r + (w / 2)),
                size=mp.Vector3(w, mp.inf, mp.inf),
                material=mp.Medium(index=n),
            )
        ]

    pml_layers = [mp.PML(dpml)]

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(r - 0.5),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=m,
    )

    # Flux regions
    refl_fr = mp.FluxRegion(center=mp.Vector3(r - 1), size=mp.Vector3(0, 0, 0.05))
    tran_fr = mp.FluxRegion(center=mp.Vector3(r + w + 1), size=mp.Vector3(0, 0, 0.05))

    refl = sim.add_flux(fcen, df, 200, refl_fr)
    tran = sim.add_flux(fcen, df, 200, tran_fr)

    # Run until sources finish
    sim.run(until_after_sources=200)

    return sim, refl, tran


def main(args):
    # Parameters
    n = 3.4
    w = 1
    r = 1
    pad = 4
    dpml = 2
    resolution = 20

    fcen = args.fcen
    df = args.df
    m = args.m

    # Plot schematic first
    sr = r + w + pad + dpml
    #plot_ring_schematic(r, w, sr, height=1)

    # --- Reference run (no ring) ---
    sim_ref, refl_ref, tran_ref = run_simulation(fcen, df, m, r, w, pad, dpml, n, resolution, with_ring=False)
    ref_flux_refl = mp.get_fluxes(refl_ref)
    ref_flux_tran = mp.get_fluxes(tran_ref)

    # --- With ring ---
    sim_obj, refl_obj, tran_obj = run_simulation(fcen, df, m, r, w, pad, dpml, n, resolution, with_ring=True)
    obj_flux_refl = mp.get_fluxes(refl_obj)
    obj_flux_tran = mp.get_fluxes(tran_obj)

    freqs = mp.get_flux_freqs(refl_obj)

    # Normalize
    R = np.array(obj_flux_refl) / np.array(ref_flux_refl)
    T = np.array(obj_flux_tran) / np.array(ref_flux_tran)
    A = 1 - R - T

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, R, label="Reflection (R)")
    plt.plot(freqs, T, label="Transmission (T)")
    plt.plot(freqs, A, label="Absorption (1-R-T)")
    plt.xlabel("Frequency (a / λ)")
    plt.ylabel("Flux ratio")
    plt.title("Ring resonator reflection/transmission spectrum")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fcen", type=float, default=0.15, help="pulse center frequency")
    parser.add_argument("-df", type=float, default=0.1, help="pulse frequency width")
    parser.add_argument("-m", type=int, default=3, help="angular mode number (exp(i m phi))")
    args = parser.parse_args()
    main(args)
