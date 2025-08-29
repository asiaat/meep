import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_hollow_horn(r_in_start, r_in_end, r_out_start, r_out_end, h,
                     cthik=0.0, metal=True, n_cyl=2.0, n_slices=100):
    """
    Return geometry list for a hollow horn (flared cylinder) in cylindrical coordinates.

    r_in_start  : inner radius at the base
    r_in_end    : inner radius at the horn opening (top)
    r_out_start : outer radius at the base
    r_out_end   : outer radius at the horn opening (top)
    h           : height of the horn
    cthik       : offset multiplier in r-direction
    metal       : True for metal, False for dielectric
    n_cyl       : refractive index if dielectric
    n_slices    : number of slices to approximate the flare
    """
    if metal:
        material = mp.metal
    else:
        material = mp.Medium(index=n_cyl)

    geometry = []

    for i in range(n_slices):
        slice_height = h / n_slices
        z_mid = (i + 0.5) * slice_height

        # Linear taper for outer and inner radii
        slice_r_outer = r_out_start + (r_out_end - r_out_start) * (z_mid / h)
        slice_r_inner = r_in_start + (r_in_end - r_in_start) * (z_mid / h)

        # Outer wall (metal)
        geometry.append(
            mp.Block(
                material=material,
                center=mp.Vector3(cthik * slice_r_outer, 0, z_mid),
                size=mp.Vector3(slice_r_outer, 0, slice_height),
            )
        )

        # Hollow core (air)
        if slice_r_inner > 0:
            geometry.append(
                mp.Block(
                    material=mp.Medium(index=1.0),
                    center=mp.Vector3(cthik * slice_r_outer, 0, z_mid),
                    size=mp.Vector3(slice_r_inner, 0, slice_height),
                )
            )

    return geometry


def plot_hollow_horn_schematic(r_in_start, r_in_end,
                               r_out_start, r_out_end, h, n_slices=50):
    """
    Plot schematic (cross-section + 3D) of hollow horn geometry with tapered inner + outer walls.
    """
    fig = plt.figure(figsize=(12, 6))

    # --- 2D Cross-section (r vs z) ---
    ax1 = fig.add_subplot(121)
    z_vals = np.linspace(0, h, n_slices)
    r_outer_vals = np.linspace(r_out_start, r_out_end, n_slices)
    r_inner_vals = np.linspace(r_in_start, r_in_end, n_slices)

    ax1.plot(r_outer_vals, z_vals, "k-", lw=2, label="Outer wall")
    ax1.plot(r_inner_vals, z_vals, "r--", lw=2, label="Inner hollow")

    ax1.fill_betweenx(z_vals, r_inner_vals, r_outer_vals, color="grey", alpha=0.5)
    ax1.set_xlabel("Radius r")
    ax1.set_ylabel("Height z")
    ax1.set_title("Horn cross-section")
    ax1.legend()

    # --- 3D Visualization ---
    ax2 = fig.add_subplot(122, projection="3d")
    theta = np.linspace(0, 2*np.pi, 60)
    z_vals = np.linspace(0, h, n_slices)
    Theta, Z = np.meshgrid(theta, z_vals)

    # Outer surface
    R_outer_vals = np.linspace(r_out_start, r_out_end, n_slices)
    R_outer = np.tile(R_outer_vals[:, None], (1, len(theta)))
    X_outer = R_outer * np.cos(Theta)
    Y_outer = R_outer * np.sin(Theta)
    ax2.plot_surface(X_outer, Y_outer, Z, color="gray", alpha=0.5, edgecolor="k")

    # Inner hollow surface
    R_inner_vals = np.linspace(r_in_start, r_in_end, n_slices)
    R_inner = np.tile(R_inner_vals[:, None], (1, len(theta)))
    X_inner = R_inner * np.cos(Theta)
    Y_inner = R_inner * np.sin(Theta)
    ax2.plot_surface(X_inner, Y_inner, Z, color="white", alpha=1.0, edgecolor="k")

    ax2.set_title("3D Hollow Horn (tapered inner+outer walls)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


# --- Example usage ---
if __name__ == "__main__":
    r_in_start = 2     # inner radius at base
    r_in_end = 6       # inner radius at top
    r_out_start = 8    # outer radius at base
    r_out_end = 20     # outer radius at top
    h = 36             # height

    geom = make_hollow_horn(r_in_start, r_in_end,
                            r_out_start, r_out_end, h, metal=True)

    plot_hollow_horn_schematic(r_in_start, r_in_end,
                               r_out_start, r_out_end, h)
