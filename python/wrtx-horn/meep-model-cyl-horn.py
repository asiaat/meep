import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def make_horn_with_cylinder(r_in=10, wall_thickness=2, cyl_height=40,
                            cone_height=60, cone_radius=30,
                            n_slices=60):
    """
    Create a hollow cylindrical waveguide with conical horn on top.
    Cylindrical coords version (mp.CYLINDRICAL).
    
    Parameters:
        r_in          : inner radius of the hollow cylinder
        wall_thickness: thickness of the metal wall
        cyl_height    : height of the cylindrical part
        cone_height   : height of the conical horn part
        cone_radius   : final outer radius at horn mouth
        n_slices      : number of slices to approximate the cone
    """
    metal = mp.metal
    air   = mp.Medium(index=1.0)
    geometry = []

    # Outer cylinder wall (metal)
    geometry.append(mp.Block(
        size=mp.Vector3(r_in + wall_thickness, 0, cyl_height),
        center=mp.Vector3(0, 0, 0.5*cyl_height),
        material=metal
    ))
    # Hollow cylinder inside (air)
    geometry.append(mp.Block(
        size=mp.Vector3(r_in, 0, cyl_height),
        center=mp.Vector3(0, 0, 0.5*cyl_height),
        material=air
    ))

    # Conical flare approximation (hollow cone as metal shell)
    for i in range(n_slices):
        z0 = cyl_height + i * cone_height/n_slices
        zc = z0 + 0.5 * cone_height/n_slices
        slice_h = cone_height/n_slices
        
        # outer and inner radii at this height
        r_outer = r_in + wall_thickness + (cone_radius - (r_in + wall_thickness)) * (i+1)/n_slices
        r_inner = r_in + (cone_radius - r_in) * (i+1)/n_slices

        # Metal slice
        geometry.append(mp.Block(
            size=mp.Vector3(r_outer, 0, slice_h),
            center=mp.Vector3(0, 0, zc),
            material=metal
        ))
        # Hollow (air) slice
        geometry.append(mp.Block(
            size=mp.Vector3(r_inner, 0, slice_h),
            center=mp.Vector3(0, 0, zc),
            material=air
        ))

    return geometry


def plot_horn_geometry(r_in=10, wall_thickness=2, cyl_height=40,
                       cone_height=60, cone_radius=30,
                       n_slices=60):
    """
    Visualize the hollow horn geometry as a 3D mesh.
    """
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # cylinder outer + inner
    z_cyl = np.linspace(0, cyl_height, 30)
    theta = np.linspace(0, 2*np.pi, 60)
    Theta, Z = np.meshgrid(theta, z_cyl)
    X_outer = (r_in + wall_thickness) * np.cos(Theta)
    Y_outer = (r_in + wall_thickness) * np.sin(Theta)
    X_inner = r_in * np.cos(Theta)
    Y_inner = r_in * np.sin(Theta)

    ax.plot_surface(X_outer, Y_outer, Z, color="gray", alpha=0.6)
    ax.plot_surface(X_inner, Y_inner, Z, color="lightblue", alpha=0.3)

    # cone outer + inner
    z_cone = np.linspace(cyl_height, cyl_height+cone_height, 30)
    r_outer = np.linspace(r_in+wall_thickness, cone_radius, len(z_cone))
    r_inner = np.linspace(r_in, cone_radius-wall_thickness, len(z_cone))

    for ri, ro, z in zip(r_inner, r_outer, z_cone):
        Xo = ro * np.cos(theta); Yo = ro * np.sin(theta)
        Xi = ri * np.cos(theta); Yi = ri * np.sin(theta)
        ax.plot(Xo, Yo, z, color="k", alpha=0.5)
        ax.plot(Xi, Yi, z, color="b", alpha=0.5)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Conical Horn with Hollow Cylinder (Metal Walls)")
    ax.set_box_aspect([1,1,1])
    plt.show()


if __name__ == "__main__":
    # Example usage
    geom = make_horn_with_cylinder(r_in=10, wall_thickness=2,
                                   cyl_height=40, cone_height=60,
                                   cone_radius=30, n_slices=80)
    plot_horn_geometry(r_in=10, wall_thickness=2,
                       cyl_height=40, cone_height=60,
                       cone_radius=30, n_slices=80)
