import meep as mp
import pyvista as pv
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

def visualize_meep_geometry(geometry, show_material=True):
    """
    Visualize a Meep geometry list with PyVista.

    Supports: mp.Block, mp.Cylinder, mp.Sphere, mp.Cone
    """
    plotter = pv.Plotter()

    for i, obj in enumerate(geometry):
        # --- choose color ---
        if not show_material:
            color = "gray"
        else:
            if obj.material == mp.metal:
                color = "silver"
            elif isinstance(obj.material, mp.Medium) and getattr(obj.material, "epsilon_diag", None):
                color = "green"   # dielectric
            else:
                color = "lightblue"  # air or generic

        # --- Block ---
        if isinstance(obj, mp.Block):
            sx, sy, sz = obj.size.x, obj.size.y, obj.size.z
            cx, cy, cz = obj.center.x, obj.center.y, obj.center.z
            sy = sy if sy > 0 else 0.01  # avoid degenerate case
            block = pv.Cube(center=(cx, cy, cz), x_length=sx, y_length=sy, z_length=sz)
            plotter.add_mesh(block, opacity=0.6, color=color, show_edges=True)

        # --- Cylinder ---
        elif isinstance(obj, mp.Cylinder):
            r = obj.radius
            h = obj.height
            cx, cy, cz = obj.center.x, obj.center.y, obj.center.z
            cyl = pv.Cylinder(center=(cx, cy, cz), radius=r, height=h, direction=(0,0,1))
            plotter.add_mesh(cyl, opacity=0.6, color=color, show_edges=True)

        # --- Sphere ---
        elif isinstance(obj, mp.Sphere):
            r = obj.radius
            cx, cy, cz = obj.center.x, obj.center.y, obj.center.z
            sph = pv.Sphere(radius=r, center=(cx, cy, cz))
            plotter.add_mesh(sph, opacity=0.6, color=color, show_edges=True)

        # --- Cone ---
        elif isinstance(obj, mp.Cone):
            r1 = obj.radius  # base radius
            r2 = obj.radius2 # top radius
            h  = obj.height
            cx, cy, cz = obj.center.x, obj.center.y, obj.center.z

            cone = pv.Cone(center=(cx, cy, cz - h/2), 
                           direction=(0,0,1), 
                           height=h, 
                           radius=r1, 
                           capping=True)

            if r2 > 0:
                # Truncated cone (frustum)
                cone = pv.Cone(center=(cx, cy, cz - h/2), 
                               direction=(0,0,1), 
                               height=h, 
                               radius=r1, 
                               capping=True).boolean_cut(
                           pv.Cone(center=(cx, cy, cz - h/2), 
                                   direction=(0,0,1), 
                                   height=h, 
                                   radius=r2, 
                                   capping=True))

            plotter.add_mesh(cone, opacity=0.6, color=color, show_edges=True)

        else:
            print(f"⚠️ Unsupported geometry type: {type(obj)}")

    plotter.show_grid()
    plotter.show()
    
    
## showtime    
    
'''
r_in_start = 2     # inner radius at base
r_in_end = 6       # inner radius at top
r_out_start = 8    # outer radius at base
r_out_end = 20     # outer radius at top
h = 36             # height

geom = make_hollow_horn(r_in_start, r_in_end,
                            r_out_start, r_out_end, h, metal=True)

geom = [
    mp.Cone(
        radius=2.0,
        radius2=0.5,    # top radius (0 for sharp tip)
        height=3.0,
        center=mp.Vector3(0,0,1.5),
        material=mp.metal
    )
]


'''


def plot_hollow_metal_cylinder_with_cone():
    """
    Creates and displays a 3D plot of a hollow metal cylinder
    with a hollow conical extension on top.
    """
    # Cylinder parameters (mm)
    outer_radius = 2.0    # outer radius of cylinder
    inner_radius = 0.65   # hollow radius of cylinder
    height_mm = 10.0      # height of cylinder

    # Cone parameters (mm)
    cone_height = 36.0
    cone_outer_r1 = 2.0   # bottom outer radius (matches cylinder outer)
    cone_outer_r2 = 15.0   # top outer radius
    cone_inner_r1 = 1.0   # bottom inner radius (hollow)
    cone_inner_r2 = 9.0   # top inner radius (hollow)

    resolution = 10  # pixels/mm

    # ------------------ Geometry ------------------
    # Cylinder outer (metal replaced with finite epsilon for visualization)
    outer_cyl = mp.Cylinder(
        radius=outer_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.Medium(epsilon=10.0)   # use mp.metal for real sim
    )

    # Cylinder inner (air hollow)
    inner_cyl = mp.Cylinder(
        radius=inner_radius,
        height=height_mm,
        axis=mp.Vector3(0, 0, 1),
        material=mp.air
    )

    # Cone outer (metal)
    outer_cone = mp.Cone(
        radius=cone_outer_r1,
        radius2=cone_outer_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.Medium(epsilon=10.0)   # use mp.metal for real sim
    )

    # Cone inner (air hollow)
    inner_cone = mp.Cone(
        radius=cone_inner_r1,
        radius2=cone_inner_r2,
        height=cone_height,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, 0.5 * height_mm + 0.5 * cone_height),
        material=mp.air
    )

    geometry = [outer_cyl, inner_cyl, outer_cone, inner_cone]
    
    return geometry


geom = plot_hollow_metal_cylinder_with_cone()
visualize_meep_geometry(geom)

