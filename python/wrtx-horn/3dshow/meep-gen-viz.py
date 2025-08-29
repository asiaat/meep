import meep as mp
import pyvista as pv

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
    
    
geom = [
    mp.Cone(
        radius=2.0,
        radius2=0.5,    # top radius (0 for sharp tip)
        height=3.0,
        center=mp.Vector3(0,0,1.5),
        material=mp.metal
    )
]

visualize_meep_geometry(geom)

