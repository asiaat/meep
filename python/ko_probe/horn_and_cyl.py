import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Horn dimensions (example from figure b) ---
R_aperture = 40       # mm (radius at mouth)
R_throat = 13.65      # mm (radius at throat)
L = 119               # mm (axial length)

# Cylinder dimensions
cyl_radius = R_throat   # same as horn throat radius
cyl_height = 1.73 * L   # cylinder height (1.73 × horn height)

# piecewise function r(z)
def r_of_z(z):
    # linear taper from throat to aperture
    return R_throat + (R_aperture - R_throat) * (z / L)

# --- Cylindrical coordinates ---
nz = 200   # number of z samples
nphi = 200 # number of angular samples

# Horn section
z_horn = np.linspace(0, L, nz)
phi = np.linspace(0, 2*np.pi, nphi)
Z_horn, PHI_horn = np.meshgrid(z_horn, phi)
R_horn = r_of_z(Z_horn)
X_horn = R_horn * np.cos(PHI_horn)
Y_horn = R_horn * np.sin(PHI_horn)

# Cylinder section (attached to throat)
z_cyl = np.linspace(-cyl_height, 0, nz)
Z_cyl, PHI_cyl = np.meshgrid(z_cyl, phi)
R_cyl = cyl_radius * np.ones_like(Z_cyl)
X_cyl = R_cyl * np.cos(PHI_cyl)
Y_cyl = R_cyl * np.sin(PHI_cyl)

# --- Plot ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# horn surface
ax.plot_surface(X_horn, Y_horn, Z_horn, color='silver', edgecolor='k', alpha=0.8, linewidth=0.2)

# cylinder surface
ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='lightblue', edgecolor='k', alpha=0.7, linewidth=0.2)

# formatting
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Conical Horn with Attached Cylinder (Cylindrical Coordinates)')

plt.show()