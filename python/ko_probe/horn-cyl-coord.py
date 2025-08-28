import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Horn dimensions (example from figure b) ---
R_aperture = 40       # mm (radius at mouth)
R_throat = 13.65      # mm (radius at throat)
L = 119               # mm (axial length)

# piecewise function r(z)
def r_of_z(z):
    # linear taper from throat to aperture
    return R_throat + (R_aperture - R_throat) * (z / L)

# --- Cylindrical coordinates ---
nz = 200   # number of z samples
nphi = 200 # number of angular samples

z = np.linspace(0, L, nz)
phi = np.linspace(0, 2*np.pi, nphi)
Z, PHI = np.meshgrid(z, phi)

# radius profile
R = r_of_z(Z)

# convert to cartesian
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

# --- Plot ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# surface
ax.plot_surface(X, Y, Z, color='silver', edgecolor='k', alpha=0.8, linewidth=0.2)

# formatting
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Conical Horn in Cylindrical Coordinates')

plt.show()
