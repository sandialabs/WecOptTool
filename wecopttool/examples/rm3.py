
import os

import numpy as np


# TODO: Create meshes programmatically, with ability to refine/coarsen.
cwd = os.path.dirname(os.path.realpath(__file__))

float_cog = np.array([0.0, 0.0, -0.7200])
float_moi = np.diag([20907301, 21306090.66, 37085481.11])
float_mesh_file = os.path.join(cwd, 'RM3', 'float.stl')

spar_cog = np.array([0.0, 0.0, -21.2900])
spar_moi = np.diag([20907301, 21306090.66, 37085481.11])
spar_mesh_file = os.path.join(cwd, 'RM3', 'spar.stl')
