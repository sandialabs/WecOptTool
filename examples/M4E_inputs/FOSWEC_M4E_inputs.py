"""FOSWEC Multibody for Everybody input file.
"""

import numpy as np
from multibody import JointSystem, normalize_prismatic

# Set FOSWEC parameters
platform_cg = [0, 0, -0.8]
flap_center_distance_apart = 1.44
flap_draft  = 0.59 
cg_height_above_hinge = 0.17
flap1_cg     = [-flap_center_distance_apart/2, 0, -flap_draft+cg_height_above_hinge] # from water surface/origin
flap2_cg     = [flap_center_distance_apart/2, 0, -flap_draft+cg_height_above_hinge] # from water surface/origin
flap_hinge_depth = 0.59

# Define joints
joints              = [[0, 1],[1, 2],[1, 3]] # Joint connectivity: [parent, child]
types               = ['F', 'R', 'R']  # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[0, platform_cg[2]],[flap1_cg[0],-platform_cg[2]-flap_hinge_depth],[flap2_cg[0],-platform_cg[2]-flap_hinge_depth]] # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[np.nan, np.nan],[0,flap_hinge_depth+flap1_cg[2]],[0,flap_hinge_depth+flap2_cg[2]]] # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
prismatic_direction = normalize_prismatic(prismatic_direction)

# Create the JointSystem using the from_data class method and define initial conditions.
joint_system        = JointSystem.from_data(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction) # DO NOT MODIFY
_, _, _, NDOF, _   = joint_system.coordinate_finder() # DO NOT MODIFY
ic                  = np.zeros(2*sum(NDOF)) # Multiplied by 2 because is position and velocity
ic[2]               = 0 * np.pi / 180  # Initial position of the first joint