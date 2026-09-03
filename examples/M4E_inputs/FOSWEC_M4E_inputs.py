# -*- coding: utf-8 -*-
"""
Created on 8/19/2026

@author: jtgrasb
"""
import sys
import os

# Add the source and examples directory to the path to run examples
source_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, source_dir)

# External libraries
import numpy as np
import sympy as sym

# This module libraries
import multibody as mb
from multibody import JointSystem, normalize_prismatic
from multibody import symvars_definition

# Do not modify 
Initial_Points  = {} 
Force           = {} 
# End of Do not Modify

# Example 3
Reference_frame_Origin = np.array([0,0])

# Bodies definition
dataNames   = [] # Define symbolic variables for force points (for example, b1, b2, b3).
BodyDataSym = symvars_definition(dataNames,globals())

platformCG = [0, 0, -0.8]
flapCenterDistanceApart = 1.44
flapDraft  = 0.59 
cgHeightAboveHinge = 0.17
flap1CG     = [-flapCenterDistanceApart/2, 0, -flapDraft+cgHeightAboveHinge] # from water surface/origin
flap2CG     = [flapCenterDistanceApart/2, 0, -flapDraft+cgHeightAboveHinge] # from water surface/origin
flapHingeDepth = 0.59

joints              = [[0, 1],[1, 2],[1, 3]] # Joint connectivity: [parent, child]
types               = ['F', 'R', 'R']  # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[0, platformCG[2]],[flap1CG[0],-platformCG[2]-flapHingeDepth],[flap2CG[0],-platformCG[2]-flapHingeDepth]]# [[3, 0],[0,1]] # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[np.nan, np.nan],[0,flapHingeDepth+flap1CG[2]],[0,flapHingeDepth+flap2CG[2]]] # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
prismatic_direction = normalize_prismatic(prismatic_direction)

# Points definition
PointNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
PointsSym   = symvars_definition(PointNames,globals())

# Define the structure for initial points.
Initial_Points          = {}
# Ground points: these are fixed and given as [x, z] coordinates.
Initial_Points["GR"]    = []

# Body-defined points ("BD"): we use a dictionary where each key (an integer)
# corresponds to a body and the value is a list of points.
Initial_Points["BD"]    = {}

# Forces definition
ForceNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
ForceSym    = symvars_definition(ForceNames,globals())

# Define the structure for Force.
Force = {}
# Forces applied on body-defined points (PointsBD):
Force["PointsBD"] = []

# Forces applied at the center of gravity (CG):
Force["CG"] = []

# Tension springs: stored as a list of tuples: (connection, [l0, stiffness]).
# Here "BD41" means (for example) the 1st point on body 4.
Force["TensionSpring"] = [] 

# Tension dampers: here we have two pairs.
# The first pair uses a constant damping coefficient.
# The second uses a lambda function to represent a nonlinear damping coefficient.
Force["TensionDamper"] = []

# Torsion springs: here the first element is a list of parameters and the second element is a lambda.
Force["TorsionSpring"] = []

# Torsion dampers:
Force["TorsionDamper"] = []

ForcesPointsSym = PointsSym + ForceSym 

# Initial conditions
# Create the JointSystem using the from_data class method.
joint_system        = JointSystem.from_data(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction) # DO NOT MODIFY
Q, QD, _, NDOF, _   = joint_system.coordinate_finder() # DO NOT MODIFY

# If you are unsure what are the systems DOF run the example and you will see on screen
ic                  = np.zeros(2*sum(NDOF)) # Multiplied by 2 because is position and velocity
ic[2]               = 0 * np.pi / 180  # Initial position of the first joint


# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))

# Simulation
TimeStep    = 0.01
tspan       = 300.
g           = 9.81                      # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = np.ones(len(types))

# Animation
animation_on    = 1 # Display animation:1, 0-otherwise
SaveMovieOn     = 'check.gif' #None # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 5 # how many times faster are we plotting w.r.t to simulation time

###########################################################################
######################### END OF USER DEFINITION ##########################
###########################################################################

#%% Checks section: ensures everything is defined correctly
t = sym.symbols('t')

mb.checks.bodies(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction)
mb.checks.points_forces(joints, types, Initial_Points, Force, t)

NBodies = len(joints)

if len(m0) != NBodies or len(J0) != NBodies or not np.all(m0) or not np.all(J0):
    raise ValueError("The mass matrix has zero-valued entries check m0 and J0")
    
'''
The following sections are for user visualization but unnecessary for the
main to run. Uncomment if you want to check how your bodies tables is defines
or if unsure on what the initial conditions are and what is the order
'''
#%% Display tables
# Print the joint system for inspection.
print('\nBodies and joints table:')
joint_system.display_table()

# Display points table
print('\nPoints table:')
mb.tables.points_table(Initial_Points)
print('\nForces table:')
mb.tables.force_table(Force)

#%% Problem variables and required initial conditions
ic_vars = Q + QD

print('\nThe variables that require initial conditions, in this order, are:\n' + str(ic_vars) + '\n')

#%% Documentation
'''
 Bodies data: defining the joint and body configuration
 data.Joints:              List where each element represents a joint [Parent, Child]
 data.Type:                List specifying joint types ('F' for floating, 'R' for revolute, 'P' for prismatic)
 data.ParentCGtoJoint:     List containing vectors from parent CG to joint location
 data.JointtoChildCG:      List containing vectors from joint to child CG
 data.prismatic_direction: Unit vectors for prismatic joints, use [nan,nan] if NO prismatic joint

 Points data: dictionary
 You can use the variables in PointNames to define points location, Ex: PointNames = ["b1"]
 Initial points of the system, categorized by grounded points (GR) and body points (BD)
 DEFINE THESE EMPTY IF NO POINTS ARE REQUIRED, Ex: Initial_Points['GR'] = []
 Initial_Points["GR"] = [[20, 0], [b1, 0]]
 Initial_Points["BD"] is a DICTIONARY where each entry is the body number they belong to
 Initial_Points["BD"][i], 'i', is the body and the rest is defined as 'GR' points

 Forces definition: dictionary
 You can use the variables in ForceNames to define points location, Ex: ForceNames = ["f1"]
 DEFINE ALL FIELDS, if empty use: Force['PointsBD ']= []
 Force.PointsBD       = list of lists: [[Initial_points.BD(i),Initial_points.BD(j),Fx,Fy,Mz]]
 Force.CG             = list of lists: [[Initial_points.BD(i),Initial_points.BD(j),Fx,Fy,Mz]]
 Force.TensionSpring  = list of tuples: [(('Point1','Point2'),[L_0,k])]
 Force.TensionDamper  = list of tuples: [(('Point1','Point2'),C_0)]
 Force.TorsionSpring  = list of tuples: [(('Body1','Body2'),[theta_0,k])]
 Force.TorsionDamper  = list of tuples: [(('Body1','Body2'),C_0)]
 LEGEND:
 Fx,Fy,Mz : force components
 L_0:       spring undeformed length
 k:         spring constant 
 C_0:       damping coefficient

 Numerical inputs for simulation
 m0:        column vector with the mass of each body    
 J0:        column vector with the inertia of each body
 gVec:      column vector of 0s and 1s, with 0 indicating if a body is not
            subject to gravity (let's say it's floating)
'''