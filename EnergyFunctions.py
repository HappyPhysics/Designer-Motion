# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:02:00 2017

@author: Salem

In this script I will I will define the energy of a deformation (quadratic) using the dynamical matrix. 

We want a given boundary deformation to have small energy so that it is easy to excite. Therefore,
for a given boundary we first minimize the energy with respect to the bulk vertices then we calculate a cost function
given by cost(spring constants) = E(boundary)/(lowest energy eigenvalue). 
Minimzing the cost function gives us the right spring constants that will result in a lattice that allows deform in the desired ways.
"""
import numpy as np
import numpy.random as npr
import importlib
import pickle
import LatticeMaking
from numpy import linalg as la
import scipy.optimize as op
importlib.reload(LatticeMaking)

from LatticeMaking import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#It's convineit to make these variables global, they also don't change after initializing
global U, rigidityMatrix, numberOfEdges, numberOfVerts

width = 20; #same as height

#A square lattice which is randomized 
(vertices, edges) = squareLattice(width, randomize=True)

numberOfVerts = len(set(list(edges.flatten())))
numberOfEdges = edges.size//2

rigidityMatrix = makeRigidityMat(vertices, edges) #defined in LatticeMaking

(boundaryIndices, bulkIndices) = getBoundaryVerts(edges) #defined in LatticeMaking

#generate the desired displacement
U = normalizeVec(npr.rand(2*numberOfVerts)) #defined in LatticeMaking, returns a normalized vector
U[:3] = np.array([0, 0, 0])  #excludes rotations and translations from the deformation, but not from DynMat eigenmodes
v = U[bulkIndices];  # These will be minimized over

k0 = np.ones(numberOfEdges)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Class for saving the data that is unique to each run
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RunData(object):
    """
    This helps me save the data from the run. It containt the lattice information which will be different for every run since it is randomly generated. 
    The boundary displacements are also generated everytime at random and need to be saved too.
    runName = name to be used in saving the fale.
    runWidth = width of the lattice, assumed to be the same as the height.
    vertices = the vertices of the lattice
    edges = edges of the lattice
    boundaryDisp = displacement vector corresponding to the boundary vertices
    """
    def __init__(self, runName="default", runWidth="default", vertices="default", edges="default", boundaryDisp="default", results="default"):
        self.name = runName
        self.width = runWidth
        self.vertices = vertices
        self.edges = edges
        self.boundaryDisp = boundaryDisp
        self.results = results
    
    def saveData(self):
        """easy to explain, uses picle.dump
        """
        pickle.dump(self, open(self.name, "wb"))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
  
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy from the dynamical matrix.
#================================================================================================================================================      
def loadData(fileName):
    """
    pickle.load
    """
    m = pickle.load(open(fileName, "rb"))
    return m
#================================================================================================================================================



#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy from the dynamical matrix.
#================================================================================================================================================
def energy(u, DynMat, boundaryInds = boundaryIndices):
    """
    v1 - be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    """
    u[boundaryInds] = U[boundaryInds]
    return np.dot(np.dot(u.transpose(), DynMat), u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy gradient from the dynamical matrix.
#================================================================================================================================================
def energy_Der(u, DynMat, boundaryInds = boundaryIndices):
    u[boundaryInds] = U[boundaryInds]
    return np.dot(DynMat, u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy Hessian from the dynamical matrix.
#================================================================================================================================================
def energy_Hess(u, DynMat, boundaryInds = boundaryIndices):
    return DynMat
#================================================================================================================================================
    
#================================================================================================================================================
# Returns the lowest eignevalue of the dynamical matrix, exluding the rigid motions of course.
#================================================================================================================================================
def lowestEigenVal(DynMat):    
    return (la.eigvalsh(DynMat)[3])
#================================================================================================================================================
    
    
#================================================================================================================================================
# This the cost function that will be minimized E/lowestEigVal
#================================================================================================================================================
def cost(springK, boundaryInds=boundaryIndices, bulkInds=bulkIndices):
   # springK, bulkPoints = np.diag(variables[:numberOfEdges]), variables[numberOfEdges:]
  # U[bulkInds] = bulkPoints
  DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=springK,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges)
  
  res = op.minimize(energy, U, method='Newton-CG', args=(DynMat,), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
  uEnergy = res.fun
  minEnergy = lowestEigenVal(DynMat)
  return uEnergy/minEnergy
#================================================================================================================================================


#res = op.minimize(cost, k0, method='BFGS', options={'disp': True})
#runResults = RunData("20-by-20-square", 20, vertices, edges, U[boundaryIndices], res)











