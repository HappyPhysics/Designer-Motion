# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 06:14:39 2017

@author: Salem

starting with a single square I want to see if I can add points to it that will make it's 4 nodes move in a desired way. 
The spring constants can be negative
"""

import numpy as np
import numpy.random as npr
import importlib
import LatticeMaking
from numpy import linalg as la
import scipy.optimize as op
importlib.reload(LatticeMaking)

from LatticeMaking import *  #custom


vertices = np.array([[1.0, 2.0], [2, 1.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
edgeArray = np.array([[0, 1], [1, 2], [2,3], [3, 0],[4,0], [4,1], [5,1], [5,2],[6,2], [6,3], [7,0], [7,3], [6,4],[7, 5], [0,8],[1, 8],[2, 8],[3, 8]])

numberOfVerts = len(set(list(edgeArray.flatten())))
numberOfEdges = edgeArray.size//2

edgeMat1 = makeEdgeMatrix1(edgeArray)
edgeMat2 = makeEdgeMatrix2(edgeArray)

boundaryIndices = flattenedIndices(np.array([0, 1, 2, 3]), numberOfVerts)
bulkIndices = flattenedIndices(np.array([4, 5, 6, 7, 8]), numberOfVerts)

k0 = np.ones(numberOfEdges)
k0[-4:] = 0.5*np.array([-1,-1,-1,-1])
k0[4:-4] = 500*k0[4:-4]
k0[:4] *= 2
print(k0)
k1 = np.ones(numberOfEdges)

rigidityMatrix = makeRigidityMat(vertices.flatten(), edgeArray=edgeArray)[:,:-3] #defined in LatticeMaking

DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=k0,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges, negativeK=True)

print(la.eigvalsh(DynMat))

#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy using the edge matrix.
#================================================================================================================================================
def energy(u, DynMat, bInds = boundaryIndices):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    
    energy(u, DynMat, BInds = boundaryIndices): calculates the energy after setting the boundary indices to the correct values. 
    """
    return 0.5*np.dot(np.dot(u.transpose(), DynMat), u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy gradient from the dynamical matrix.
#================================================================================================================================================
def energy_Der(u, DynMat, bInds = boundaryIndices):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    """
    return np.dot(DynMat, u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy Hessian from the dynamical matrix.
#================================================================================================================================================
def energy_Hess(u, DynMat, bInds = boundaryIndices):
    return DynMat
#================================================================================================================================================  

#================================================================================================================================================
# Returns the lowest eignevalue of the dynamical matrix, exluding the rigid motions of course.
#================================================================================================================================================
def lowestEigenVal(DynMat):    
    return (la.eigvalsh(0.5*DynMat)[0])
#================================================================================================================================================
    
#================================================================================================================================================
# This the cost function that will be minimized E/lowestEigVal
#================================================================================================================================================
def cost(Vars, bInds = boundaryIndices, bkInds = bulkIndices, eMat1=edgeMat1, eMat2=edgeMat2):
  points = Vars[numberOfEdges:]
  springK = Vars[:numberOfEdges]
  points[bInds] = vertices.flatten()[bInds]
  rigidityMatrix = makeRigidityMat(points, edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
  DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=springK,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges, negativeK=True)
  
  res0 = op.minimize(energy, U, method='Newton-CG', args=(DynMat, ), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False}) 
  minE = lowestEigenVal(DynMat)
  
  return (res0.fun/minE)
#================================================================================================================================================

#================================================================================================================================================
# This the cost function that will be minimized E/lowestEigVal
#================================================================================================================================================
def test(newVars, bInds = boundaryIndices, bkInds = bulkIndices, eMat1=edgeMat1, eMat2=edgeMat2):
  points = newVars[numberOfEdges:]
  springK = newVars[:numberOfEdges]
  points[bInds] = vertices.flatten()[bInds]
  rigidityMatrix = makeRigidityMat(points, edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
  DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=springK,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges, negativeK=True)
  
  res0 = op.minimize(energy, U, method='Newton-CG', args=(DynMat, ), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False}) 
  minE = lowestEigenVal(DynMat)
  print("eigenvalues: ", (la.eigvalsh(0.5*DynMat)[:]))
  return res0.fun , minE

#================================================================================================================================================  






























