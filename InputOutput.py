# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 06:39:30 2017

@author: Salem

In this script we will design a lattice that has the desired input output relations. 
"""

import numpy as np
import numpy.random as npr
import importlib
import pickle
import LatticeMaking
import EnergyFunctions
from numpy import linalg as la
import scipy.optimize as op
importlib.reload(LatticeMaking)
importlib.reload(EnergyFunctions)

from LatticeMaking import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
width = 25;
height = 4;
#A square lattice which is randomized 
(vertices, edges) = squareLattice(width, height=height, randomize=True)

numberOfVerts = len(set(list(edges.flatten())))
numberOfEdges = edges.size//2

edgeMat1 = makeEdgeMatrix1(edges)
edgeMat2 = makeEdgeMatrix2(edges)

#rigidityMatrix = makeRigidityMat(vertices, edges) #defined in LatticeMaking

(boundaryIndices, bulkIndices) = getBoundaryVerts(edges) # I need to know the bulk indices

#(inputIndices, outputIndices) = getIONodes(vertices, height)

#a simpler way to get the input output indices
inputIndices = np.array([0, 1, int(height -1) , int(height -2)])
outputIndices = np.array([-int(height/2), -int(height/2) + 1 ])
flattenedInInds , flattenedOutInds = flattenedIndices(inputIndices,numberOfVerts), flattenedIndices(outputIndices,numberOfVerts)
#these will be alternating between 1 and zero, which might be a clean way to select the two possibles inputs. 
#input1 = np.array([0.5*(1.0 - (-1.0)**i)  for i in range(height)])          
#input2 = np.array([0.5*(1.0 + (-1.0)**i)  for i in range(height)])  

# generate a random vector of displacements which contains inputs and bulk, also notice that this will need to be flattened when minimizing
Un = npr.rand(2*numberOfVerts) #defined in LatticeMaking, returns a normalized vector
#U[:3] = np.array([0, 0, 0])  
uN = Un.reshape((numberOfVerts, 2))
#inputDisp1 = (input1[:, np.newaxis]*uN[inputIndices]).flatten()
#inputDisp2 = (input2[:, np.newaxis]*uN[inputIndices]).flatten()
inputDisp1 = 0.3*np.array([-1, -1, 1, 1, 1, 1, -1, -1])
inputDisp2 = 0.3*np.array([-1, -1, 1, 1, -1, -1, 1, 1])
inputDisps = (inputDisp1, inputDisp2)

#outputDisp1 = npr.rand(uN[outputIndices].size)
#outputDisp2 = uN[outputIndices].flatten()
#outDisps = (outputDisp1, outputDisp2)
# a reference set of spring constants
k0 = np.ones(numberOfEdges)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy using the edge matrix.
#================================================================================================================================================
def energy(u, DynMat, inputCase, inputInds = flattenedInInds):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    
    energy(u, DynMat, inputInds = inputIndices, inputCase = 0): calculates the energy after setting the input indices to the correct values. 
    
    inputCase is an integer that runs over the possible inputs, for now it is just 0 or 1
    """
    u[inputInds] = inputDisps[inputCase]
    return 0.5*np.dot(np.dot(u.transpose(), DynMat), u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy gradient from the dynamical matrix.
#================================================================================================================================================
def energy_Der(u, DynMat, inputCase, inputInds = flattenedInInds):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    
    inputCase is an integer that runs over the possible inputs, for now it is just 0 or 1
    """
    u[inputInds] = inputDisps[inputCase]
    return np.dot(DynMat, u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy Hessian from the dynamical matrix.
#================================================================================================================================================
def energy_Hess(u, DynMat, inputInds = flattenedInInds):
    return DynMat
#================================================================================================================================================  

#================================================================================================================================================
# This the cost function that will be minimized E/lowestEigVal
#================================================================================================================================================
def cost(points, springK=k0, flatOutInds=flattenedOutInds,eMat1=edgeMat1, eMat2=edgeMat2):
   # springK, bulkPoints = np.diag(variables[:numberOfEdges]), variables[numberOfEdges:]
  # U[bulkInds] = bulkPoints
  rigidityMatrix = makeRigidityMat(points, edgeMat1=eMat1, edgeMat2=eMat2)
  DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=springK,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges)
  
  res0 = op.minimize(energy, Un, method='Newton-CG', args=(DynMat, 0), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False}) 
  res1 = op.minimize(energy, Un, method='Newton-CG', args=(DynMat, 1), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
  
  return np.sum((res0.x[flattenedOutInds])**2.0 - (res1.x[flattenedOutInds])**2.0)

#================================================================================================================================================

def test(points, springK=k0, eMat1=edgeMat1, eMat2=edgeMat2):
    rigidityMatrix = makeRigidityMat(points, edgeMat1=eMat1, edgeMat2=eMat2)
    dynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=springK,  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges)
    res0 = op.minimize(energy, Un, method='Newton-CG', args=(dynMat, 0), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False}) 
    res1 = op.minimize(energy, Un, method='Newton-CG', args=(dynMat, 1), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
    return (res0.x[flattenedOutInds] , res1.x[flattenedOutInds])



res = op.minimize(cost, vertices, method='BFGS', options={'disp': True})
test(res.x)


