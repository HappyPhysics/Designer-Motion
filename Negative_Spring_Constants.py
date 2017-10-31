# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 06:41:12 2017

@author: Salem

In this script we allow the spring constants to be negative and see if that helps in designing the network
"""
import numpy as np
import numpy.random as npr
import importlib
import pickle
import LatticeMaking
from numpy import linalg as la
import scipy.optimize as op
importlib.reload(LatticeMaking)

from LatticeMaking import *  #custom

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#It's convineit to make these variables global, they also don't change after initializing
global U, rigidityMatrix, numberOfEdges, numberOfVerts

width = 10;
height = 10;

#A square lattice which is randomized 
(vertices, edges) = squareLattice(width, height, randomize=True)

#For computing the rigidity Matrix
edgeMat1 = makeEdgeMatrix1(edges)
edgeMat2 = makeEdgeMatrix2(edges)

numberOfVerts = len(set(list(edges.flatten())))
numberOfEdges = edges.size//2

#rigidityMatrix = makeRigidityMat(vertices, edges) #defined in LatticeMaking

#generate the desired displacement
U = npr.rand(2*numberOfVerts) #defined in LatticeMaking, returns a normalized vector
U[:3] = np.array([0, 0, 0])  #excludes rotations and translations from the deformation, but not from DynMat eigenmodes
normalizeVec(U)
k0 = 0.1*np.ones(numberOfEdges)

Vars0 = np.hstack((vertices.flatten(), k0))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy from the dynamical matrix.
#================================================================================================================================================
def energy(u, DynMat):
    """
    Calculates the energy corresponding to u from knowing the Dynamical matrix
    """
    return 0.5* np.dot(np.dot(u.transpose(), DynMat), u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy gradient from the dynamical matrix.
#================================================================================================================================================
def energy_Der(u, DynMat):
    return np.dot(DynMat, u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy Hessian from the dynamical matrix.
#================================================================================================================================================
def energy_Hess(u, DynMat):
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
def cost(Vars):
    """
    The cost depends both on the points P and the spring constants k. 
    
    Vars[:2*NumberOfPoints] represents the points.
    Vars[2*NumberOfPoints:] represents the spring constants
    """
    rigidityMat = makeRigidityMat(verts=Vars[:2*numberOfVerts], edgeMat1=edgeMat1, edgeMat2=edgeMat2) 
    DynMat = makeDynamicalMat(RigidityMat=rigidityMat, springK=Vars[2*numberOfVerts:],  numOfVerts=numberOfVerts, numOfEdges=numberOfEdges, negativeK=True)
  
  #res = op.minimize(energy, U, method='Newton-CG', args=(DynMat,), jac=energy_Der, hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
  #uEnergy = res.fun
    minEnergy = lowestEigenVal(DynMat)
    return 15*energy(U, DynMat)/minEnergy- minEnergy*minEnergy
#================================================================================================================================================

def test(newVars):
    rigidityMat = makeRigidityMat(verts=newVars[:2*numberOfVerts], edgeMat1=edgeMat1, edgeMat2=edgeMat2) 
    DynMat = makeDynamicalMat(RigidityMat=rigidityMat, springK=newVars[2*numberOfVerts:], 
                              numOfVerts=numberOfVerts, numOfEdges=numberOfEdges, negativeK=True)
    print("lowest eigenvalues: ", la.eigvalsh(DynMat)[:5])
    print("energy of mode: ", energy(U, DynMat))
    return  
    

res = op.minimize(cost, Vars0, method='BFGS', options={'disp': True})
test(res.x)




