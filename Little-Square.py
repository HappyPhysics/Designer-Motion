# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 06:14:39 2017

@author: Salem

starting with a single square I want to see if I can add points (which I will call grey matter) to it that will make it's 4 nodes move in a desired way. 

All the points will be connected, when the cost is minimized some of the spring constants will be allowed to go to zero, (2 N - 4) of them to be specific.

Elastic energy is minimized first, then the cost function brings this energy to zero for the desired motion.

normalizeVec, connect_all_verts, makeRigidityMat are defined in LatticeMaking

Methods: initialize_lattice(num_of_added_points)
         rand_displacement_field(num_of_vertices)
         isotropic_contraction(vertices)
"""

import numpy as np
import numpy.random as npr
import importlib
import LatticeMaking
from numpy import linalg as la
from matplotlib import pyplot as plt
import scipy.optimize as op
importlib.reload(LatticeMaking)

from LatticeMaking import *  #custom
from enum import Enum

NUM_OF_ADDED_VERTS = 11;
NUM_OF_DIMENSIONS = 2;
NUM_OF_EDGES = (NUM_OF_ADDED_VERTS + 4) * (NUM_OF_ADDED_VERTS + 3)//2


# this is the part we want to control the motion of, these vertices will be fixed.
little_square = np.array([[0.0, 0.0], [0, 1.0], [1.0, 1.0], [1.0, 0.0]])

#This enum represents the different types of deformations that you can have 
#TODO this def might fit in lattice making
class DispType(Enum):
    random = 1
    isotropic = 2


#================================================================================================================================================
# Runs the minimization procedure to return the results for the spring constants and the positions
#================================================================================================================================================
def find_desired_lattice(deformationType = DispType.random):
    """
    """
    #initialize the lattice
    vertices, edge_array = initialize_lattice()
    
    num_of_verts = NUM_OF_ADDED_VERTS + 4
    num_of_edges = NUM_OF_EDGES
    
    # connectivity dependent matrices that are used to calculate the rigidity matrix
    edgeMat1 = makeEdgeMatrix1(edge_array, numOfEdges=num_of_edges, numOfVerts=num_of_verts)
    edgeMat2 = makeEdgeMatrix2(edge_array, numOfEdges=num_of_edges, numOfVerts=num_of_verts)
    
    #generate displacement field for the square
    U = displacement_field(vertices, num_of_vertices=num_of_verts, DeformType=deformationType)

   #initalize var: points and spring constants
    k0  = np.ones(num_of_edges)
    var0 = np.hstack((vertices.flatten(), k0))
    
    #minimize cost funcion
    res = op.minimize(cost_function, var0, method='BFGS',args=(U, edgeMat1, edgeMat2), options={'disp': True})
    
    test_results(res.x, U, edgeMat1, edgeMat2)
    
    return res
    

#================================================================================================================================================
# The cost function penalizes energy of the desired displacement of the square vertices
#================================================================================================================================================
def cost_function(var, disp_field, eMat1, eMat2):
    """
    var is the combined variables to be minimized over. It represents all the vertices and spring constants
    var[:2*num_of_vertices] are the points 
    var[2*num_of_vertices:] are the spring constants
    """
    
    #the square positions are fixed
    var[:8] = little_square.flatten()
    
    num_of_vertices = NUM_OF_ADDED_VERTS + 4
    num_of_edges = NUM_OF_EDGES
    
   # var[:num_of_vertices] are the points of the lattice
   # var[num_of_vertices:] are the spring constants
   
    rigidityMatrix = makeRigidityMat(var[:2*num_of_vertices], edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
    
    #calculate the dynamical matrix
    DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=var[2*num_of_vertices:], numOfVerts=num_of_vertices, numOfEdges=num_of_edges)
    
    
    # minimize the energy subject to the constraint that the square displacements are fixed
    res0 = op.minimize(energy, disp_field, method='Newton-CG', args=(DynMat, disp_field[:5]), jac=energy_Der, 
                       hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
   
    # minimize this energy with respect to the lowest energy eigenvalue
    return res0.fun/lowestEigenVal(DynMat) 
#================================================================================================================================================    

#================================================================================================================================================
# Initializing the lattice
#================================================================================================================================================
def initialize_lattice():
    """
    This method returns an array of position vectors (vertices) and an array of edge vectors (edge_array).
    
    The vertices include a square with of unit width and (num_of_added_points) extra points that are inserted at random positions in a square 
    of width = 2. The square vertices must be the first 0,1,2,3.
    
    Every point is connected to every other point so it generates the maximum number of edges. 
    
    Example: initialize_lattice(2)
    Out[45]: 
(array([[ 0.        ,  0.        ],
        [ 0.        ,  1.        ],
        [ 1.        ,  1.        ],
        [ 1.        ,  0.        ],
        [ 0.49850383,  0.26623088]]), array([[0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4]]))
    """

    # this part I call grey matter, these are the added to the square vertices 
    grey_matter = npr.rand(NUM_OF_ADDED_VERTS, NUM_OF_DIMENSIONS)*2.0 - 0.5

    # add them together to get the entire list of vertices
    vertices = np.vstack((little_square, grey_matter))
    
    # make the edge array, connect all points for now
    edge_array = connect_all_verts(get_num_of_verts(vertices))
        
    return vertices, edge_array
#================================================================================================================================================

#================================================================================================================================================
# generate the displacement field wanted
#================================================================================================================================================    
def displacement_field(vertices, DeformType = DispType.random, num_of_vertices = -1):
    """
   DispType.random:     Makes a random displacement field. The first 3 degrees of freedom are assumed to 
   be zero in order to fix rotation and translation of the lattice.
   DispType.isotropic: Every point moves towards the origin with an amount propotional to the distance from the origin
    """
    if(DeformType == DispType.random):
        
        if(num_of_vertices < 0):
            get_num_of_verts(vertices)
            
        return normalizeVec(npr.rand(2*num_of_vertices - 3)) 
    
    elif(DeformType == DispType.isotropic):
        return normalizeVec(vertices.flatten()[3:])
#================================================================================================================================================    

#================================================================================================================================================
# makes a random displacement field 
#================================================================================================================================================  
def isotropic_contraction(vertices):
    """
    
    """
    return normalizeVec(vertices.flatten()[3:])
    
#================================================================================================================================================  
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy using the edge matrix.
#================================================================================================================================================
def energy(u, DynMat, squareDisp):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    The vertices of the squares have fixed displacements, the rest will be allowed to relax to minimum energy
    TODO: A more general energy function that takes in the boundary conditions directly
    
    energy(u, DynMat, BInds = boundaryIndices): calculates the energy after setting the boundary indices to the correct values. 
    """
    u[:5] = squareDisp #this assumes that the square vertex indices are 0,1,2,3
    u = normalizeVec(u)
    return 0.5*np.dot(np.dot(u.transpose(), DynMat), u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy gradient from the dynamical matrix.
#================================================================================================================================================
def energy_Der(u, DynMat, squareDisp):
    """
    Be careful about using this in different scripts, because this assumes boundary conditions when computing the energy.
    TO DO: A more general energy function that takes in the boundary conditions directly
    """
    u[:5] = squareDisp
    u = normalizeVec(u)
    return np.dot(DynMat, u)
#================================================================================================================================================
    
#================================================================================================================================================
# After setting the boundary indices to the desired values, calculates the energy Hessian from the dynamical matrix.
#================================================================================================================================================
def energy_Hess(u, DynMat,  squareDisp):
    return DynMat
#================================================================================================================================================  

#================================================================================================================================================
# Returns the lowest eignevalue of the dynamical matrix, exluding the rigid motions of course.
#================================================================================================================================================
def lowestEigenVals(DynMat):    
    return (la.eigvalsh(0.5*DynMat)[:6])
#================================================================================================================================================
    
#================================================================================================================================================
# Returns the lowest eignevalue of the dynamical matrix, exluding the rigid motions of course.
#================================================================================================================================================
def lowestEigenVal(DynMat):    
    return (la.eigvalsh(0.5*DynMat)[0])
#================================================================================================================================================
  

#================================================================================================================================================
# Test the results of the minimization procedure
#================================================================================================================================================
def test_results(new_var, disp_field, eMat1, eMat2):
    """
    var is the combined variables to be minimized over. It represents all the vertices and spring constants
    var[:2*num_of_vertices] are the points 
    var[2*num_of_vertices:] are the spring constants
    """
    
    #the square positions are fixed
    new_var[:8] = little_square.flatten()
    
    num_of_vertices = NUM_OF_ADDED_VERTS + 4
    num_of_edges = NUM_OF_EDGES
    
   # var[:num_of_vertices] are the points of the lattice
   # var[num_of_vertices:] are the spring constants
   
    rigidityMatrix = makeRigidityMat(new_var[:2*num_of_vertices], edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
    
    #calculate the dynamical matrix
    DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=new_var[2*num_of_vertices:], numOfVerts=num_of_vertices, numOfEdges=num_of_edges)
    
    
    # minimize the energy subject to the constraint that the square displacements are fixed
    res0 = op.minimize(energy, disp_field, method='Newton-CG', args=(DynMat, disp_field[:5]), jac=energy_Der, 
                       hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
    print("Number of edges: ", NUM_OF_EDGES)
    print("energy: ", energy(normalizeVec(res0.x), DynMat, disp_field[:5]))
    print("eigenvalues: ", lowestEigenVals(DynMat))
    lowestEigVector = normalizeVec(la.eigh(DynMat)[1][:,0])
    secondEigVector = normalizeVec(la.eigh(DynMat)[1][:,1])
    print("dot produce: ", np.dot(lowestEigVector, normalizeVec(res0.x)))
    print("square disps in lowest: ", normalizeVec(lowestEigVector[:5]))
    print("square disps in solution: ", normalizeVec(res0.x[:5]))
    print("square disps in next to lowest: ", normalizeVec(secondEigVector[:5]))
    
    
    plotPoints(new_var[:2*num_of_vertices], num_of_vertices)

    return 
#================================================================================================================================================ 


#================================================================================================================================================
# plots the points as a scatter plot
#================================================================================================================================================
def plotPoints(flattenedPoints, num_of_verts):
    """
    Takes in a list of point positions which is then reshaped into a list 2-vectors.
    A different color and size is chosen for the original square vertices.
    """
    #reshape the points to look like a list of vectors
    Points = flattenedPoints.reshape(num_of_verts, 2)
    
    #chose the area of the square vertices to be bigger
    area = 200*np.ones(num_of_verts)
    area[4:] *= 0.4 
    #also a different color for the square vertices
    color = np.copy(area)
    
    
    plt.scatter(Points[:,0], Points[:,1], s=area, c=color)
#================================================================================================================================================



























