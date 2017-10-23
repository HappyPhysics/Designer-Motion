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

NUM_OF_ADDED_VERTS = 5;
NUM_OF_DIMENSIONS = 2;
#NUM_OF_EDGES = (NUM_OF_ADDED_VERTS + 4) * (NUM_OF_ADDED_VERTS + 3)//2


# this is the part we want to control the motion of, these vertices will be fixed.
little_square = np.array([[0.0, 0.0], [0, 1.0], [1.0, 1.0], [1.0, 0.0]])

#This enum represents the different types of deformations that you can have 
#TODO this def might fit in lattice making
class DispType(Enum):
    random = 1
    isotropic = 2

#this enumumerates the possible ways to connect the added vertices to each other and the square
class EdgeTypes(Enum):
    all_connected = 1
    all_to_square = 2
    square_lattice = 3
#================================================================================================================================================
# Runs the minimization procedure to return the results for the spring constants and the positions
#================================================================================================================================================
def find_desired_lattice(deformationType = DispType.random, edgeType = EdgeTypes.all_connected, num_of_added_verts = NUM_OF_ADDED_VERTS):
    """
    """
    #initialize test results so that the while loop goes at least once
    test_result = True
    
    #how many times the minimization procedure ran
    trial_num = 0
    
    #initialize the lattice
    vertices, edge_array = initialize_lattice(edgeType, num_of_added_verts)
    
    num_of_verts = vertices.size//2
    num_of_edges = edge_array.size//2
    
    #generate displacement field for the square. outside loop because we don't want to keep changing this
    U = displacement_field(vertices, num_of_vertices=num_of_verts, DeformType=deformationType)
    
    while (test_result):
    
    
        # connectivity dependent matrices that are used to calculate the rigidity matrix
        edgeMat1 = makeEdgeMatrix1(edge_array, numOfEdges=num_of_edges, numOfVerts=num_of_verts)
        edgeMat2 = makeEdgeMatrix2(edge_array, numOfEdges=num_of_edges, numOfVerts=num_of_verts)
    

        #initalize var: points and spring constants
        k0  = npr.rand(num_of_edges)
        k1 = np.ones(num_of_edges)
        var0 = np.hstack((vertices.flatten(), k0))
        var1 = np.hstack((vertices.flatten(), k1))
    
        #minimize cost funcion
        res = op.minimize(cost_function, var0, method='BFGS',args=(U, edgeMat1, edgeMat2, num_of_edges, num_of_verts), options={'disp': False})
        
        trial_num += 1; print("Trial Number: ", trial_num)
        
        #if this returns true then keep trying, checks if U is close to the minimum on the little_square 
        test_result = test_results(res.x, U, edgeMat1, edgeMat2, num_of_edges, num_of_verts)
        
        #initialize the lattice
        vertices, edge_array = initialize_lattice(edgeType, num_of_added_verts)
        
    return res
    

#================================================================================================================================================
# The cost function penalizes energy of the desired displacement of the square vertices
#================================================================================================================================================
def cost_function(var, disp_field, eMat1, eMat2, num_of_edges,num_of_vertices):
    """
    var is the combined variables to be minimized over. It represents all the vertices and spring constants
    var[:2*num_of_vertices] are the points 
    var[2*num_of_vertices:] are the spring constants
    """
    
    #the square positions are fixed
    var[:8] = little_square.flatten()
    
    
   # var[:2*num_of_vertices] are the points of the lattice
   # var[2*num_of_vertices:] are the spring constants
   
    rigidityMatrix = makeRigidityMat(var[:2*num_of_vertices], edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
    
    #calculate the dynamical matrix
    DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=var[2*num_of_vertices:], numOfVerts=num_of_vertices, numOfEdges=num_of_edges)
    
    
    # minimize the energy subject to the constraint that the square displacements are fixed
    res0 = op.minimize(energy, disp_field, method='Newton-CG', args=(DynMat, disp_field[:5]), jac=energy_Der, 
                       hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
    
    #get the lowest eigen vector
    #lowestEigVector = normalizeVec(la.eigh(DynMat)[1][:5,0])
    
    lowestEs = lowestEigenVals(DynMat)
    # minimize this energy with respect to the lowest energy eigenvalue
    return res0.fun/lowestEs[0] + 0.1 * (lowestEs[0]/lowestEs[1])**2
#================================================================================================================================================    

#================================================================================================================================================
# Initializing the lattice
#================================================================================================================================================
def initialize_lattice(edgeType = EdgeTypes.all_connected, num_of_added_verts = NUM_OF_ADDED_VERTS):
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
    gray_matter = npr.rand(num_of_added_verts, NUM_OF_DIMENSIONS)*2.0 - 0.5

    # add them together to get the entire list of vertices
    vertices = np.vstack((little_square, gray_matter))
    
    if(edgeType == EdgeTypes.all_connected):
    # make the edge array, connect all points for now
        edge_array = connect_all_verts(get_num_of_verts(vertices))
        
    elif(edgeType == EdgeTypes.all_to_square):
        #connect each gray matter vertex to the square vertices
        edge_array = connect_all_to_square(num_of_added_verts)
    elif(edgeType == EdgeTypes.square_lattice):
        #add an internal square lattice to the initial square
        width=int(num_of_added_verts)   
        verts, edges =  squareLattice(width=width, randomize=True, ax=1/width)
        
        #connect to the little_square
        edge_array = np.vstack(((edges + 4), [0,0], [width - 1, 1], [width**2 - width - 1, 3], [width**2 - 1, 2]))
        
        #add the two square to vertices array
        vertices = np.vstack((little_square, verts))
        
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
def lowestEigenVals(DynMat, num_of_eigs = 2):    
    return (la.eigvalsh(0.5*DynMat)[:num_of_eigs])
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
def test_results(new_var, disp_field, eMat1, eMat2, num_of_edges, num_of_vertices):
    """
    this returns True if the dot product between the desired diplacement and the lowest eigen vector after minimization satisfies dotproduct < 0.95.
    this will result in trying the minimization procedure again.
    
    var is the combined variables to be minimized over. It represents all the vertices and spring constants
    var[:2*num_of_vertices] are the points 
    var[2*num_of_vertices:] are the spring constants
    """
    
    #the square positions are fixed
    new_var[:8] = little_square.flatten()
    
   # var[:num_of_vertices] are the points of the lattice
   # var[num_of_vertices:] are the spring constants
   
    rigidityMatrix = makeRigidityMat(new_var[:2*num_of_vertices], edgeMat1=eMat1, edgeMat2=eMat2)[:, 3:]
    
    #calculate the dynamical matrix
    DynMat = makeDynamicalMat(RigidityMat= rigidityMatrix,
                              springK=new_var[2*num_of_vertices:], numOfVerts=num_of_vertices, numOfEdges=num_of_edges)
    
    
    # minimize the energy subject to the constraint that the square displacements are fixed
    res0 = op.minimize(energy, disp_field, method='Newton-CG', args=(DynMat, disp_field[:5]), jac=energy_Der, 
                       hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
    
    lowestEigVector = normalizeVec(la.eigh(DynMat)[1][:5,0])
    secondEigVector = normalizeVec(la.eigh(DynMat)[1][:5,1])
    
    #return false if the vectors are not close enough
    dotProduct = np.dot(lowestEigVector, normalizeVec(res0.x[:5]))
    dotProduct *= np.sign(dotProduct)
    lowestEigVector *= np.sign(dotProduct)
    
    gap = (lowestEigenVals(DynMat, 2)[1] - lowestEigenVals(DynMat, 2)[0])/lowestEigenVals(DynMat, 2)[1]
    if((dotProduct < 0.995) or gap < 0.5):
        print("dot produce: ", dotProduct, "\n")
        print("square disps in lowest energy: ", normalizeVec(lowestEigVector[:5]), "\n")
        print("square disps in desired motion: ", normalizeVec(res0.x[:5]), "\n")
        print("eigenvalues: ", lowestEigenVals(DynMat, 5), "\n")
        print("trying again ... \n\n")
        return True;
    
    print("Number of edges: ", rigidityMatrix.shape[0], "\n")
    print("energy: ", energy(normalizeVec(res0.x), DynMat, disp_field[:5]), "\n")
    print("eigenvalues: ", lowestEigenVals(DynMat, 5), "\n")
    print("dot produce: ", dotProduct, "\n")
    print("square disps in lowest energy: ", lowestEigVector, "\n")
    print("square disps in desired motion: ", normalizeVec(res0.x[:5]), "\n")
    print("square disps in next to lowest: ", normalizeVec(secondEigVector[:5]), "\n")
    
    
    plotPoints(new_var[:2*num_of_vertices], num_of_vertices)

    return False
#================================================================================================================================================ 


#================================================================================================================================================
# plots the points as a scatter plot
#================================================================================================================================================
def plotPoints(flattenedPoints, num_of_verts = -1):
    """
    Takes in a list of point positions which is then reshaped into a list 2-vectors.
    A different color and size is chosen for the original square vertices.
    """
    if (num_of_verts < 0):
       num_of_verts = flattenedPoints.size//2 
    
    #reshape the points to look like a list of vectors
    Points = flattenedPoints.reshape(num_of_verts, 2)
    
    #chose the area of the square vertices to be bigger
    area = 200*np.ones(num_of_verts)
    area[4:] *= 0.4 
    #also a different color for the square vertices
    color = np.copy(area)
    
    
    plt.scatter(Points[:,0], Points[:,1], s=area, c=color)
#================================================================================================================================================



























