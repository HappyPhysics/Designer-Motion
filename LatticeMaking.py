# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 06:27:18 2017

@author: Salem

This script generates the required lattices which will be used by other lattices.

Variables: 
    edgeArray: Appears in the methods, it's (numOfEdges, 2) array containing the indices of the vertices. 
    
Methods:
    squareLattice:
        makes a square lattice with corresponding edges. Diagonal edges are included, making it a triangulation
    
    getBoundaryVerts: 
        From a given edgeArray finds the indices of the boundary and bulk vertices. It does that based on the number
    of neighbors each vertex has. It will be 6 in the bulk
    
    makeAdjacencyMatrix:
        Makes the adjacency matrix based from a given edgeArray. 
        
   makeRigidityMatrix:
       This matrix is useful for vectorizing the expression for the dynamical matrix and cost function (see below).
    
    
    
"""

import numpy as np
import itertools as it
import scipy.sparse as sps

#===========================================================================================================================================
# returns a square lattice with the corresponding edges
#===========================================================================================================================================
def squareLattice(width, height=None, ax=1.0, ay=1.0, randomize=False, randomBox=0):

    """
    squareLattice(width, height=None, ax=1.0, ay=1.0, randomize=False, randomBox=0)
    returns a square lattice and edges, including diagonals.
    
    width: number of points in the x direction
    height: number of points in the y direction
    ax,ay: spacing between points in the x,y directions
    randomize: Add randomness to the positions of the points, its a uniform distribution within the box...
    randomBox: fraction of the unit cell that the point might be in. It will be 1 if nothing is entered and randomize is true
    
    
    Example: squareLattice(2) = (array([[ 0.,  0.],
        [ 0.,  1.],
        [ 1.,  0.],
        [ 1.,  1.]]), array([[0, 1],
        [2, 3],
        [0, 2],
        [1, 3],
        [0, 3]]))
    """
    if height is None:
        height = width
        
    if randomize and randomBox == 0.0:
        randomBox = 1.0
    elif not randomize:
        randomBox = 0
        
    vertices = np.array([[i * ax + randomize* np.random.uniform(-randomBox*ax/2.0, randomBox* ax/2.0), j * ay + 
                      randomize* np.random.uniform(-randomBox*ay/2.,randomBox* ay/2.0)]
                      for i in range(width) for j in range(height)])
    
    edges = [[n + m*height, m*height + n + 1] for m in range(width) for n in range(height - 1)]
    edges.extend([[n*height + m, m + n*height + height] for m in range(height) for n in range(width - 1)])
    edges.extend([[n*height + m, m + n*height + height + 1] for m in range(height - 1) for n in range(width - 1)])
    return (vertices, np.array(edges))
#===========================================================================================================================================
 
#===========================================================================================================================================
# returns the indices of the boundary and bulk indices
#===========================================================================================================================================    
def getBoundaryVerts(edgeArray):
    """
    getBoundaryVerts(edges)
    returns the indices of the boundary edges and the bulk edges based on the number of neighbors they have
    in the edge list.
    
    Example: edges = squareLattice(3)[1];
             getBoundaryVerts(edges) 
    Out: (array([0, 1, 2, 3, 5, 6, 7, 8]), array([4]))
    """
    edgeList = list(edgeArray.flatten())
    
    numOfVerts = len(set(edgeList))
    
    boundaryVert = [edgeList.count(i) < 6 for i in range(numOfVerts)]
    bulkVert = [not bVert for bVert in boundaryVert]
    return (np.arange(numOfVerts)[boundaryVert],  np.arange(numOfVerts)[bulkVert])
#===========================================================================================================================================

#===========================================================================================================================================
# returns the adjacency matrix as an array
#===========================================================================================================================================  
def makeAdjacencyMatrix(edgeArray, numOfVerts=-1):
    """
    makeAdjacencyMatrix(edgeList):
        Takes in the edgeArray then converts it to a list, which has elements of the form [vert1, vert2] and finds the (2 numOfVerts x 2 numOfVerts) 
        adjacency matrix.
    """
    edgeList = edgeArray.tolist()
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))

    adjacencyMat = np.zeros((numOfVerts, numOfVerts));      
    for i in range(numOfVerts):
        for j in range(numOfVerts):
            adjacencyMat[i, j] = 1 if [i, j] in edgeList or [j, i] in edgeList else 0
    return adjacencyMat
#===========================================================================================================================================


#===========================================================================================================================================
# returns the Rigidity Matrix as an array
#===========================================================================================================================================  
def makeRigidityMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1):
    """
    makeRigidityMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1):
        Takes in the edgeArray then finds Rigidity matrix. The rigidity matrix helps
        to find the bond stretching to linear order in displacement u which has 
        size = 2 numOfVerts. Bond stretchings are equal to 
        dl_e = R_ei * u_i, where i is summed over
        
    Example: 
            (verts, edges) = squareLattice(2, randomize=False); 
            R = makeRigidityMat(verts, edges) 
            R 
        Out: array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])
    """
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
      
    RigidityMat = np.zeros((numOfEdges, 2 * numOfVerts))
    
    for edgeNum, edge in enumerate(edgeArray):
        t = np.zeros((numOfVerts, 2))
        t[edge[1]] = verts[edge[1]] - verts[edge[0]]
        t[edge[0]] = verts[edge[0]] - verts[edge[1]]
        RigidityMat[edgeNum] = t.flatten()
    
    return RigidityMat
#===========================================================================================================================================
   

    
#===========================================================================================================================================
# returns the Rigidity Matrix as an array
#===========================================================================================================================================  
def makeDynamicalMat(edgeArray = np.zeros(1), verts = np.zeros(1), RigidityMat= np.zeros(1), springK= np.zeros(1),  numOfVerts=-1, numOfEdges=-1,):
    """
    makeDynamicalMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1):
        Takes in the edgeArray then finds dynamical matrix. The dynamical matrix
        help in calculating the potential energy of a displacement u which has 
        size = 2 numOfVerts. The energy is given by E[u] = u.T D u.
        
    Example: 
            (verts, edges) = squareLattice(2, randomize=False); 
             makeDynamicalMat(edgeArray=edges, RigidityMat=R)
        Out: array([[ 2.,  1.,  0.,  0., -1.,  0., -1., -1.],
       [ 1.,  2.,  0., -1.,  0.,  0., -1., -1.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [-1., -1., -1.,  0.,  0.,  0.,  2.,  1.],
       [-1., -1.,  0.,  0.,  0., -1.,  1.,  2.]])
    """
    if numOfEdges < 0:
            if(not edgeArray.any()):
                raise NameError("Please either provide the the number of edges or the edge array")
            numOfEdges = edgeArray.size//2
            
    if(not RigidityMat.any()):
        if not verts.any():
            raise NameError("Please either provide the rigidity matrix or the vertices for calculating the rigidity matrix")
        if numOfVerts < 1:
            numOfVerts = len(set(list(edgeArray.flatten())))
        RigidityMat = makeRigidityMat(verts, edgeArray, numOfVerts, numOfEdges) 
    
    if(not springK.any()):
        springK = np.diag(np.ones(numOfEdges))
    
    
    dynMat = np.dot(np.dot(RigidityMat.transpose(), (springK**2)), RigidityMat)
    return dynMat
#===========================================================================================================================================

    
    

 
    
    
