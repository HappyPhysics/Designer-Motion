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
from numpy import linalg as la
from enum import Enum

#===========================================================================================================================================
# returns a square lattice with the corresponding edges
#===========================================================================================================================================
def squareLattice(width, height=None, ax=1.0, ay=None, randomize=False, randomBox=0, is_with_diagonals=True):

    """
    
    returns a square lattice and edges, including diagonals.
    
    width: number of points in the x direction
    height: number of points in the y direction
    ax,ay: spacing between points in the x,y directions
    randomize: Add randomness to the positions of the points, its a uniform distribution within the box...
    randomBox: fraction of the unit cell that the point might be in. It will be 1 if nothing is entered and randomize is true
    is_with_diagonals: whether the diagonal bonds (lower-left to upper-right) will be added to the edge array. True by default.
    
    Example: squareLattice(2) = (array([[ 0.,  0.],
        [ 0.,  1.],
        [ 1.,  0.],
        [ 1.,  1.]]), array([[0, 1],
        [2, 3],
        [0, 2],
        [1, 3],
        [0, 3]]))
    
    squareLattice(2)[0] is the vertices array
    squareLattice(2)[1] is the edge array
    """
    if height is None:
        height = width
        
    if ay is None:
        ay = ax
        
    if randomize and randomBox == 0.0:
        randomBox = 1.0
    elif not randomize:
        randomBox = 0
    
    # a square lattice with random displacements, counting starts from left and goes up column after column 
    vertices = np.array([[i * ax + randomize* np.random.uniform(-randomBox*ax/2.0, randomBox* ax/2.0), j * ay + 
                      randomize* np.random.uniform(-randomBox*ay/2.,randomBox* ay/2.0)]
                      for i in range(width) for j in range(height)])
    
    #vertical edges
    edges = [[n + m*height, m*height + n + 1] for m in range(width) for n in range(height - 1)]
    
    #horizontal edges
    edges.extend([[n*height + m, m + n*height + height] for m in range(height) for n in range(width - 1)])
    
    if(is_with_diagonals):
        #add diagonal edges for each square, diagonals goes lower-left to upper-right
        edges.extend([[n*height + m, m + n*height + height + 1] for m in range(height - 1) for n in range(width - 1)])
        
        
    return (vertices, np.array(edges))
#===========================================================================================================================================
 
#===========================================================================================================================================
# returns an edge array where every point is connected to any other
#===========================================================================================================================================    
def connect_all_verts(num_of_verts):
    """
    Returns an edge array with all the points connected to each other. 
    The edge array is a list neighbors, for each edge it gives the an array the indices of the points 
    attached to it np.array([index1, index2])
    """
    edge_array = [];
    #loop over all points
    for vert_index in range(num_of_verts - 1):
        # for each point make an edge to every other points with index bigger than it's own index
        for neib_index in range(vert_index + 1, num_of_verts):
            edge_array.append([vert_index, neib_index])
    #return all the edges together into a single edge array
    return np.array(edge_array)
#===========================================================================================================================================
    

#===========================================================================================================================================
# returns an edge array where every point is connected to the vertices of a square
#===========================================================================================================================================    
def connect_all_to_square(num_of_added_points):
    """
    The edge array is a list neighbors, for each edge it gives the an array the indices of the points 
    attached to it np.array([index1, index2]).
    
    This method assumes a square, or 4 initial vertices that are connected to each other. Then (num_of_added_points) points
    are added to the lattice and each one of these added points is connected to all the vertices of the initial square
    
    """
    edge_array = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    #loop over all added point points
    for vert_index in range(4, 4 + num_of_added_points):
        # for each point make an edge to every point on the square
        for neib_index in range(4):
            edge_array.append([vert_index, neib_index])
            
    #return all the edges together into a single edge array
    return np.array(edge_array)
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
# returns the indices of the input node and the output node
#===========================================================================================================================================    
def getIONodes(verts, height):
    """
    getIONodes(verts, height)
     returns the indices of the input node and the output node. This is done in the dumbest possible way: 
         Input = indices of the left most verts. There's (height) many of them 
         Output = indices of right most verts. There's (height) many of them 
    
    Example: sq = squareLattice(2)[0];
             getIONodes(sq, 2)
    Out: (array([0, 1], dtype=int64), array([2, 3], dtype=int64))
    """
        
    #select the x components of the vertices 
    inIndx = np.argpartition(verts[:, 0], height)
    outIndx = np.argpartition(verts[:,0], -height)
    return (inIndx[:height], outIndx[-height:])
#===========================================================================================================================================

#===========================================================================================================================================
# returns the corresponding indices in the flattened array
#===========================================================================================================================================    
def flattenedIndices(indices, numOfVerts):
    """
    The displacement vectors are naturally ordered into (numOfvectors, 2) ararys however for computing the energy and minimization it is
    better to flatten this array. 
    This causes inconviniences because we want to be able to select the indices of points, each index would correspond to a 2-vector. 
    this method return the corresponding indices in the flattened array
    
    Example: sq = squareLattice(2)[0];
             s = getIONodes(sq, 2)[0] #gets input nodes array([0 , 1])
             s
    Out: array([0, 1, 2, 3])
    """
    return (np.arange(2*numOfVerts).reshape((numOfVerts, 2))[indices]).flatten()    
#===========================================================================================================================================    

#===========================================================================================================================================
# returns the adjacency matrix as an array
#===========================================================================================================================================  
def makeAdjacencyMatrix(edgeArray, numOfVerts=-1):
    """
    makeAdjacencyMatrix(edgeList):
        Takes in the edgeArray then converts it to a list, which has elements of the form [vert1, vert2] and finds the (2 numOfVerts x 2 numOfVerts) 
        adjacency matrix.
        
      Example: verts, edges = squareLattice(2)
            EdgeMat = makeEdgeMatrix(edges)
       Out:  array([[ 0.,  1.,  1.,  1.],
       [ 1.,  0.,  0.,  1.],
       [ 1.,  0.,  0.,  1.],
       [ 1.,  1.,  1.,  0.]])
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
# returns the Edge Matrix given the edge array
#=========================================================================================================================================== 
def makeEdgeMatrix1(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)):
    """
    makeEdgeMatrix(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (numOfEdges, numOfVerts).
        For each edge there is a row in the matrix, the row is only nonzero at the positions 
        corresponding to the points connected by that edge, one of them will be 1 the other will be -1.
        When useSpringK is True, each edge will be multiplied by the spring constant which is a convenient thing
        
        Example: verts, edges = squareLattice(2)
            EdgeMat1 = makeEdgeMatrix(edges); EdgeMat1
       Out:  array([[ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.]])
    """
    
    if useSpringK:
        if (springK < 0).all():
            springK = np.ones(numOfEdges)
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    edgeMat = np.zeros((2*numOfEdges, 2*numOfVerts)) #CONSIDER MODIFICATION: dtype=np.dtype('int32')
    
    for edgeNum, edge in enumerate(edgeArray):
        if not useSpringK:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 
            edgeMat[2*edgeNum, 2*edge[1]] = -1 
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1
        else:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 *springK[edgeNum]
            edgeMat[2*edgeNum, 2*edge[1]] = -1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1 *springK[edgeNum]
        
    return edgeMat
#===========================================================================================================================================  
    
#===========================================================================================================================================
# returns the Edge Matrix given the edge array
#=========================================================================================================================================== 
def makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1):
    """
    makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (numOfEdges, 2*numOfEdges).
        For each edge there is a row in the matrix, the row is only nonzero at 2 positions in which 
        it is equal to 1, this is used for adding together the two rows corresponding to the different
        x and y componenets that resulted from multiplying edgeMatrix1 with the vertices. 
        
        Example: verts, edges = squareLattice(2)
            EdgeMat2 = makeEdgeMatrix2(edges); EdgeMat2
            array([[ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.]])
    """

    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    edgeMat = np.zeros((numOfEdges, 2*numOfEdges)) #CONSIDER MODIFICATION: dtype=np.dtype('int32')
    
    for edgeNum, edge in enumerate(edgeArray):
        edgeMat[edgeNum, 2*edgeNum] = 1 
        edgeMat[edgeNum , 2*edgeNum + 1] = 1 
       
        
    return edgeMat
#===========================================================================================================================================  
    
#===========================================================================================================================================
# returns the Rigidity Matrix as an array
#===========================================================================================================================================  
def makeRigidityMat(verts, edgeArray=np.array([0]), numOfVerts=-1, numOfEdges=-1, edgeMat1 = np.zeros(1), edgeMat2 = np.zeros(1)):
    """
    makeRigidityMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1,method):
        Takes in the edgeArray then finds Rigidity matrix. The rigidity matrix helps
        to find the bond stretching to linear order in displacement u which has 
        size = 2 numOfVerts. Bond stretchings are equal to 
        dl_e = R_ei * u_i, where i is summed over.
        
        The method parameter desides how the rigidity matrix will be computed. When method = 1
        the edgeMatrices will be used, which is useful when the vertex positions are minimized over. 
        verts should be flattened when this method is used
        
    Example1: 
            sq = squareLattice(2, randomize=False); 
            edgeMat1= makeEdgeMatrix1(sq[1])
            edgeMat2 = makeEdgeMatrix2(sq[1])
            R = makeRigidityMat(sq[0].flatten(), edgeMat1=edgeMat1, edgeMat2=edgeMat2)
            R 
        Out: array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])
    
    Example2:
        (verts, edges) = squareLattice(2, randomize=False); 
        edgeMat1 = 
            R = makeRigidityMat(verts, edges) ;R
      array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])       
    """
    
    if not (edgeMat1==0).all():
        RMat = np.dot(edgeMat1, verts)
        RMat = np.multiply(edgeMat1.transpose(), RMat).transpose()
        return np.dot(edgeMat2, RMat)
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
def makeDynamicalMat(edgeArray = np.zeros(1), verts = np.zeros(1), RigidityMat= np.zeros(1), springK= np.zeros(1),  
                     numOfVerts=-1, numOfEdges=-1, negativeK=False):
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
        print("This is not supposed to be true during minimization because we would be using a rigidity matrix")
        if not verts.any():
            raise NameError("Please either provide the rigidity matrix or the vertices for calculating the dynamical matrix")
        if numOfVerts < 1:
            numOfVerts = len(set(list(edgeArray.flatten())))

        RigidityMat = makeRigidityMat(verts, edgeArray, numOfVerts, numOfEdges) 
    
    if(not springK.any()):
        springK = np.ones(numOfEdges)

    if not negativeK:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK**2)), RigidityMat)
    else:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK)), RigidityMat)
    return dynMat
#================================================================================================================================================


#================================================================================================================================================
# Normalize a vector, assuming that norm(V) != 0 
#================================================================================================================================================
def normalizeVec(V):
    """
    Returns the vector V normalized to unity. It doesn't do a lot of smart things, the vector 
    is assumed to not have zero norm and the input array should be 1-dimensional.
    """
    V = V/la.norm(V)   
    return   V
#================================================================================================================================================

#================================================================================================================================================
# returns the number of edges given the edge array
#================================================================================================================================================
def get_num_of_edges(edge_array):
    """
    This method checks the shape of the edge array and returns an error if the number of neighbors is not equal to 2. 
    If there are no errors the number of edges is returned
    """
     #check that edge_array dimensions makes sense
    num_of_edges, num_Of_Neibs  = edge_array.shape[0], edge_array.shape[1]

    if(num_Of_Neibs != 2):
        raise NameError("Something is wrong with the edge array. Number of neighbors should be 2")
        
    return   num_of_edges
#================================================================================================================================================
    
#================================================================================================================================================
# returns the number of edges given the edge array
#================================================================================================================================================
def get_num_of_verts(vertices, dimensionality = 2):
    """
    This method checks the shape of the vertex array and returns an error if the dimensionality of space is not equal to 2. 
    If there are no errors the number of vertices is returned.
    
    dimentionality is the number of spacial dimensions will be taken as 2 by default.
    """
     #check that vertices dimensions makes sense
    num_of_verts, num_Of_dimensions  = vertices.shape[0], vertices.shape[1]

    if(num_Of_dimensions != dimensionality):
        raise NameError("Something is wrong with the vertex array. Dimensions of space should be 2")
        
    return   num_of_verts
#================================================================================================================================================
 
    
    
