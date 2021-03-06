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

PI = np.pi

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
        
        #add diagonal edges for each square, diagonals goes upper-left to lower-right
        edges.extend([[n*height + m + 1, m + n*height + height] for m in range(height - 1) for n in range(width - 1)])
        
    return (vertices, np.array(edges))
#===========================================================================================================================================
    
#===========================================================================================================================================
# returns a triangular lattice with the corresponding edges
#===========================================================================================================================================
def triangular_lattice(width, height=None, ax=1.0, ay=None, randomize=False, randomBox=0):

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
        ay = ax * np.sin(PI/3)
        
    if randomize and randomBox == 0.0:
        randomBox = 1.0
    elif not randomize:
        randomBox = 0
    
    # a square lattice with random displacements, counting starts from left and goes up column after column 
    vertices = np.array([[(i + 0.5* np.mod(j, 2)) * ax + randomize* np.random.uniform(-randomBox*ax/2.0, randomBox* ax/2.0), j * ay + 
                      randomize* np.random.uniform(-randomBox*ay/2.,randomBox* ay/2.0)]
                      for j in range(height) for i in range(width)])
    
    
    
    #horizontal edges
    edges = [[n + m * width, n + 1 + m * width] for m in range(height) for n in range(width - 1)]
    
    # up and to the right, starting from even rows
    edges.extend([[n + 2*m*width, n + width + 2*m*width] for m in range(height//2) for n in range(width)])
    
    # up and to the right, starting from odd rows
    edges.extend([[n + (2*m + 1)*width, n + width + 1 + (2*m + 1)*width] for m in range((height - 1)//2) for n in range(width - 1)])
    
    # down and to the right, starting from even rows
    edges.extend([[n + 2*m*width, n - width + 2*m*width] for m in range(1, (height + 1)//2) for n in range(width)])
    
    # down and to the right, starting from odd rows
    edges.extend([[n + (2*m + 1)*width, n - width + 1 + (2*m + 1)*width] for m in range((height - 1)//2) for n in range(width - 1)])
    
    
        
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
# returns an edge array where every point is connected to any other
#===========================================================================================================================================  
def connect_all_of_square(num_of_verts):
    """
    Returns an edge array with all the points connected to each other. It assumes a square + gray matter
    The edge array is a list neighbors, for each edge it gives the an array the indices of the points 
    attached to it np.array([index1, index2]). The edges are ordered in such a way that the square vertices are given first
    """
   
    #connect the square vertices
    edge_array = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [2, 1]];
    
    
    all_to_gray = connect_all_to_gray(np.arange(4), np.arange(4, num_of_verts))
    
    #return all the edges together into a single edge array
    return np.vstack((np.array(edge_array), all_to_gray))
#===========================================================================================================================================
    
#===========================================================================================================================================
# returns an edge array where every point is connected to any other
#===========================================================================================================================================  
def connect_all_of_tri(num_of_verts, include_tri_edges = False):
    """
    Returns an edge array with all the points connected to each other. It assumes a triangle + gray matter
    The edge array is a list neighbors, for each edge it gives the an array the indices of the points 
    attached to it np.array([index1, index2]). The edges are ordered in such a way that the triangle vertices are given first
    """
   
    #connect the tri vertices
    #edge_array = [[0, 1], [1, 2], [0, 2]];
    #edge_array = []
    
    all_to_gray = connect_all_to_gray(np.arange(3), np.arange(3, num_of_verts))
    
    if not include_tri_edges:
        return all_to_gray
    else: 
        tri_edges = [[0, 1], [1, 2], [0, 2]];
        
        #return all the edges together into a single edge array
        return np.vstack((np.array(tri_edges), all_to_gray))
#===========================================================================================================================================

#===========================================================================================================================================
# returns an edge array where every added point is maximally connected to the rest
#===========================================================================================================================================    
def connect_all_to_gray(original_verts, added_verts):
    """
    The edge array is a list neighbors, for each edge it gives the an array the indices of the points 
    attached to it np.array([index1, index2]).
    """
    edge_array = [];
    
    #loop over all added point points
    for vert_index in added_verts:
        # for each point make an edge to every point on the square
        
        all_verts = np.hstack((original_verts, added_verts))
            
        for neib_index in all_verts:
            
            if(vert_index != neib_index):
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
# takes in a triangulated mesh and returns the faces as (idx1, idx2, idx3)
#===========================================================================================================================================    
def get_mesh_faces(edge_array):
    """
    Uses an edge array of mesh to generate the faces of the mesh. For each triangle in the mesh this returns the list of indices 
    contained in it as a tuple (index1, index2, index3)
    """
    triangles = []
    
    neibs = neibs_from_edges(edge_array)
    
    for edge in edge_array:
        for vert in get_opposite_verts(neibs, edge):
            triangle = sorted([edge[0], edge[1], vert])
            if not (triangle in triangles):
                triangles.append(sorted([edge[0], edge[1], vert]))
    
    return triangles
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
def makeDynamicalMat(edgeArray=np.zeros(1), verts=np.zeros(1), RigidityMat= np.zeros(1), springK= np.zeros(1),  
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
        #print("This is not supposed to be true during minimization because we would be using a rigidity matrix")
        if not verts.any():
            raise NameError("Please either provide the rigidity matrix or the vertices for calculating the dynamical matrix")
        if numOfVerts < 1:
            numOfVerts = len(set(list(edgeArray.flatten())))

        RigidityMat = makeRigidityMat(verts, edgeArray, numOfVerts, numOfEdges) 
        #print(RigidityMat.shape)
        
    if(not springK.any()):
        springK = np.ones(numOfEdges)

    if not negativeK:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK**2)), RigidityMat)
    else:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK)), RigidityMat)
    return dynMat
#================================================================================================================================================


def deformation_energy(u, dyn_mat = np.zeros(1), vertices = None, edges = None):
    '''
    '''
    
    if(u.shape[0] != u.size):
        v  = u.flatten()
    else: v = u
    
    if dyn_mat.any():
        return np.dot(np.dot(v, dyn_mat), v)

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
    
    
    
    
    plt.scatter(Points[:,0], Points[:,1])
#================================================================================================================================================


#===============================================================================================================================================
# Returns a Neighbor list and a map between the edges list and Neighbor list.
#===============================================================================================================================================
def neibs_from_edges(edge_list, num_of_verts = -1):
    """
    a neighbor list is a list of (num_of_verts) lists each contain the indices of the neighbors of the corresponding verts.
    The vertex being refered to is assumed to be implied by the position of it's neighbors in the neib_list.
    
    We also return a map between the edge_list and the neighbor list.
    
             edges = make_cyl_edges(2,3,is_capped=False)
    Example: neibs_from_edges(edges)
    [[2, 3, 5, 1],
  [0, 4, 3, 2],
  [0, 1, 5, 4],
  [0, 1, 5, 4],
  [1, 2, 3, 5],
  [0, 2, 3, 4]],
    
"""

    if (num_of_verts < 1):
        num_of_verts = len(set(list(edge_list.flatten())))
    #num_of_edges = edge_list[:, 0].size
    
    #for each row neib_list gives the neighbor indices of the vertex correspoding to the row index                       
    neib_list = [[]]*num_of_verts
    
    #For each index in a row, neibs-to-edges will point to the correct edge index  in the edge array               
    neibs_to_edges = [[]]*num_of_verts 
     
    
    #loop over the vertices      
    for Vindx in np.nditer(np.arange(num_of_verts)):
        
        #for each vertex list it's neighbors by finding the edges it appears in
        for Eindx, edge in enumerate(edge_list): 
            
            #when you find it in one index of the edge, 
             if edge[0] == Vindx:
            #add the second index to the neib_list     
                neib_list[Vindx] = [neib_list[Vindx], edge[1]]  #doing it this way is important for the nesting to come out right
            #make the map too
                neibs_to_edges[Vindx] = [neibs_to_edges[Vindx], Eindx]
                
                
            #when you find it in one index of the edge, 
             elif edge[1] == Vindx:
            #add the second index to the neib_list     
                neib_list[Vindx] = [neib_list[Vindx], edge[0]]
            #make the map too
                neibs_to_edges[Vindx] = [neibs_to_edges[Vindx], Eindx]
                
        neib_list[Vindx] = flatten(neib_list[Vindx]) #flatten the rows to get rid of extra nesting
        neibs_to_edges[Vindx] = flatten(neibs_to_edges[Vindx])
     
                
    return neib_list
#===============================================================================================================================================



#===============================================================================================================================================
#find the two vertices oppisite to each edge
#===============================================================================================================================================
def get_opposite_verts(neib_list, edge):
    ''' Calculates the dihedral vertices for the triangulation from the neighbor list.
        return (numEdges, 2) array containing the indices of the two 
        vertices corresponding to the triangles that include the edge. In other words,
        return the two vertices opposite to the edge. This is useful for implementing 
        bending rigidity
        
        Neibs can be and array or list
        '''
       
    #find the two triangles intersecting at the edge by finding common neighbors of the two edge vertices of the edge
    return  np.intersect1d(neib_list[edge[0]], neib_list[edge[1]])

#===============================================================================================================================================


#=================================================================================
#flatten a list 
#=================================================================================
def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
#=================================================================================     
    
import warnings    
#===========================================================================================================================================
# angle from vec1 to vec2
#===========================================================================================================================================   
def angle_between (vec1, vec2):
    """
    Angle from vec1 to vec2. 
    This gives the angle you have to rotate vec1 to make it parallel to vec2, could be negative.
    """
    
    norm1 = la.norm(vec1)
    norm2 = la.norm(vec2)
    
    if (norm1 == 0 or norm2 == 0):
        warnings.warn("Warning, angle between zero vector assumed to be pi/2 ")  
        return np.pi/2
    
    #find the angles with respect to the x axis then take the different. us x hat helps get the sign right
    return np.arccos(np.dot(vec2, [1, 0])/(norm2))*np.sign(vec2[1]) - np.arccos(np.dot(vec1, [1, 0])/(norm1))*np.sign(vec1[1])
#========================================================================================================================================

#===========================================================================================================================================
# returns a projection operator that projects on the complement of the rigid trasnformations of a given lattice
#===========================================================================================================================================   
def get_rigid_transformations (vertices):
    """
    returns a projection operator that projects on the complement of the rigid trasnformations of a given lattice
    
    The lowest energy deformations of any (free floating) mesh are uniform rotations and translations. When considering the possible 
    deformations of a mesh or lattice it is convinient to project out the components along the rigid transformations. 
    Assuming that the rigid transformations are the only one with zero energy cost, they will be perperdicular to all the other 
    eigenvalues of the dynamical matrix. So we will project them out from the beginning.
   
    
    Input: 
    vertices: shape is (num_of_vertices, 2). Reference positions of the points on the lattice 

    """
    num_of_vertices = vertices.shape[0]
    
    # the translation of each vetex in the x direction (flattened)
    x_translations = np.array([1, 0]*(num_of_vertices))
    # the translation of each vetex in the x direction (flattened)
    y_translations = np.array([0, 1]*(num_of_vertices))
    
    
    average_position = np.sum(vertices, 0)/num_of_vertices
    
    translated_verts = vertices - average_position
    
    
    # this is the same as  uR_i = hat(z) X (P_i - Paverage) for a CC rotation around z axis
    rotation_matrix = np.array([[0, 1],[-1, 0]])
    z_rotation = np.dot(translated_verts,rotation_matrix) #will be flattened on return
    
    return np.array([x_translations, y_translations, z_rotation.flatten()])
#========================================================================================================================================

#===========================================================================================================================================
# generates a projection operator and it's complement from a set of vector bases
#===========================================================================================================================================   
def get_complement_space (vectors, only_complement = True):
    """
    returns a projection operator on the space that is complement to the span of the given vectors.
    
    vectors is an array of vectors. The dimensionality of the ambient space is determined by the lengths of the individual vectors. 
    which are all equal. The number of indepently given vectors determines the dimensionality of the space that we project onto
    (or way from). 
    
    for each one of the vectors we will use the method projection_operator(vector).
    
    Input: 
    vectors: shape is (num_of_vectors, space_dimensionality). 
    
    Example: get_complement_space([[1,1,1]])[0]
             Out[73]: 
             array([[ 0.66666667, -0.33333333, -0.33333333],
                       [-0.33333333,  0.66666667, -0.33333333],
                       [-0.33333333, -0.33333333,  0.66666667]])
    """
    vectors = np.array(vectors)
    
    # for example this would be 2*N for the displacement field of N particles in 2D
    space_dimensionality = vectors.shape[1]
    
   #onto means that the projection operator returns a vector in the space spanned by vectors
    onto_Projection = np.zeros((space_dimensionality, space_dimensionality))
    
    for vector in vectors:
        
        #get component orthogonal to the previous ones.
        orth_vector = vector - np.dot(onto_Projection, vector)
        
        onto_Projection +=  projection_operator(orth_vector)
        
    complement_Projector = np.eye(space_dimensionality) - onto_Projection 
    
    if only_complement: return complement_Projector
    else: return complement_Projector, onto_Projection
#========================================================================================================================================


#===========================================================================================================================================
# generates a projection operator onto the direction of the given vector
#===========================================================================================================================================   
def projection_operator(proj_vector):
    """
    Generates a projection operator onto the direction of the given vector.
    
    The projection operator when acting on any other vector will return the component of the vector
    that is along the given by proj_vector
    
        
    Input: 
    proj_vector: shape is (space_dimensionality, ).  Project onto this
    
    """
    proj_vector = np.array(proj_vector)
    
    vec_SqrMag = np.sum(proj_vector**2)
    
    # raise an error for zero norm instead of returning zero #TODO maybe return a warning instead and return zero?
    if(vec_SqrMag == 0):
        raise NameError("Zero norm vector given! can't project onto zero vector direction.")
            
    #finding the projection operator involves the outer product of the vector with itself. 
    outer_prod = np.outer(proj_vector, proj_vector)
    
    return outer_prod/vec_SqrMag # = projection_operator
#========================================================================================================================================



























