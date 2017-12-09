# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:39:08 2017

@author: Salem

This script starts with a lattice and then divides it into a bunch of triangles.
Then for each triangle it calls Little-Triangle and implements the addition of gray matter to
each triangle separately. Before calling Little-Triangle a change of frame must be implemented. 

Methods: 
        
        find_desired_lattice(disp_type=DispType.random):
            Finds the lattice that has the desired motion as lowest energy mode
        
            
        rotation_matrix(angle):
            rotates a vector by angle degrees 
        
        angle_between(vec1, vec2):
            angle from vec1 to vec2
            
"""

import numpy as np
import numpy.random as npr
import LatticeMaking
import Little_Triangle
from numpy import linalg as la
from matplotlib import pyplot as plt


import importlib
importlib.reload(LatticeMaking)
importlib.reload(Little_Triangle)

from LatticeMaking import *  #custom
from Little_Triangle import * #custom


# take in a triangulated mesh.
# generate a list or an array of triangles with elements (indx1, indx2, indx3)
# take in the desired displacements of each triangle. 
# transform the displacements to a canonical frame of reference. 
# run little_triangle on them. 
# store the results of all the little triangles. 
# combine the triangles back to a single mesh adding the extra vertices and edges.
# minimize the cost function for the whole lattice as a function of the original edges (optional).


PI = np.pi

# take in a triangulated lattice.
LATTICE_WIDTH = 3 # same as height
TRI_LATTICE = triangular_lattice(LATTICE_WIDTH) #contains vertices and edges as tuple


#===========================================================================================================================================
# A test lattice for changing a transverse wave to a longitudinal one
#=========================================================================================================================================== 
def wave_changer():
    
    verts = np.array([[0.0, 0.0], [0.0, 1.0], [np.sin(PI/3), 0.5], [np.sin(PI/3), -0.5]])
    
    edges = np.array([[0, 1], [0, 2], [0, 3], [1,2], [2,3]], dtype=np.int64)
    
    mesh = [verts, edges]
    
    disp = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])  
    
    test = True;
    while(test):
        res = find_desired_lattice(mesh=mesh, desired_disp=disp, edgeType=EdgeTypes.all_to_gray)
        test = check_answer(res)
    
    return res

#===========================================================================================================================================
# Finds the lattice that has the desired motion as lowest energy mode
#=========================================================================================================================================== 
def find_desired_lattice(disp_type=DispType.random, edgeType = EdgeTypes.all_connected, 
                         num_of_added_verts= 3, mesh = list(TRI_LATTICE), desired_disp = None):
    print("\n")
    num_of_verts = mesh[0].size//2
    
    mesh1 = mesh.copy()
    
    # define the desired displacement field of the entire lattice
    if desired_disp is None:
        desired_disp = make_desired_disp(mesh1[0], DeformType=disp_type, num_of_vertices=num_of_verts)
        desired_disp = np.hstack(([0,0,0], desired_disp)).reshape((num_of_verts, 2))
   
    # take in an edge list, generate a list or an array of triangles with elements (indx1, indx2, indx3)
    triangles = get_mesh_faces(mesh1[1])
    print("The mesh contrains the triangles: ", triangles)
    
    results = []
    if(edgeType == EdgeTypes.all_to_gray):
        mesh1[1] = np.ones((0,2), dtype=np.int64)
        springConstants = np.ones(0) # no springs started on the original mesh
    else:
        springConstants = np.ones(mesh1[1].size//2)
        
    
    total_added = 0
    
    # run over all the triangeles or faces of the mesh. 
    for tri_ID, triangle in enumerate(triangles):
            
        
        triangle_verts = np.array(mesh1[0][triangle])
        
        #get the indices of the triangle as a mask and use it to return the desired displacements of the triangle
        triangle_disps = desired_disp[triangle].flatten()
        
        # change thier frame of reference to the canonical one (from little triangle).
        #new_tri_verts, canon_disps = lab_to_triangle_frame(triangle_disps, triangle_verts)
       
        
        # call little triangle to add the gray matter
        res = find_desired_face(num_of_added_verts=num_of_added_verts, face_Disps=triangle_disps,
                           face_verts=triangle_verts, face_ID=tri_ID + 1, edgeType=edgeType)
        
        
        add_res_to_mesh(mesh1, res, triangle, np.arange(num_of_verts + total_added, num_of_verts + total_added + res[0].size//2))
         
        total_added += res[0].size//2
        
        #the spring constants they are not squared anymore, they will be squared when the energy is evaluated
        springConstants = np.hstack((springConstants, res[2]))
        
        results.append(res)
        
        
        #handle edges
        #add the grey matter on the end of the lattice.
        #increment the corresponding edges in the edge array
        #add in the correct spring constants
        print("\n")
        
    #springConstants /= np.max(springConstants)

    mask = springConstants**2 < 0.01
    
    #show vertices
    plotPoints(mesh1[0], num_of_originals=num_of_verts)    
     
    #spring constants are normalized so that the maximum value is one
    return results, mesh1, springConstants
#===========================================================================================================================================         
        
 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this enumumerates the possible outputs of get_triangle_indices
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OutType(Enum):
    mask = 1
    indices = 2
    #TRI_LATTICE = 3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
#===========================================================================================================================================
# puts the triangle displacements in canonical form
#=========================================================================================================================================== 
def get_triangle_indices(indx, num_of_verts, output = OutType.mask):
    """
    description
    
    """
    mask = np.zeros(num_of_verts, dtype=np.bool)
    
    #ignore the indices at the boundaries that are not at the Lower-Left corner of a triangle 
    if np.mod(indx + 1, LATTICE_WIDTH) == 0 or indx >= LATTICE_WIDTH *(LATTICE_WIDTH - 1):
        #return nothing because they do not represent a triangle
        return None
    
    if output is not OutType.mask:
        return np.array([indx, indx + 1, indx + LATTICE_WIDTH, indx + LATTICE_WIDTH + 1])
    # gets the indices of the triangle corresponding to indx (indx of lower left corner)
    # start with indx, then find the indices of the other three
    mask[indx] = mask[indx + 1] = mask[indx + LATTICE_WIDTH] = mask[indx + LATTICE_WIDTH + 1] = True
    
    return mask
#===========================================================================================================================================    
    
        
    
    # return thier positions and displacements to change frame. 

#===========================================================================================================================================
# puts the triangle displacements in canonical form
#===========================================================================================================================================     
def lab_to_triangle_frame(triangle_disps, triangle_verts, direction = 1):
    """
        Translates between the lab and triangle frames. triangle frame is the natural frame introduced in the script Little_triangle.
        
        Input:
            direction:  gives the direction of the transformation. If direction is +1 the transformation goes from lab to triangle frame.
                if direction is -1 then the transformation goes the other way
                (note that the outputs will have different shapes because of flattening and reshaping)
                
            
    """
# find the triangle vertices in canonical for, [0,0], [0, y],...
    new_verts = np.copy(triangle_verts)
    
    new_verts -= new_verts[0]
    
    new_verts = np.dot(new_verts, rotation_matrix(angle_between([0, 1],new_verts[1])))     
    
    
#for disps, find the amount needed to be rotated.
    
    #subtract the displacement of the first vertex from all the displacements. 
    new_disps = triangle_disps - triangle_disps[0]
    
    
    #rotation vectors, these are vectors normal to the position vector (starting from the pivot point)
    rotation_vecs = np.dot((new_verts), rotation_matrix(-PI/2)) # C.C. rotation
    print(rotation_vecs)
    # find (negative of ) the component of the first displacement perpendicular to the first edge. 
    # find the ratio of length of this component to the length of the first edge, this is the  rotation parameter. 
    rotation_param = -np.dot(new_disps[1], rotation_vecs[1])/(la.norm(rotation_vecs[1])**2)
    
    
#rotate the right amount    
    #start with the triangle positions. (DONE ABOVE)
    #subtract the first position from all of them. (DONE ABOVE)
    # rotate the result by 90 degrees. (DONE ABOVE)
    
    # multiply by the rotation parameter. 
    #add to displacements. 
    new_disps += rotation_param*rotation_vecs
    
    
    # return the displacemetns in canonical form.     
    return new_verts, new_disps.flatten()[3:]
    
    
#===========================================================================================================================================    

#===========================================================================================================================================
# changes from the canonical form of the triangle back to the "lab" frame (acts on vertex positions)
#===========================================================================================================================================     
def back_to_lab_frame(output, triangle_verts):
    """
       translates the gray matter back to lab frame         
            
    """
    #subtract the displacement of the first vertex from all the displacements. 
    output += triangle_verts[0]                       

    output = np.dot(output, rotation_matrix(-angle_between([0, 1], triangle_verts[1])))     
    
    
    return output
#===========================================================================================================================================   

#===========================================================================================================================================
# rotates a vector by angle degrees
#===========================================================================================================================================       
def rotation_matrix(angle):
    """
    rotates a vector by angle degrees in the counter clockwise direction. (2 by 2)
    """ 
    
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])     
#===========================================================================================================================================    

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
#===========================================================================================================================================
    

#===========================================================================================================================================
# adds the new results to the original mesh
#===========================================================================================================================================       
def add_res_to_mesh(mesh, res, triangle, new_verts):
    '''
    adds the extra vertices from res to the mesh. And adds the edges too after replacing the values of the indices 
    by the order in which they appear in the new mesh
    '''
   
    mesh[0] = np.vstack((mesh[0], res[0])) 
    
    a = res[1]
    #print("edges: ", a)
    palette = np.arange(0, 3 + new_verts.size)
    #print("palette: ", palette)
    key = np.hstack((triangle, new_verts))
    #print("key: ", key)
    #print("a.ravel:", a.ravel())
    index = np.digitize(a.ravel(), palette, right=True)
    #print("index: ", index)
    #print(key[index].reshape(a.shape))
    mesh[1] = np.vstack((mesh[1], key[index].reshape(a.shape)))
    
#===========================================================================================================================================    


#===========================================================================================================================================
# check the final answer
#===========================================================================================================================================     
def check_answer(result):
    '''
    checks whether the results returned match the desired dispalcement. The desired displacements are given explicitly for now. 
    
    '''

    face_verts = np.array([[0.0, 0.0], [0.0, 1.0], [np.sin(PI/3), 0.5], [np.sin(PI/3), -0.5]])
    desired_disp = np.array([ 0.        ,  1.        ,  0.        , -1.        , -1.        ,
        0.        ,  1.        ,  0.        ,  0.83382341,  0.56617335,
        0.53577003,  0.39596225,  0.55284973,  0.15255589,  0.85216995,
        0.05556033,  0.10565699,  0.17175687,  0.42789355,  0.83339344])

    
    
    
    mesh = result[1].copy()
    k = result[2].copy()
    #print("spring constants:" , k**2/max(k**2))
    mask = k**2/max(k**2) < 0.01
    
    
    dyn_mat = makeDynamicalMat(verts=mesh[0], edgeArray=mesh[1], springK=k)
    
    lowestEigVector = normalizeVec(la.eigh(dyn_mat)[1][:,3])
    
    # project out the euclidean  transforms
    euclid_transforms = get_rigid_transformations(mesh[0])
    euclid_projector = get_complement_space(euclid_transforms)
    
    projected_disp = normalizeVec(np.dot(euclid_projector, desired_disp.flatten()))    
    
    
    disp, disp_energy= check_energy(desired_disp[:face_verts.size], mesh, k)
    
    projected_disp = normalizeVec(np.dot(euclid_projector, disp)) 
    
    
    print("eigenvalues: ", lowestEigenVals(dyn_mat, 2, 4))
    print("result energy: " , disp_energy)
    
    
    
    dot_product = np.dot(projected_disp[:8], lowestEigVector[:8])
    print("desired dot product: ", dot_product, "\n\n")
    
    projected_disp[:8] *= np.sign(dot_product)
    
    print("differece:", (projected_disp[:8] - lowestEigVector[:8])  ) 
    
    
    if dot_product*np.sign(dot_product) < 0.995: return True

    return False;




def check_energy(original_disps, mesh, k):
    
    num_of_verts = mesh[0].size//2
    
    disps = npr.rand(num_of_verts*2)
    
    disps[:original_disps.size] = original_disps
    
    euclid_transforms = get_rigid_transformations(mesh[0])
    euclid_projector = get_complement_space(euclid_transforms)
    
    projected_disp = normalizeVec(np.dot(euclid_projector, disps))    
    
    # project out the euclidean  transforms
    
    
    dyn_mat = makeDynamicalMat(verts=mesh[0], edgeArray=mesh[1], springK=k)
    
    res = op.minimize(energy, projected_disp, method='Newton-CG', args=(dyn_mat, euclid_projector, original_disps), jac=energy_Der, 
                       hess=energy_Hess, options={'xtol': 1e-8, 'disp': False})
    
    return res.x, res.fun
    















