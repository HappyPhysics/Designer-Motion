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
# Finds the lattice that has the desired motion as lowest energy mode
#=========================================================================================================================================== 
def find_desired_lattice(disp_type=DispType.random, edgeType = EdgeTypes.all_connected, num_of_added_verts = 3, mesh = TRI_LATTICE):
    print("\n")
    num_of_verts = mesh[0].size//2
    
    # define the desired displacement field of the entire lattice
    desired_disp = make_desired_disp(mesh[0], DeformType=disp_type, num_of_vertices=num_of_verts)
    desired_disp = np.hstack(([0,0,0], desired_disp)).reshape((num_of_verts, 2))
   
    # take in an edge list, generate a list or an array of triangles with elements (indx1, indx2, indx3)
    triangles = get_mesh_faces(mesh[1])
    
    
    results = []
  
    # run over all the triangeles or faces of the mesh. 
    for tri_ID, triangle in enumerate(triangles):
            
        #rotate by 45 degrees so that the first edge is not in the x axis 
        triangle_verts = np.dot(np.array(mesh[0][triangle]), rotation_matrix(PI/4))
        
        #get the indices of the triangle as a mask and use it to return the desired displacements of the triangle
        triangle_disps = desired_disp[triangle]
        
        
        # change thier frame of reference to the canonical one (from little triangle).
        canon_disps = np.array(lab_to_triangle_frame(triangle_disps))
       
        
        
        # call little triangle to add the gray matter
        res = find_desired_face(num_of_added_verts=num_of_added_verts, face_Disps=canon_disps,
                           face_verts=triangle_verts, face_ID=tri_ID + 1)
        
         
        results.append(res)
        
        
        #handle edges
        #add the grey matter on the end of the lattice.
        #increment the corresponding edges in the edge array
        #add in the correct spring constants
        print("\n")
        
    return results
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
def lab_to_triangle_frame(triangle_disps, direction = 1):
    """
        Translates between the lab and triangle frames. triangle frame is the natural frame introduced in the script Little_triangle.
        
        Input:
            direction:  gives the direction of the transformation. If direction is +1 the transformation goes from lab to triangle frame.
                if direction is -1 then the transformation goes the other way
                (note that the outputs will have different shapes because of flattening and reshaping)
                
            
    """

    #subtract the displacement of the first vertex from all the displacements. 
    new_disps = triangle_disps - triangle_disps[0]

    # find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between(new_disps[1], [0, 1]))
    
    # apply this rotation to all the displacements. 
    new_disps = np.dot(rot_matrix, new_disps.transpose())
    # return the displacemetns in canonical form.     
    return new_disps.transpose().flatten()[3:]
    
    
#===========================================================================================================================================    

#===========================================================================================================================================
# changes from the canonical form of the triangle back to the "lab" frame (acts on vertex positions)
#===========================================================================================================================================     
def triangle_to_lab_frame(c_vertices, triangle_disps):
    """
        Translates the solution given by Little_triangle back to the original lab frame. Uses the original triangle displacements to do so.
        
        Input:
            c_vertices:  vertices return as the solution to the minimization in Little_triangle. They shuold be reshaped before an output 
            is produced. 
            triangle_disps: The original desired triangle displacements
       
        Output: the vertices put in the lab frame and reshaped into a list of 2-vectors          
            
    """
    
    # find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between([0, 1], new_disps[1]))
    
    
    #subtract the displacement of the first vertex from all the displacements. 
    triangle_disps = np.hstack(([0,0,0], canon_disps)).reshape(())

    # find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between(new_disps[1], [0, 1]))
    
    # apply this rotation to all the displacements. 
    new_disps = np.dot(rot_matrix, new_disps.transpose())
    # return the displacemetns in canonical form.     
    return new_disps.transpose().flatten()[3:]
    
    
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
        print("Warning, angle between zero vector assumed to be pi/2 ") #TODO see if I can use Warning without stoping the execution 
        return np.pi/2
    
    #find the angles with respect to the x axis then take the different. us x hat helps get the sign right
    return np.arccos(np.dot(vec2, [1, 0])/(norm2))*np.sign(vec2[1]) - np.arccos(np.dot(vec1, [1, 0])/(norm1))*np.sign(vec1[1])
#===========================================================================================================================================
    
    
def recombine_lattice(results):
    #start with the original lattice
    new_lattice = TRI_LATTICE
    
    #number of vertices in the triangle lattice
    num_of_triangle_verts = TRI_LATTICE[0].size//2
    
    #keep track of the number of added gray matter verts so far
    current_added_verts = 0
    
    #for each triangle of the lattice, pick out the triangles one by one. 
    for vert_indx in range(num_of_verts):
      
        #check if the vert is a lower left corner of a triangle, these indicate the different triangles
        if(get_triangle_indices(vert_indx, num_of_triangle_verts, output=OutType.indices) is None):
            continue
        
        triangle_indices = get_triangle_indices(vert_indx, num_of_triangle_verts, output=OutType.indices)
        
        #the number of extra vertices in the current triangle
        num_extra_verts = results[vert_indx - vert_indx//LATTICE_WIDTH][0].shape[0] - LITTLE_triangle.shape[0]
        print("num of extra verts in triangle ", vert_indx - vert_indx//LATTICE_WIDTH, " is: ", num_extra_verts)
        
        gray_indices = np.arange(num_extra_verts) + num_of_triangle_verts + current_added_verts
        
        
        
        current_added_verts += num_extra_verts
        
        #add the corresponding grey matter to the end of the vertex array.
    #get the vertices of the triangle and added gray matter and connect everything 
    #exclude the triangle bonds because they are already there
    
    pass
    
    
    
    
    
    
    
    
    
    
    
