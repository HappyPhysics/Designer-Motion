# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:39:08 2017

@author: Salem

This script starts with a square lattice and then divides it into a bunch of squares.
Then for each square it calls Little-Square and implements the addition of gray matter to
each square separately. Before calling Little-Square a change of frame must be implemented. 

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
import Little_Square
from numpy import linalg as la
from matplotlib import pyplot as plt


import importlib
importlib.reload(LatticeMaking)
importlib.reload(Little_Square)

from LatticeMaking import *  #custom
from Little_Square import * #custom


#===========================================================================================================================================
# Makes a square lattice and gives a list of squares 
#===========================================================================================================================================  


LATTICE_WIDTH = 3 #by 10 vertices
SQUARE_LATTICE = squareLattice(LATTICE_WIDTH, is_with_diagonals=False) #contains vertices and edges as tuple


#===========================================================================================================================================
# Finds the lattice that has the desired motion as lowest energy mode
#=========================================================================================================================================== 
def find_desired_lattice(disp_type=DispType.random, edgeType = EdgeTypes.all_connected, num_of_added_verts = 3):
    print("\n")
    num_of_verts = SQUARE_LATTICE[0].size//2
    
    # define the desired displacement field of the entire lattice
    desired_disp = make_desired_disp(SQUARE_LATTICE[0], DeformType=disp_type, num_of_vertices=num_of_verts)
    desired_disp = np.hstack(([0,0,0], desired_disp)).reshape((num_of_verts, 2))
   
    
    results = []
  
    # Pick out the squares one by one. 
    for vert_indx in range(num_of_verts):
        
        #check if the vert is a lower left corner of a square, these index the different squares
        if(get_square_indices(vert_indx, num_of_verts) is None):
            continue
        
        #get the indices of the square as a mask and use it to return the desired displacements of the square
        square_disps = desired_disp[get_square_indices(vert_indx, num_of_verts)]
        
        # change thier frame of reference to the canonical one (from little square).
        canon_disps = lab_to_square_frame(square_disps)
        
        # call little square to add the gray matter
        res = find_desired_square(num_of_added_verts=num_of_added_verts, squareDisp=canon_disps, edgeType=edgeType,
                            square_ID=vert_indx - (vert_indx//LATTICE_WIDTH) + 1)
        
        #takes the positions and translates them to the right position
        res[0] = res[0] + [vert_indx//LATTICE_WIDTH, np.mod(vert_indx, LATTICE_WIDTH)]
         
        results.append(res)
        
        
        #handle edges
        #add the grey matter on the end of the lattice.
        #increment the corresponding edges in the edge array
        #add in the correct spring constants
        print("\n")
        
    return results
#===========================================================================================================================================         
        

# store the resulting edges and vertices
# change the vertices back to original frame of reference.
# add all of the squares together to get the entire final lattice. 
# test that lowest energy modes of the lattice satisfy the desired motion.
 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this enumumerates the possible outputs of get_square_indices
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OutType(Enum):
    mask = 1
    indices = 2
    #square_lattice = 3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
#===========================================================================================================================================
# puts the square displacements in canonical form
#=========================================================================================================================================== 
def get_square_indices(indx, num_of_verts, output = OutType.mask):
    """
    description
    
    """
    mask = np.zeros(num_of_verts, dtype=np.bool)
    
    #ignore the indices at the boundaries that are not at the Lower-Left corner of a square 
    if np.mod(indx + 1, LATTICE_WIDTH) == 0 or indx >= LATTICE_WIDTH *(LATTICE_WIDTH - 1):
        #return nothing because they do not represent a square
        return None
    
    if output is not OutType.mask:
        return np.array([indx, indx + 1, indx + LATTICE_WIDTH, indx + LATTICE_WIDTH + 1])
    # gets the indices of the square corresponding to indx (indx of lower left corner)
    # start with indx, then find the indices of the other three
    mask[indx] = mask[indx + 1] = mask[indx + LATTICE_WIDTH] = mask[indx + LATTICE_WIDTH + 1] = True
    
    return mask
#===========================================================================================================================================    
    
        
    
    # return thier positions and displacements to change frame. 

#===========================================================================================================================================
# puts the square displacements in canonical form
#===========================================================================================================================================     
def lab_to_square_frame(square_disps, direction = 1):
    """
        Translates between the lab and square frames. square frame is the natural frame introduced in the script Little_Square.
        
        Input:
            direction:  gives the direction of the transformation. If direction is +1 the transformation goes from lab to square frame.
                if direction is -1 then the transformation goes the other way
                (note that the outputs will have different shapes because of flattening and reshaping)
                
            
    """

    #subtract the displacement of the first vertex from all the displacements. 
    new_disps = square_disps - square_disps[0]

    # find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between(new_disps[1], [0, 1]))
    
    # apply this rotation to all the displacements. 
    new_disps = np.dot(rot_matrix, new_disps.transpose())
    # return the displacemetns in canonical form.     
    return new_disps.transpose().flatten()[3:]
    
    
#===========================================================================================================================================    

#===========================================================================================================================================
# changes from the canonical form of the square back to the "lab" frame (acts on vertex positions)
#===========================================================================================================================================     
def square_to_lab_frame(c_vertices, square_disps):
    """
        Translates the solution given by Little_Square back to the original lab frame. Uses the original square displacements to do so.
        
        Input:
            c_vertices:  vertices return as the solution to the minimization in Little_Square. They shuold be reshaped before an output 
            is produced. 
            square_disps: The original desired square displacements
       
        Output: the vertices put in the lab frame and reshaped into a list of 2-vectors          
            
    """
    
    # find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between([0, 1], new_disps[1]))
    
    
    #subtract the displacement of the first vertex from all the displacements. 
    square_disps = np.hstack(([0,0,0], canon_disps)).reshape(())

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
    new_lattice = SQUARE_LATTICE
    
    #number of vertices in the square lattice
    num_of_square_verts = SQUARE_LATTICE[0].size//2
    
    #keep track of the number of added gray matter verts so far
    current_added_verts = 0
    
    #for each square of the lattice, pick out the squares one by one. 
    for vert_indx in range(num_of_verts):
      
        #check if the vert is a lower left corner of a square, these indicate the different squares
        if(get_square_indices(vert_indx, num_of_square_verts, output=OutType.indices) is None):
            continue
        
        square_indices = get_square_indices(vert_indx, num_of_square_verts, output=OutType.indices)
        
        #the number of extra vertices in the current square
        num_extra_verts = results[vert_indx - vert_indx//LATTICE_WIDTH][0].shape[0] - LITTLE_SQUARE.shape[0]
        print("num of extra verts in square ", vert_indx - vert_indx//LATTICE_WIDTH, " is: ", num_extra_verts)
        
        gray_indices = np.arange(num_extra_verts) + num_of_square_verts + current_added_verts
        
        
        
        current_added_verts += num_extra_verts
        
        #add the corresponding grey matter to the end of the vertex array.
    #get the vertices of the square and added gray matter and connect everything 
    #exclude the square bonds because they are already there
    
    pass
    
    
    
    
    
    
    
    
    
    
    
