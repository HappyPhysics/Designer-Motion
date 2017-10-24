# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:39:08 2017

@author: Salem

This script starts with a square lattice and then divides it into a bunch of squares.
Then for each square it calls Little-Square and implements the addition of gray matter to
each square separately. Before calling Little-Square a change of frame must be implemented. 

Methods: 
    
    initialize_lattice():
    
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


#===========================================================================================================================================
# Makes a square lattice and gives a list of squares 
#===========================================================================================================================================  


LATTICE_WIDTH = 10 #by 10 vertices
SQUARE_LATTICE = squareLattice(LATTICE_WIDTH, is_with_diagonals=False) #contains vertices and edges as tuple

# Pick out the squares one by one. 
# change thier frame of reference to the canonical one (from little square).
# call little square to add the gray matter. 
# change back to original frame of reference.
# add all of the squares together to get the entire final lattice. 
# test that lowest energy modes of the lattice satisfy the desired motion.
    

def get_square(indx):
    return
    # start with the index of the lower left corner of the square (there are LATTICE_WIDTH-1 by LATTICE_WIDTH-1 of them)    
    # find the indices of the other three
    # return thier positions and displacements to change frame. 

#===========================================================================================================================================
# puts the square displacements in canonical form
#===========================================================================================================================================     
def change_square_frame(squareDisps):
    return
#subtract the displacement of the first vertex from all the displacements. 

# find a rotation that will put the second displacement along the y axis.
    rot_matrix = rotation_matrix(angle_between(second_disp, [0, 1]))
# apply this rotation to all the displacements. 

# return the displacemetns in canonical form.     
    
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
