# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 08:07:09 2017

@author: Salem and Wife
"""

import numpy as np
from numpy import random as npr
import numpy.linalg as la
import scipy as sp
import scipy.optimize as op
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import LatticeMaking
import importlib
importlib.reload(LatticeMaking)

from LatticeMaking import makeDynamicalMat
from numpy.linalg import inv

EQUI_TRIANGLE = np.array([[0, 0], [0, 1], [np.sin(np.pi/3), 0.5]])

triangLines = np.array([[1, 2], [1, 3], [2, 3]]) - 1
grayLines = np.array([[4, 5], [4, 6], [5, 6]]) - 1
crossLines = np.array([[1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 
    4], [3, 5], [3, 6]]) - 1



EDGES = np.vstack((triangLines, grayLines, crossLines))

DESIRED_DYN_MAT  = np.array([[100, 5, 5], [5, 200, 5], [5,5, 300]])

def find_gray_matter():
    
    init_gray_matter = np.random.rand(6)
    
    res = op.root(effective_dynamical_matrix, init_gray_matter,method="anderson")
    
    return res



def effective_dynamical_matrix(gray_matter): 
    edges = EDGES
    num_of_edges = edges.size//2
    
    springK = np.ones((num_of_edges))
    
    verts = np.vstack((EQUI_TRIANGLE, gray_matter.reshape((3,2))))
    
    dyn_mat =  makeDynamicalMat(edgeArray=edges, verts=verts, springK=springK)[3:, 3:]
    #print(dyn_mat[:3, :3], "\n\n")
    #print(dyn_mat[:3, 3:],"\n\n")
    effective_dyn_mat = dyn_mat[:3, :3] - np.dot(dyn_mat[:3, 3:], np.dot(inv(dyn_mat[3:, 3:]), np.transpose(dyn_mat[:3, 3:])))
    
    return [effective_dyn_mat[0, 1], effective_dyn_mat[0, 2], effective_dyn_mat[1, 2]]
    
    # indices to select upper triangular elements 
    #uptri = np.triu_indices(3)
    
    #return (effective_dyn_mat)[uptri]/ (la.norm((effective_dyn_mat)[uptri]))


# divide the 5 dimensional sphere into bins.
# evaluate the effective matrix millions of times
    #add each to the corresponding bins.
# check the density of bins. 


def generate_plot(num_of_points):
    x = []
    y = []
    z = []
    for i in range(num_of_points):
        gray_matter = npr.rand(6)
        
        Deff = effective_dynamical_matrix(gray_matter)
        x.append(Deff[0])
        y.append(Deff[1])
        z.append(Deff[2])
    
    plotPoints(x, y, z)


def plotPoints(x, y, z): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x =[1,2,3,4,5,6,7,8,9,10]
    y =[5,6,2,3,13,4,1,2,4,8]
    z =[2,3,3,3,5,7,9,11,9,10]



    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

plt.show()





