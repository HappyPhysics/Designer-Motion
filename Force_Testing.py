# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 06:36:36 2017

@author: Salem
This script takes the resulting mesh and spring constants from the design process and tests it by applying forces and checking if the 
desired mode comes out. 

The energy used here does not assume linear displacements so we only expect agreement for small enough forces. We will see what happens
when the force becomes too large. 
"""
import numpy as np
import scipy.optimize as op
import numpy.random as npr
import numpy.linalg as la

import Many_Triangles as MT
import LatticeMaking as LM




import importlib
importlib.reload(LM)
importlib.reload(MT)



#====================================================================================================================================
# energy of the configuration given by vertices
#====================================================================================================================================
def elastic_energy(verts, edges, spring_constants, sqr_reference_lengths, force):
    '''
    computes the elastic energy of the mesh given by vertices and edges. 
    The sqr_reference_lengths, which are the preferred lengths of the bonds squared
    represent the zero energy condition and spring_constants represent 
    the cost of stretching or compressing a bond. 
    verts are flattened, vertices are the reshaped version
    
    force is flattened
    '''
    energy = 0
    
    #vertices is to unflatten verts
    vertices = verts.reshape((verts.size//2, 2))
    
    for index, edge in enumerate(edges):
        #print(edge)
        #print(vertices[edge[1]])
        separation = vertices[edge[1]] - vertices[edge[0]]
        #print(separation)
        #energy from each bond
        energy += (spring_constants[index]**2 / 8)*(np.sum(separation**2) - sqr_reference_lengths[index])**2
    
    #print(energy)    
     #add the energy from the applied external force and return   
    return energy - np.dot(force,verts)
#====================================================================================================================================
    
def apply_force(vertices, edges, spring_constants, force):
    '''
    applies the force to the mesh given by vertices and edges. The reference lengths
    are given by the vertices. The force results in the stretching of the bonds whos 
    rigidity is given by spring_constants.
    
    This is going to return the displacements. 
    
    '''
    
     # project out the euclidean  transforms
    euclid_transforms = get_rigid_transformations(vertices)
    euclid_projector = get_complement_space(euclid_transforms)
    
    
    projected_force = np.dot(euclid_projector, force.flatten())
    #print(projected_force)
    
    # the distance between the neighbors squared. 
    sqr_reference_lengths =  np.sum((vertices[edges[:,1]] - vertices[edges[:,0]])**2, axis=1)

    #print(sqr_reference_lengths)
    res = op.minimize(elastic_energy, vertices.flatten(), method='BFGS', args=(edges, spring_constants, 
            sqr_reference_lengths, projected_force), options={ 'disp': False})
    
    new_verts = res.x.reshape(vertices.shape)
    return new_verts - np.average(new_verts, axis=0)
    

def test_wave_changer(result=None):

    if result is None: result = MT.wave_changer()
    
    vertices = result[1][0].copy()
    edges = result[1][1].copy()
    
    num_of_verts = vertices.shape[0]
    
    force = LM.normalizeVec(npr.rand(num_of_verts*2))
    
    force[:4] = [0, 1, 0, -1]
    
    k = result[2].copy()
    
    
    dyn_mat = LM.makeDynamicalMat(verts=vertices, edgeArray=edges, springK=k)
    
    lowestEigVector = LM.normalizeVec(la.eigh(dyn_mat)[1][:,3])
    
    print("force along lowest: ", np.dot(lowestEigVector, force))
    
    force = force.reshape((num_of_verts, 2))
    
    new_verts = apply_force(vertices, edges, k, force)
    
    disps = (new_verts - vertices).flatten()
    
    
     # project out the euclidean  transforms
    euclid_transforms = LM.get_rigid_transformations(vertices)
    euclid_projector = LM.get_complement_space(euclid_transforms)
    
    return  LM.normalizeVec(np.dot(euclid_projector, disps))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    