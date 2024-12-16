import numpy as np

from typing import Optional, Union, List, Tuple, Any

from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from .biodata import HYDROPHOBIC

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fibonacci_sphere(samples=1000): 
    points = [] 
    phi = np.pi * (3. - np.sqrt(5.)) # golden angle in radians 
    for i in range(samples): 
        y = 1 - (i / float(samples - 1)) * 2 # y goes from 1 to -1 
        radius = np.sqrt(1 - y * y) # radius at y 
        theta = phi * i # golden angle increment 
        x = np.cos(theta) * radius 
        z = np.sin(theta) * radius 
        points.append([x, y, z])
        
    return np.array(points)
        
def calc_center_of_mass(obj: Union[Model, Chain], weight_by_mass=False, CA_only=True):
    mass = 0
    center = np.zeros(3)
    for atom in obj.get_atoms():
        if CA_only and atom.name != "CA":
            continue
        if weight_by_mass:
            mass += atom.mass
            center += atom.mass * atom.get_coord()
        else:
            mass += 1
            center += atom.get_coord()
    return center / mass

def get_CA_coords(obj: Union[Model, Chain]):
    ca_coords = []
    for residue in obj.get_residues():
        if residue.has_id("CA"):
            ca_coords.append(residue["CA"].get_coord())
    return np.array(ca_coords)

def get_atom_coords(obj: Union[Model, Chain]):
    atom_coords = []
    for residue in obj.get_residues():
        for atom in residue:
            atom_coords.append(atom.get_coord())
    return np.array(atom_coords)

def calc_symmetry_axis(model):
    calpha_coords = get_CA_coords(model)
    calpha_coords = np.array(calpha_coords)
    centroid = np.mean(calpha_coords, axis=0)
    calpha_coords -= centroid
    
    I_xx = np.sum(calpha_coords[:, 1]**2 + calpha_coords[:, 2]**2)
    I_yy = np.sum(calpha_coords[:, 0]**2 + calpha_coords[:, 2]**2)
    I_zz = np.sum(calpha_coords[:, 0]**2 + calpha_coords[:, 1]**2)
    I_xy = np.sum(-calpha_coords[:, 0] * calpha_coords[:, 1])
    I_xz = np.sum(-calpha_coords[:, 0] * calpha_coords[:, 2])
    I_yz = np.sum(-calpha_coords[:, 1] * calpha_coords[:, 2])
    
    I = np.array([[I_xx, I_xy, I_xz],
                [I_xy, I_yy, I_yz],
                [I_xz, I_yz, I_zz]])
    
    eigvals, eigvecs = np.linalg.eig(I)
    
    symmetry_axis = eigvecs[:, np.argmax(eigvals)]
    return symmetry_axis / np.linalg.norm(symmetry_axis)