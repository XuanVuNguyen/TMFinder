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

# class GeometricResidue:
#     def __init__(self, res: Residue):
#         self.res = res
#         if not self.res.has_id("CA"):
#             raise ValueError("Residue does not have CA atom")
                
#         self.coord = res["CA"].get_coord()
        
#         self.geom = None
#         self.in_membrane = None
#         self._proj = None
    
#     def __getattr__(self, name: str) -> Any:
#         return getattr(self.res, name)
    
#     def is_hydrophobic(self) -> bool:
#         return self.res.resname in HYDROPHOBIC
    
#     # def project(self, vector: np.ndarray) -> float:
#     #     proj = np.dot(self.coord, vector)
#     #     self._proj = proj
#     #     return proj
    
#     def __repr__(self):
#         return f"<GeometricResidue {self.resname} resseq={self.res.get_id()[1]} geo={self.geom}>"
    
# class GeometricChain:
#     def __init__(self, chain: Chain):
#         self.chain = chain
#         self.residues = [GeometricResidue(res) for res in chain if res.has_id("CA")]
    
# class GeometricModel:
#     def __init__(self, model: Model, normal:Optional[np.ndarray]=None):
#         self.model = model
        
#         self.normal = normal if normal is not None else calc_symmetry_axis(model)
        
#         # project the residues onto the normal vector and identify their structural properties (straight, turn, or chain-end)
#         self.residues = self.init_geometric_residues(self.normal)
                
#     def __getattr__(self, name: str):
#         return getattr(self.model, name)
    
#     def init_geometric_residues(self, projection_ax: np.ndarray):
#         residues = []
#         for chain in self.model:
#             ca_coords = get_CA_coords(chain)
            
#             indexed_residues = {res.get_id()[1]: GeometricResidue(res) for res in chain if res.has_id("CA")}
            
#             if ca_coords.shape[0] == 0:
#                 logger.debug("No CA atoms found in chain %s", chain.get_id())
#                 continue
            
#             if ca_coords.shape[0] != len(indexed_residues):
#                 raise ValueError("Number of residues and CA atoms do not match")
            
#             projected_ca_coords = np.matmul(ca_coords, projection_ax)
            
#             for i, res in enumerate(indexed_residues.values()):
#                 res._proj = projected_ca_coords[i]
            
#             # TODO: How to handle non continuous chain?
#             for res_id in indexed_residues:
#                 pre_res = indexed_residues.get(res_id - 3)
#                 pos_res = indexed_residues.get(res_id + 3)
#                 res = indexed_residues[res_id]
                
#                 if pre_res is None or pos_res is None:
#                     res.geom = "end"
#                 elif pre_res._proj < res._proj and res._proj < pos_res._proj:
#                     res.geom = "straight"
#                 else:
#                     res.geom = "turn"
                
#                 residues.append(res)
#         return residues
    
        

def sphere_sample(n_points: int, radius: float=1, 
                  phi_range: Tuple[float, float]=(0, 2*np.pi),
                  theta_range: Tuple[float, float]=(0, np.pi),
                  seed: Optional[int]=None):
    if seed is not None:
        np.random.seed(seed)
    phi = np.random.uniform(*phi_range, n_points)
    theta = np.random.uniform(*theta_range, n_points)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return np.array([x, y, z]).T

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