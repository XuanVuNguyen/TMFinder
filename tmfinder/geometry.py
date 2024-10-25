import numpy as np

from typing import Optional, Union, List, Tuple, Any

from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from .biodata import HYDROPHOBIC

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GeometricResidue:
    def __init__(self, res: Residue):
        self.res = res
        if not self.res.has_id("CA"):
            raise ValueError("Residue does not have CA atom")
                
        self.coord = res["CA"].get_coord()
        
        self.geom = None
        self.in_membrane = None
        self.proj = None
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.res, name)
    
    def is_hydrophobic(self) -> bool:
        return self.res.resname in HYDROPHOBIC
    
    def __repr__(self):
        return f"<GeometricResidue {self.resname} resseq={self.res.get_id()[1]} geo={self.geom}>"
    
class GeometricModel:
    def __init__(self, model: Model):
        self.model = model
        self.residues = []
        
    def __getattr__(self, name: str):
        return getattr(self.model, name)
    
    def proj_geometric_residues(self, projection_ax: np.ndarray):
        for chain in self.model:
            ca_coords = get_CA_coords(chain)
            
            indexed_residues = {res.get_id()[1]: GeometricResidue(res) for res in chain if res.has_id("CA")}
            
            if ca_coords.shape[0] == 0:
                logger.debug("No CA atoms found in chain %s", chain.get_id())
                continue
            
            if ca_coords.shape[0] != len(indexed_residues):
                raise ValueError("Number of residues and CA atoms do not match")
            
            projected_ca_coords = np.matmul(ca_coords, projection_ax)
            
            for i, res in enumerate(indexed_residues.values()):
                res.proj = projected_ca_coords[i]
            
            # TODO: How to handle non continuous chain?
            for res_id in indexed_residues:
                pre_res = indexed_residues.get(res_id - 3)
                pos_res = indexed_residues.get(res_id + 3)
                res = indexed_residues[res_id]
                
                if pre_res is None or pos_res is None:
                    res.geom = "end"
                elif pre_res.proj < res.proj and res.proj < pos_res.proj:
                    res.geom = "straight"
                else:
                    res.geom = "turn"
                
                self.residues.append(res)

# class Plane:
#     def __init__(self, normal, pivot):
#         self.normal = normal
#         self.pivot = pivot

class ProteinSlicer:
    def __init__(
        self, 
        model: GeometricModel, 
        normal: np.ndarray, 
        width:Optional[float]=None, 
        pivot1: Optional[np.ndarray]=None, 
        pivot2: Optional[np.ndarray]=None
        ):
        self.model = model
        self.normal = normal / np.linalg.norm(normal)
        
        # project the residues onto the normal vector and identify their structural properties (straight, turn, or chain-end)
        self.model.proj_geometric_residues(self.normal)
        
        if pivot1 is not None:
            self.pivot1 = pivot1
            if pivot2 is not None:
                self.pivot2 = pivot2
                self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
            elif width is not None:
                self.pivot2 = pivot1 + width * self.normal
                self.width = width
            else:
                raise ValueError("When `pivot1` is provided, either `width` or `pivot2` must be provided")
            
        elif width is not None:
            self.width = width
            self.pivot_init()

        else:
            raise ValueError("Either `pivot1` or `width` must be provided")
        
        # self.in_membrane_residues = None
        self.fit_sliced_residues()
        
        
        # self.plane1 = Plane(self.normal, self.pivot1)
        # self.plane2 = Plane(self.normal, self.pivot2)
    def __repr__(self):
        return f"<ProteinSlice normal={self.normal} pivot1={self.pivot1} pivot2={self.pivot2} width={self.width}>"
    
    @property
    def proj_pivots(self):
        return np.dot(self.pivot1, self.normal), np.dot(self.pivot2, self.normal)
    
    @property
    def in_membrane_residues(self):
        return [res for res in self.model.residues if res.in_membrane]
    
    def get_normal_boundaries(self):
        proj_coords = np.array([res.proj for res in self.model.residues])
        lowest_res_id = np.argmin(proj_coords)
        lowest_res_coord = self.model.residues[lowest_res_id].coord
        lower_bound = np.dot(lowest_res_coord, self.normal)
        
        highest_res_id = np.argmax(proj_coords)
        highest_res_coord = self.model.residues[highest_res_id].coord
        higher_bound = np.dot(highest_res_coord, self.normal)
        
        return lower_bound, higher_bound
    
    def pivot_init(self):
        lower_bound, _ = self.get_normal_boundaries()
        
        # the first pivot lies lower than the lowest residues by 1A
        self.pivot1 = self.normal * (lower_bound -1)
        self.pivot2 = self.pivot1 + self.width*self.normal
    
    def fit_sliced_residues(self):
        self.residues = []
        proj_pivot1 = np.dot(self.pivot1, self.normal)
        proj_pivot2 = np.dot(self.pivot2, self.normal)
        for res in self.model.residues:
            proj_res = np.dot(res.coord, self.normal)
            if proj_pivot1 < proj_res < proj_pivot2 or proj_pivot2 < proj_res < proj_pivot1:
                res.in_membrane = True
            else:
                res.in_membrane = False
    
    def Q_score(self):
        return self.calc_Q_score(self.in_membrane_residues, self.width)
    
    def hydrophobic_factor(self):
        return self.calc_hydrophobic_factor(self.in_membrane_residues)
    
    def structure_factor(self):
        return self.calc_structure_factor(self.in_membrane_residues)
    
    def move_slice(self, step: float):
        self.pivot1 = self.pivot1 + step * self.normal
        self.pivot2 = self.pivot2 + step * self.normal
        self.fit_sliced_residues()
        
    def expand_slice(self, step: float):
        self.pivot1 = self.pivot1 - step * self.normal
        self.pivot2 = self.pivot2 + step * self.normal
        
        self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self.fit_sliced_residues()
        
    def move_pivot(self, step1: Optional[np.ndarray]=None, step2: Optional[np.ndarray]=None):
        if step1 is not None:
            self.pivot1 = self.pivot1 + step1*self.normal
        if step2 is not None:
            self.pivot2 = self.pivot2 + step2*self.normal
        self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self.fit_sliced_residues()
    
    def set_pivots(self, pivot1: Optional[np.ndarray]=None, pivot2: Optional[np.ndarray]=None):
        if pivot1 is not None:
            self.pivot1 = pivot1
        if pivot2 is not None:
            self.pivot2 = pivot2
        self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self.fit_sliced_residues()
            
    def n_crossing_segments(self):
        n_cross = 0
        for i in range(len(self.model.residues) - 1):
            res1 = self.model.residues[i]
            res2 = self.model.residues[i+1]
            
            if (res1.in_membrane and not res2.in_membrane) or (not res1.in_membrane and res2.in_membrane):
                n_cross += 1
        return n_cross
            
        
    # def rotate_normal
    
    @classmethod
    def calc_hydrophobic_factor(cls, residues: List[GeometricResidue]):
        return sum(res.sasa for res in residues if res.is_hydrophobic())
    
    @classmethod
    def calc_structure_factor(cls, residues: List[GeometricResidue]):
        straight_count = 0
        turn_count = 0
        end_count = 0
        
        for res in residues:
            if res.geom == "straight":
                straight_count += 1
            elif res.geom == "turn":
                turn_count += 1
            elif res.geom == "end":
                end_count += 1
        
        straight_rate = straight_count / len(residues)
        turn_rate = 1 - turn_count / len(residues)
        end_rate = 1 - end_count / len(residues)
        
        return straight_rate * turn_rate * end_rate
    
    @classmethod
    def calc_Q_score(cls, residues: List[GeometricResidue], width: float):
        return cls.calc_hydrophobic_factor(residues) * cls.calc_structure_factor(residues) / width
    
        

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