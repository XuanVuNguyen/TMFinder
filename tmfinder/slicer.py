import numpy as np

from typing import Optional, Union, List, Tuple, Any

from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from .biodata import HYDROPHOBIC
from .geometry import GeometricModel, GeometricResidue, calc_symmetry_axis, sphere_sample, get_CA_coords

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProteinSlicer:
    def __init__(
        self, 
        model: Model, 
        width:Optional[float]=None, 
        pivot1: Optional[np.ndarray]=None, 
        pivot2: Optional[np.ndarray]=None,
        normal: Optional[np.ndarray]=None, 
        step: float=0.5,
        n_normals_sample: int=10000,
        seed: Optional[int]=None
        ):
        self.model_ = model
        self.model = GeometricModel(model, normal)
        normal = normal if normal is not None else calc_symmetry_axis(model)
        self.normal_ = normal
        self.normal = normal / np.linalg.norm(normal)
        
        self._projected_pivot1 = None
        self._projected_pivot2 = None
        self._CA_coords = get_CA_coords(self.model)
        self._projected_CA_coords = np.dot(self._CA_coords, self.normal)

        self._lower_bound, self._higher_bound = self.get_normal_boundaries()
        if pivot1 is not None:
            self.pivot1 = pivot1
            if pivot2 is not None:
                self.pivot2 = pivot2
            elif width is not None:
                self.pivot2 = pivot1 + width * self.normal
            else:
                raise ValueError("When `pivot1` is provided, either `width` or `pivot2` must be provided")
            
            self._projected_pivot1 = np.dot(self.pivot1, self.normal)
            self._projected_pivot2 = np.dot(self.pivot2, self.normal)
            
        elif width is not None:
            self.init_pivot(self._lower_bound, width)

        else:
            raise ValueError("Either `pivot1` or `width` must be provided")
        self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self.step = step
        self.n_normals_sample = n_normals_sample
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)        
        # self.in_membrane_residues = None
                
        
        self.fit_sliced_residues()
        
        
    def __repr__(self):
        return f"<ProteinSlice normal={self.normal} pivot1={self.pivot1} pivot2={self.pivot2} width={self.width}>"
    
    @property
    def cur_width(self):
        return np.dot(self.pivot2 - self.pivot1, self.normal)
    
    # @property
    # def proj_pivots(self):
    #     return np.dot(self.pivot1, self.normal), np.dot(self.pivot2, self.normal)
    
    @property
    def in_membrane_residues(self):
        return [res for res in self.model.residues if res.in_membrane]
    
    def get_normal_boundaries(self):
        # CA_coords = get_CA_coords(self.model)
        # proj_coords = np.dot(CA_coords, self.normal)
        lower_bound = np.min(self._projected_CA_coords)
        higher_bound = np.max(self._projected_CA_coords)
        # lowest_res_id = np.argmin(proj_coords)
        # lowest_res_coord = self.model.residues[lowest_res_id].coord
        # lower_bound = np.dot(lowest_res_coord, self.normal)
        
        # highest_res_id = np.argmax(proj_coords)
        # highest_res_coord = self.model.residues[highest_res_id].coord
        # higher_bound = np.dot(highest_res_coord, self.normal)
        
        return lower_bound, higher_bound
    
    def init_pivot(self, lower_bound: float, width: float):       
        # the first pivot lies lower than the lowest residues by 1A
        self.pivot1 = self.normal * (lower_bound -1)
        self.pivot2 = self.pivot1 + width*self.normal
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
    
    def fit_sliced_residues(self):
        proj_pivot1 = self._projected_pivot1
        proj_pivot2 = self._projected_pivot2
        for res, proj_coords in zip(self.model.residues, self._projected_CA_coords):
            if proj_pivot1 < proj_coords < proj_pivot2 or proj_pivot2 < proj_coords < proj_pivot1:
                res.in_membrane = True
            else:
                res.in_membrane = False
    
    def fit_pivot(self):
        max_q_score = None
        best_pivot1 = None
        best_pivot2 = None
        while self._projected_pivot2 <= self._higher_bound:
            if max_q_score is None or self.Q_score > max_q_score:
                max_q_score = self.Q_score
                best_pivot1 = self.pivot1
                best_pivot2 = self.pivot2
            self.move_slice(self.step)
        
        self.set_pivots(best_pivot1, best_pivot2)
    
    def fit_normal(self):
        sample_normals = sphere_sample(self.n_normals_sample,
                                       theta_range=(0, np.pi/2),
        )
        sample_normals = np.concatenate([sample_normals, self.normal.reshape(1, -1)], axis=0)
        best_args = None
        best_q_score = None
        for normal in tqdm(sample_normals):
            self.set_normal(normal)

            self.fit_pivot()
            if best_q_score is None or self.Q_score > best_q_score:
                best_q_score = self.Q_score
                best_args = (normal, self.pivot1, self.pivot2)
        
        self.set_normal(best_args[0])
        self.set_pivots(best_args[1], best_args[2])
                
    @property
    def Q_score(self):
        return self.calc_Q_score(self.in_membrane_residues, self.cur_width)
    
    @property
    def hydrophobic_factor(self):
        return self.calc_hydrophobic_factor(self.in_membrane_residues)
    
    @property
    def structure_factor(self):
        return self.calc_structure_factor(self.in_membrane_residues)
    
    def move_slice(self, step: float):
        self.pivot1 = self.pivot1 + step * self.normal
        self.pivot2 = self.pivot2 + step * self.normal
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        
        self.fit_sliced_residues()
        
    def expand_slice(self, step: float):
        self.pivot1 = self.pivot1 - step * self.normal
        self.pivot2 = self.pivot2 + step * self.normal
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        # self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self.fit_sliced_residues()
        
    def move_pivot(self, step1: Optional[np.ndarray]=None, step2: Optional[np.ndarray]=None):
        if step1 is not None:
            self.pivot1 = self.pivot1 + step1*self.normal
        if step2 is not None:
            self.pivot2 = self.pivot2 + step2*self.normal
        # self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        self.fit_sliced_residues()
    
    def set_pivots(self, pivot1: Optional[np.ndarray]=None, pivot2: Optional[np.ndarray]=None):
        if pivot1 is not None:
            self.pivot1 = pivot1
        if pivot2 is not None:
            self.pivot2 = pivot2
        # self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        self.fit_sliced_residues()
    
    def set_normal(self, normal: np.ndarray):
        self.normal = normal / np.linalg.norm(normal)
        self._projected_CA_coords = np.dot(self._CA_coords, self.normal)
        self._lower_bound, self._higher_bound = self.get_normal_boundaries()
        self.init_pivot(self._lower_bound, self.width)
        self.fit_sliced_residues()
        
        #TODO: can be optimized more
            
    def n_crossing_segments(self):
        n_cross = 0
        for i in range(len(self.model.residues) - 1):
            res1 = self.model.residues[i]
            res2 = self.model.residues[i+1]
            
            if (res1.in_membrane and not res2.in_membrane) or (not res1.in_membrane and res2.in_membrane):
                n_cross += 1
        return n_cross
    
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