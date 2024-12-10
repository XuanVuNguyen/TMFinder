import numpy as np

from typing import Optional, Union, List, Tuple, Any

from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from .biodata import HYDROPHOBIC
from .geometry import (
    GeometricModel, 
    GeometricResidue, 
    calc_symmetry_axis, 
    sphere_sample, 
    fibonacci_sphere, 
    get_CA_coords,
    get_atom_coords
)
from .utils import init_logger

import logging
from tqdm import tqdm

logger = init_logger(__name__)

class ProteinSlicer:
    def __init__(
        self, 
        model: Model, 
        width:Optional[float]=None, 
        pivot1: Optional[np.ndarray]=None, 
        pivot2: Optional[np.ndarray]=None,
        normal: Optional[np.ndarray]=None, 
        step: float=0.5,
        n_normals_sample: int=5000,
        seed: Optional[int]=None
        ):
        self.model_ = model
        self.model = GeometricModel(model, normal)
        normal = self.model.normal
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
        # self.anchors = None
        # self.sliced_q_scores = None
        self.slice()
        
        # self._fit_sliced_residues()
        
        
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
        # self.pivot1 = self.normal * lower_bound
        # self.pivot2 = self.pivot1 + width*self.normal
        # self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        # self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        
        # self._fit_sliced_residues()
        self.pivot1 = self.anchors[0]
        self.pivot2 = self.anchors[self.width]
    
    def _fit_sliced_residues(self):
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
        # sample_normals = sphere_sample(self.n_normals_sample,
        #                                theta_range=(0, np.pi/2),
        # )
        # sample_normals = np.concatenate([sample_normals, self.normal.reshape(1, -1)], axis=0)
        sample_normals = fibonacci_sphere(self.n_normals_sample)
        sample_normals = np.concatenate([sample_normals, self.normal.reshape(1, -1)], axis=0)
        best_args = None
        best_q_score = None
        for normal in tqdm(sample_normals):
            self.set_normal(normal)

            self.fit_pivot()
            if best_q_score is None or self.Q_score > best_q_score:
                best_q_score = self.Q_score
                best_args = (normal, self.pivot1, self.pivot2)
        
        self.set_normal(best_args[0], init_pivot=False)
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
        
        self._fit_sliced_residues()
        
    def expand_pivots(self, step: float=0.5):
        atom_coords = get_atom_coords(self.model)
        proj_coords = np.dot(atom_coords, self.normal)
        atom_lowest = np.min(proj_coords)
        atom_highest = np.max(proj_coords)
        
        cur_n_segments = self.n_crossing_segments()
        while self.n_crossing_segments() == cur_n_segments:
            self.move_pivot(step1=-step)
            if self._projected_pivot1 < atom_lowest:
                logger.warning("Pivot1 reached the lowest atom. Stopping.")
                break
        self.move_pivot(step1=step)
        
        cur_n_segments = self.n_crossing_segments()
        while self.n_crossing_segments() == cur_n_segments:
            self.move_pivot(step2=step)
            if self._projected_pivot2 > atom_highest:
                logger.warning("Pivot2 reached the highest atom. Stopping.")
                break
        self.move_pivot(step2=-step)
        
        self._fit_sliced_residues()
        
    def move_pivot(self, step1: Optional[np.ndarray]=None, step2: Optional[np.ndarray]=None):
        if step1 is not None:
            self.pivot1 = self.pivot1 + step1*self.normal
        if step2 is not None:
            self.pivot2 = self.pivot2 + step2*self.normal
        # self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        self._fit_sliced_residues()
    
    def set_pivots(self, pivot1: Optional[np.ndarray]=None, pivot2: Optional[np.ndarray]=None):
        if pivot1 is not None:
            self.pivot1 = pivot1
        if pivot2 is not None:
            self.pivot2 = pivot2
        # self.width = np.dot(self.pivot2 - self.pivot1, self.normal)
        self._projected_pivot1 = np.dot(self.pivot1, self.normal)
        self._projected_pivot2 = np.dot(self.pivot2, self.normal)
        self._fit_sliced_residues()
    
    def set_normal(self, normal: np.ndarray, init_pivot: bool=True):
        self.normal = normal / np.linalg.norm(normal)
        self._projected_CA_coords = np.dot(self._CA_coords, self.normal)
        self._lower_bound, self._higher_bound = self.get_normal_boundaries()
        self.slice()
        
        if init_pivot:
            self.init_pivot(self._lower_bound, self.width)
            self._fit_sliced_residues()

            
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
        hydrophobic_sum = sum(res.sasa for res in residues if res.is_hydrophobic())
        all_sum = sum(res.sasa for res in residues)
        return hydrophobic_sum / all_sum
    
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
    
    
    def slice(self):
        """
        Call everytime a new normal is defined.
        """
        sorted_res_ids = sorted(range(len(self._projected_CA_coords)), key=lambda i: self._projected_CA_coords[i])
        
        anchor_points = [self._lower_bound]
        while anchor_points[-1] < self._higher_bound:
            anchor_points.append(anchor_points[-1] + 1)
        sliced_res_ids = []
        cur_res_group = []
        cur_anchor_idx = 0
        for res_ids in sorted_res_ids:
            if not cur_res_group:
                cur_res_group.append(res_ids)
                continue
            if self._projected_CA_coords[res_ids] >= anchor_points[cur_anchor_idx] and self._projected_CA_coords[res_ids] < anchor_points[cur_anchor_idx + 1]:
                cur_res_group.append(res_ids)
            else:
                sliced_res_ids.append(cur_res_group)
                cur_res_group = [res_ids]
                cur_anchor_idx += 1
        
        sliced_q_scores = []
        for res_id_group in sliced_res_ids:
            sliced_res = [self.model.residues[res_id] for res_id in res_id_group]
            sliced_q_scores.append(self.calc_Q_score(sliced_res, 1))
        
        self.anchors = anchor_points
        self.sliced_q_scores = sliced_q_scores