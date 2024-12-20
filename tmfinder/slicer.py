import numpy as np

from typing import Dict, Optional, Union, List, Tuple, Any

from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.SASA import ShrakeRupley

from .biodata import HYDROPHOBIC
from .geometry import (
    calc_symmetry_axis,
    fibonacci_sphere,
    get_CA_coords,
    get_atom_coords,
)
from .utils import init_logger

from tqdm import tqdm
from dataclasses import dataclass

logger = init_logger(__name__)


@dataclass
class ResidueProperties:
    """
    Represents the properties of a residue.

    Attributes:
        in_membrane (bool, optional): Indicates if the residue is between the two membrane planes.
        is_hydrophobic (bool, optional): Indicates if the residue is hydrophobic.
        geometry (str, optional): Whether this residue is "straight", "turn", or at the "end" of the strain.
        normal_projection (float, optional): The projection of the CA atom onto the normal vector of the slicer.
    """
    in_membrane: Optional[bool] = None
    is_hydrophobic: Optional[bool] = None
    geometry: Optional[str] = None
    normal_projection: Optional[float] = None


class ResiduePropertyMapper:
    """
    A class that maps a residue to its properties.
    Can be indexed like a dictionary with keys as strings or 
    an instance of Bio.PDB.Residue.Residue.
    
    Parameters:
        model (Bio.PDB.Model.Model): The model containing the residues.
        normal (np.ndarray): The normal vector defining the orientation of the membrane.
        pivot1 (Optional[np.ndarray]): Pivot vector defining a plane of the membrane. Default is None.
        pivot2 (Optional[np.ndarray]): Pivot vector defining a plane of the membrane. Default is None.
    """
    def __init__(
        self,
        model: Model,
        normal: np.ndarray,
        pivot1: Optional[np.ndarray] = None,
        pivot2: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.normal = normal
        self.pivot1 = pivot1
        self.pivot2 = pivot2
        self.properties = {}
        for chain in self.model:
            if len(chain) == 0:
                continue
            chain_id = chain.get_id()
            chain_properties = {}
            ca_coords = get_CA_coords(chain)
            proj_ca_coords = np.dot(ca_coords, normal)

            if proj_ca_coords.shape[0] != len(chain):
                raise ValueError("CA coordinates and chain length do not match.")
            for residue, proj_ca_coord in zip(chain, proj_ca_coords):
                _, res_id = self.get_res_id(residue)

                is_hydrophobic = residue.resname in HYDROPHOBIC
                chain_properties[res_id] = ResidueProperties(
                    in_membrane=None,
                    is_hydrophobic=is_hydrophobic,
                    geometry=None,
                    normal_projection=proj_ca_coord,
                )

            self.properties[chain_id] = chain_properties

        self._assign_geometry()
        if pivot1 is not None and pivot2 is not None:
            self.update_pivots(pivot1, pivot2)

    def __repr__(self):
        return f"<ResiduePropertyMapper normal={self.normal} pivot1={self.pivot1} pivot2={self.pivot2}>"

    def __getitem__(
        self, key: Union[Tuple[str, int], Residue, str]
    ) -> ResidueProperties:
        if isinstance(key, str):
            return self.properties[key]
        try:
            k = self.get_res_id(key) if isinstance(key, Residue) else key
            return self.properties[k[0]][k[1]]
        except KeyError as e:
            k = key.get_full_id() if isinstance(key, Residue) else key
            raise KeyError(f"Residue {k} not found in the model.") from e

    def values(self):
        """
        Returns a generator that yields all ResidueProperties objects.
        """
        for chain in self.properties.values():
            for res_prop in chain.values():
                yield res_prop

    def get_res_id(self, res: Residue) -> Tuple[str, int]:
        _, _, chain_id, res_id = res.get_full_id()
        return chain_id, res_id[1]

    def update_normal(self, normal: np.ndarray):
        """
        Updates the normal vector and recalculates the normal projection for all residues.

        Args:
            normal (np.ndarray): The new normal vector.
        """
        self.normal = normal
        proj_ca_coords = np.dot(get_CA_coords(self.model), normal)
        for res_prop, proj_ca_coord in zip(self.values(), proj_ca_coords):
            res_prop.normal_projection = proj_ca_coord

        self._assign_geometry()

    def _assign_geometry(self):
        """
        Assign the geometry property for all residues.
        """
        for chain in self.properties.values():
            for k, cur_res in chain.items():
                pre_res = chain.get(k - 3)
                pos_res = chain.get(k + 3)
                if pre_res is None or pos_res is None:
                    cur_res.geometry = "end"
                elif (
                    pre_res.normal_projection
                    < cur_res.normal_projection
                    < pos_res.normal_projection
                    or pos_res.normal_projection
                    < cur_res.normal_projection
                    < pre_res.normal_projection
                ):
                    cur_res.geometry = "straight"
                else:
                    cur_res.geometry = "turn"

    def update_pivots(self, pivot1: Optional[np.ndarray], pivot2: Optional[np.ndarray]):
        """
        Updates the pivot points and reassign the `in_membrane` property.

        Parameters:
            pivot1 (Optional[np.ndarray]): The new first pivot point.
            pivot2 (Optional[np.ndarray]): The new second pivot point.
        """
        if pivot1 is not None:
            self.pivot1 = pivot1
        if pivot2 is not None:
            self.pivot2 = pivot2
        proj_pivot1 = np.dot(self.pivot1, self.normal)
        proj_pivot2 = np.dot(self.pivot2, self.normal)
        for res_prop in self.values():
            res_prop.in_membrane = (
                proj_pivot1 < res_prop.normal_projection < proj_pivot2
                or proj_pivot2 < res_prop.normal_projection < proj_pivot1
            )

    def n_crossing_segments(self, max_gap: int = 10):
        """
        Returns the number of crossing segments in the model.
        A crossing segment is defined where among two consecutive residues, one is inside the membrane and the other is outside.
        
        Args:
            max_gap (int): The maximum gap between residues to consider them as consecutive. Default is 10.

        Returns:
            int: The number of crossing segments.
        """
        n_cross = 0
        for chain_id, chain_prop in self.properties.items():
            for i, resi in enumerate(chain_prop.keys()):
                pre_res = chain_prop.get(resi)
                gap = 1
                pos_res = chain_prop.get(resi + gap)

                # get the closest residue in case of gaps
                while pos_res is None and i + gap < len(chain_prop) and gap < max_gap:
                    gap += 1
                    pos_res = chain_prop.get(resi + gap)

                if pre_res is None or pos_res is None:
                    continue

                if pre_res.in_membrane is None or pos_res.in_membrane is None:
                    raise ValueError("In membrane property not defined.")
                if (pre_res.in_membrane and not pos_res.in_membrane) or (
                    not pre_res.in_membrane and pos_res.in_membrane
                ):
                    n_cross += 1
        return n_cross

    @property
    def projected_ca_coords(self):
        """
        Returns an array of the normal projections of all residues.

        Returns:
            np.ndarray: The array of normal projections.
        """
        return np.array([res_prop.normal_projection for res_prop in self.values()])


class ProteinSlicer:
    """
    `ProteinSlicer` slices a protein model into 1 angstrom slices based on a given normal vector.
    For each slice, a Q value is calculated from the hydrophobic factor and structure factor of the residues in that slice.
    A window of slices scans along the normal vector to find the interval with the highest window-averaged Q value.
    
    Attributes:
        model (Bio.PDB.Model.Model): protein model.
        width (int): The width of the slice window. Defaults to None.
        normal (Optional[np.ndarray]): The normal vector used for slicing. If not defined then the rotational symmetry axis is used as normal vector.
        n_normals_sample (int): The number of normal vectors to sample during the fitting process. Defaults to 5000.
        seed (Optional[int]): The seed value for random number generation. Defaults to None.
    
    Important methods:
        fit_normal(self): Find the normal vector with the highest Q value.
        fit_pivots(self): Given a fixed normal vector, a window scans along the normal vector to find the interval of width `self.width` that has the highest Q value.
        expand_pivots(self, step: float = 0.5): Expands the membrane planes as much as possible such that the total number of crossing segment does not change.
        move_pivots(self, step1: Optional[np.ndarray] = None, step2: Optional[np.ndarray] = None, update_residue_properties: bool = False): Moves the pivot points by the specified steps.
        set_pivots(self, pivot1: Optional[np.ndarray] = None, pivot2: Optional[np.ndarray] = None, update_residue_properties: bool = False): Sets the pivot points to the specified values.
        update_normal(self, normal: np.ndarray): Updates the normal vector and recalculates the slices based on the new normal.
        slice(self, residue_properties): Slices the protein model into 1 angstrom slices based on the current normal vector and calculate the Q values for the slices.
    """
    def __init__(
        self,
        model: Model,
        width: int,
        normal: Optional[np.ndarray] = None,
        n_normals_sample: int = 5000,
        seed: Optional[int] = None,
    ):
        self.sasa_calculator = ShrakeRupley()
        self.sasa_calculator.compute(model, level="R")
        self.model = model

        normal = normal if normal is not None else calc_symmetry_axis(model)
        self.normal_ = normal
        self.normal = normal / np.linalg.norm(normal)
        self.residue_properties = ResiduePropertyMapper(self.model, self.normal)

        self._anchors = None
        self._sliced_q_scores = None
        self.slice(self.residue_properties)

        self.width = width

        self.pivot1 = None
        self.pivot2 = None
        # self.init_pivot(self.normal)

        self.n_normals_sample = n_normals_sample
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        self.Q_score = sum(self._sliced_q_scores[: self.width]) / self.width

    def __repr__(self):
        return f"<ProteinSlice normal={self.normal} pivot1={self.pivot1} pivot2={self.pivot2} width={self.width}>"

    # def init_pivot(self, normal: np.array):
    #     """
    #     Initiate the pivot points as the first two delimiter points of the slices.
    #     """
    #     self.pivot1 = self._anchors[0] * normal
    #     self.pivot2 = self._anchors[self.width] * normal

    def fit_pivots(self, update_residue_properties: bool = False):
        """
        Given a fixed normal vector, a window scans along the normal vector to find the interval of width `self.width` that has the highest Q value.
        
        Args:
            update_residue_properties (bool): Whether to update the residue properties after setting the pivots. Defaults to False
        """
        width_sum_q_scores = []
        cur_sum = 0
        for i in range(len(self._sliced_q_scores) - self.width + 1):
            if i == 0:
                cur_sum = sum(self._sliced_q_scores[: self.width])
            else:
                cur_sum = (
                    cur_sum
                    - self._sliced_q_scores[i - 1]
                    + self._sliced_q_scores[i + self.width - 1]
                )
            width_sum_q_scores.append(cur_sum)

        best_q_score_id = np.argmax(width_sum_q_scores)
        best_pivot1 = self._anchors[best_q_score_id] * self.normal
        best_pivot2 = self._anchors[best_q_score_id + self.width] * self.normal

        self.Q_score = width_sum_q_scores[best_q_score_id] / self.width

        self.set_pivots(best_pivot1, best_pivot2, update_residue_properties)

        # We don't call self.residue_properties.update_pivots() here since pivots only affect in_membrane property,
        # which is only used when we expand the pivots.

    def fit_normal(self):
        """
        Exhaustively search for the normal vector that gives the highest Q value.
        """
        sample_normals = fibonacci_sphere(self.n_normals_sample)
        sample_normals = np.concatenate(
            [sample_normals, self.normal.reshape(1, -1)], axis=0
        )
        best_args = None
        best_q_score = None
        for normal in tqdm(sample_normals, desc="Exhaustive search for the normal vector"):
            self.update_normal(normal)
            self.fit_pivots()
            if best_q_score is None or self.Q_score > best_q_score:
                best_q_score = self.Q_score
                best_args = (normal, self.pivot1, self.pivot2)

        self.update_normal(best_args[0])

        self.set_pivots(best_args[1], best_args[2],
                        update_residue_properties=True)

        self.Q_score = best_q_score

    def expand_pivots(self, step: float = 0.5):
        """
        Expands the membrane planes as much as possible such that the total number of crossing segment does not change.
        
        Args:
            step (float): The step size to expand the pivots. Defaults to 0.5.
        """
        atom_coords = get_atom_coords(self.model)
        proj_coords = np.dot(atom_coords, self.normal)
        atom_lowest = np.min(proj_coords)
        atom_highest = np.max(proj_coords)

        self.residue_properties.update_pivots(self.pivot1, self.pivot2)
        cur_n_segments = self.residue_properties.n_crossing_segments()
        while self.residue_properties.n_crossing_segments() == cur_n_segments:
            self.move_pivots(step1=-step, update_residue_properties=True)
            if np.dot(self.pivot1, self.normal) < atom_lowest:
                logger.warning("Pivot1 reached the lowest atom. Stopping.")
                break
        self.move_pivots(step1=step, update_residue_properties=True)

        cur_n_segments = self.residue_properties.n_crossing_segments()
        while self.residue_properties.n_crossing_segments() == cur_n_segments:
            self.move_pivots(step2=step, update_residue_properties=True)
            if np.dot(self.pivot2, self.normal) > atom_highest:
                logger.warning("Pivot2 reached the highest atom. Stopping.")
                break
        self.move_pivots(step2=-step, update_residue_properties=True)

    def move_pivots(
        self,
        step1: Optional[np.ndarray] = None,
        step2: Optional[np.ndarray] = None,
        update_residue_properties: bool = False,
    ):
        if step1 is not None:
            self.pivot1 = self.pivot1 + step1 * self.normal
        if step2 is not None:
            self.pivot2 = self.pivot2 + step2 * self.normal
        if update_residue_properties:
            self.residue_properties.update_pivots(self.pivot1, self.pivot2)

    def set_pivots(
        self,
        pivot1: Optional[np.ndarray] = None,
        pivot2: Optional[np.ndarray] = None,
        update_residue_properties: bool = False,
    ):
        if pivot1 is not None:
            self.pivot1 = pivot1
        if pivot2 is not None:
            self.pivot2 = pivot2
        if update_residue_properties:
            self.residue_properties.update_pivots(self.pivot1, self.pivot2)

    def update_normal(self, normal: np.ndarray):
        self.normal = normal / np.linalg.norm(normal)
        self.residue_properties.update_normal(normal)
        self.slice(self.residue_properties)
        # self.init_pivot(self.normal)

    def slice(self, residue_properties):
        """
        Call everytime a new normal is defined.
        """
        proj_ca_coords = residue_properties.projected_ca_coords
        sorted_res_ids = sorted(
            range(len(proj_ca_coords)), key=lambda i: proj_ca_coords[i]
        )
        lower_bound = np.min(proj_ca_coords)
        higher_bound = np.max(proj_ca_coords)
        anchor_points = [lower_bound]
        while anchor_points[-1] < higher_bound:
            anchor_points.append(anchor_points[-1] + 1.)
        sliced_res_ids = []
        cur_res_group = []
        cur_anchor_idx = 0
        for res_ids in sorted_res_ids:
            proj_res = proj_ca_coords[res_ids]
            if not cur_res_group:
                cur_res_group.append(res_ids)
                continue
            if (
                proj_res >= anchor_points[cur_anchor_idx]
                and proj_res < anchor_points[cur_anchor_idx + 1]
            ):
                cur_res_group.append(res_ids)
            else:
                sliced_res_ids.append(cur_res_group)
                cur_res_group = [res_ids]
                cur_anchor_idx += 1
                while proj_res >= anchor_points[cur_anchor_idx + 1]:
                    sliced_res_ids.append([])
                    cur_anchor_idx += 1

        sliced_res_ids.append(cur_res_group)

        _sliced_q_scores = []
        residues = list(self.model.get_residues())
        for res_id_group in sliced_res_ids:
            sliced_res = [residues[res_id] for res_id in res_id_group]
            _sliced_q_scores.append(
                calc_Q_score(sliced_res, residue_properties, 1)
            )

        self._anchors = anchor_points
        self._sliced_q_scores = _sliced_q_scores

        if len(self._anchors) != len(self._sliced_q_scores) + 1:
            raise ValueError(
                f"Number of anchors and slices do not match: {len(self._anchors)} != {len(self._sliced_q_scores)+1}"
            )



def calc_hydrophobic_factor(
    residues: List[Residue], 
    residue_properties: ResiduePropertyMapper
):
    hydrophobic_sum = sum(
        res.sasa for res in residues if residue_properties[res].is_hydrophobic
    )
    all_sum = sum(res.sasa for res in residues)
    return hydrophobic_sum / (all_sum + 1e-6)


def calc_structure_factor(
    residues: List[Residue], 
    residue_properties: ResiduePropertyMapper
):
    if not residues:
        return 0
    straight_count = 0
    turn_count = 0
    end_count = 0

    for res in residues:
        if residue_properties[res].geometry == "straight":
            straight_count += 1
        elif residue_properties[res].geometry == "turn":
            turn_count += 1
        elif residue_properties[res].geometry == "end":
            end_count += 1

    straight_rate = straight_count / len(residues)
    turn_rate = 1 - turn_count / len(residues)
    end_rate = 1 - end_count / len(residues)

    return straight_rate * turn_rate * end_rate


def calc_Q_score(
    residues: List[Residue],
    residue_properties: ResiduePropertyMapper,
    width: float,
):
    """
    Calculate the Q score for a list of residues.
    
    Args:
        residues (List[Bio.PDB.Residue.Residue]): The list of residues.
        residue_properties (ResiduePropertyMapper): The mapped residue properties.
    """
    return (
        calc_hydrophobic_factor(residues, residue_properties)
        * calc_structure_factor(residues, residue_properties)
        / width
    )
