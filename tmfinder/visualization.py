from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import numpy as np

def add_atom_plane(model: Model, 
               com: np.ndarray, 
               normal: np.ndarray, 
               pivot: np.ndarray, 
               edge:int,
               element:str,
               space = 3):
    def _get_last_serial_number(model: Model):
        res = None
        for atom in model.get_atoms():
            res = atom.get_serial_number()
        return res
    
    normal = normal / np.linalg.norm(normal)
    proj_com = np.dot(com, normal)
    proj_pivot = np.dot(pivot, normal)
    
    # The center of the plane is the projection of COM onto the plane
    center = com + (proj_pivot-proj_com) * normal
    
    v_axis = center - normal*proj_pivot
    v_axis /= np.linalg.norm(v_axis)
    
    h_axis = np.cross(v_axis, normal)
    h_axis /= np.linalg.norm(h_axis)
    
    atoms = []
    serial_num = _get_last_serial_number(model) + 1
    for i in range(edge):
        h_offset = (i - edge//2) * h_axis * space
        for j in range(edge):
            v_offset = (j - edge//2) * v_axis * space
            # id1 = _int_to_alphabet(i)
            # id2 = _int_to_alphabet(j)
            atom = Atom(
                name=f"{element}", 
                coord=center+h_offset+v_offset,
                bfactor=0,
                occupancy=1,
                altloc=" ",
                fullname=f"{element}",
                element=element,
                serial_number=serial_num)
            serial_num += 1
            atoms.append(atom)
    
    # residue = Residue(('D', 1, ' '), "DUM", 1)
    
    residues = []
    for i, atom in enumerate(atoms):
        residue = Residue(('d', atom.get_serial_number(), ' '), "DUM", "    ")
        residue.add(atom)
        residues.append(residue)
    
    chain = Chain(" ") if " " not in [chain.id for chain in model] else model[" "]
    for residue in residues:
        chain.add(residue)
    
    if " " not in [chain.id for chain in model]:
        model.add(chain)