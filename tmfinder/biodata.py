from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
from Bio.PDB.PDBIO import PDBIO

import os

RESNAME_TO_FASTA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
HYDROPHOBIC = ["PHE", "GLY", "ILE", "LEU", "MET", "VAL", "TRP", "TYR"]
HYDROPHILIC = ["ALA", "CYS", "ASP", "GLU", "HIS", "LYS", "ASN", "PRO", "GLN", "ARG", "SER", "THR"]

# def protonate(pdb_in, pdb_out=None, pH=7.4):
#     if pdb_out is None:
#         pdb_out = pdb_in.replace(".pdb", "_processed.pdb")
#     os.system(f"obabel -i pdb {pdb_in} -O {pdb_out} -p {pH}")

def remove_hetatm(pdb_in, pdb_out=None):
    parser= PDBParser()
    structure = parser.get_structure("structure", pdb_in)
    
    remove_residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    remove_residue_ids.append(residue.get_full_id())
    
    for rm_id in remove_residue_ids:
        _, model_id, chain_id, res_id = rm_id
        structure[model_id][chain_id].detach_child(res_id)
    
    if pdb_out is None:
        pdb_out = pdb_in.replace(".pdb", "_processed.pdb")
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_out)

def get_res_property(resname: str):
    if resname in HYDROPHOBIC:
        return "hydrophobic"
    elif resname in HYDROPHILIC:
        return "hydrophilic"
    else:
        return "unknown"
