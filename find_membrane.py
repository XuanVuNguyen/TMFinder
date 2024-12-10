from typing import Optional
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
from Bio.PDB.PDBIO import PDBIO

from tmfinder.biodata import remove_hetatm
from tmfinder.geometry import (
    get_CA_coords, 
    calc_symmetry_axis,
    sphere_sample,
    GeometricModel,
    GeometricResidue,
    calc_center_of_mass
)
from tmfinder.slicer import ProteinSlicer
from tmfinder.visualization import add_atom_plane
from tmfinder.utils import init_logger

import numpy as np
import os
import argparse
from tqdm import tqdm


def main(args):
    logger = init_logger(__name__, args.log)
    
    in_pdb = args.in_pdb
    out_pdb = args.out_pdb
    if out_pdb is None:
        out_pdb = os.path.splitext(in_pdb)[0] + "_tmfinder.pdb"
        
    parser = PDBParser()
    sr = ShrakeRupley()
    
    struct = parser.get_structure("protein", in_pdb)
    remove_hetatm(struct)
    model = struct[0]
    sr.compute(struct, level="R")
    
    rotation_axis = calc_symmetry_axis(model)
    
    # TODO Check if the symmetry axis is enough so that we don't have to test all the normals.
    
    slicer = ProteinSlicer(
        model, 
        normal=rotation_axis, 
        width=args.width, 
        step=args.step, 
        n_normals_sample=args.n_normals,
        seed=args.rand,
    )
    slicer.fit_normal()
    logger.info(f"Fitted normal: {slicer.normal}")
    logger.info(f"Fitted pivots: {slicer.pivot1}, {slicer.pivot2}")
    logger.info(f"Fitted Q score: {slicer.Q_score}")
    slicer.expand_pivots(0.1)
    
    model_com = calc_center_of_mass(model)
    add_atom_plane(struct[0],
                   model_com,
                   normal=slicer.normal,
                   pivot=slicer.pivot1,
                   edge=21,
                   element="N",
                   space =3)
    add_atom_plane(struct[0],
                   model_com,
                   normal=slicer.normal,
                   pivot=slicer.pivot2,
                   edge=21,
                   element="O",
                   space =3)
    
    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMfinder")
    parser.add_argument("-i", "--in_pdb", type=str, help="Input PDB file")
    parser.add_argument("-o", "--out_pdb", type=str, help="Output PDB file", required=False)
    parser.add_argument("-l", "--log", type=str, help="Log file", default="tmfinder.log")
    parser.add_argument("-n", "--n_normals", type=int, default=5000, help="Number of normals to sample")
    parser.add_argument("-w", "--width", type=float, default=15, help="Width of the slicer")
    parser.add_argument("-s", "--step", type=float, default=2, help="Step size for the slicer")
    parser.add_argument("-r", "--rand", type=Optional[int], default=None, help="Random seed")
    
    args = parser.parse_args()
    main(args)

