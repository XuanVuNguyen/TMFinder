from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
from Bio.PDB.PDBIO import PDBIO

from tmfinder.biodata import remove_hetatm
from tmfinder.geometry import (
    get_CA_coords, 
    calc_symmetry_axis,
    ProteinSlicer,
    sphere_sample,
    GeometricModel,
    GeometricResidue,
    calc_center_of_mass
)

import numpy as np
import os
import argparse
from tqdm import tqdm

def main(args):
    in_pdb = args.in_pdb
    out_pdb = args.out_pdb
    if out_pdb is None:
        out_pdb = os.path.splitext(in_pdb)[0] + "_tmfinder.pdb"
        
    parser = PDBParser()
    sr = ShrakeRupley()
    
    struct = parser.get_structure("protein", in_pdb)
    model = struct[0]
    sr.compute(struct, level="R")
    
    symm_ax = calc_symmetry_axis(model)
    gm = GeometricModel(model, symm_ax)
    
    # TODO Check if the symmetry axis is enough so that we don't have to test all the normals.
    
    normals_sampled = sphere_sample(n_points = args.n_normals,
                                    theta_range = (0, np.pi/2)
                                    )
    
    for normal in tqdm(normals_sampled):
        slicer = ProteinSlicer(gm, normal, width=args.width)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMfinder")
    parser.add_argument("in_pdb", type=str, help="Input PDB file")
    parser.add_argument("out_pdb", type=str, help="Output PDB file", required=False)
    parser.add_argument("-n", "--n_normals", type=int, default=10000, help="Number of normals to sample")
    parser.add_argument("-w", "--width", type=float, default=15, help="Width of the slicer")
    parser.add_argument("-s", "--step", type=float, default=0.5, help="Step size for the slicer")
    
    args = parser.parse_args()
    main(args)

