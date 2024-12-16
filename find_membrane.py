from typing import Optional
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
from Bio.PDB.PDBIO import PDBIO

from tmfinder.biodata import remove_hetatm
from tmfinder.geometry import calc_symmetry_axis, calc_center_of_mass
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

    struct = parser.get_structure("protein", in_pdb)
    remove_hetatm(struct)
    model = struct[0]

    rotation_axis = calc_symmetry_axis(model)

    # TODO Check if the symmetry axis is enough so that we don't have to test all the normals.

    slicer = ProteinSlicer(
        model,
        normal=rotation_axis,
        width=args.width,
        n_normals_sample=args.n_normals,
        seed=args.rand,
    )

    slicer.fit_normal()
    logger.info(f"Fitted normal: {slicer.normal}")
    logger.info(f"Fitted pivots: {slicer.pivot1}, {slicer.pivot2}")
    logger.info(f"Fitted Q score: {slicer.Q_score}")

    slicer.expand_pivots(0.5)

    model_com = calc_center_of_mass(model)
    add_atom_plane(
        struct[0],
        model_com,
        normal=slicer.normal,
        pivot=slicer.pivot1,
        edge=21,
        element="N",
        space=3,
    )
    add_atom_plane(
        struct[0],
        model_com,
        normal=slicer.normal,
        pivot=slicer.pivot2,
        edge=21,
        element="O",
        space=3,
    )

    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TMfinder: Locating transmembrane regions of proteins.",
        add_help=False,
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument(
        "-i", "--in_pdb", type=str, help="Input PDB file", required=True
    )

    optional.add_argument(
        "-o",
        "--out_pdb",
        type=str,
        help="Output PDB file. Defaut: <input>_tmfinder.pdb",
        required=False,
    )
    optional.add_argument(
        "-l",
        "--log",
        type=str,
        help="Log file. Default: tmfinder.log",
        default="tmfinder.log",
    )
    optional.add_argument(
        "-w",
        "--width",
        type=float,
        default=15,
        help="Width of the slicer. Default: 15 (angstrom)",
    )
    optional.add_argument(
        "-n",
        "--n_normals",
        type=int,
        default=5000,
        help="Number of normal vectors to be sampled. Default: 5000",
    )
    optional.add_argument(
        "-r",
        "--rand",
        type=Optional[int],
        default=None,
        help="Random seed. Defaut: None.",
    )
    optional.add_argument(
        "--help", "-h", action="help", help="show this help message and exit"
    )

    args = parser.parse_args()
    main(args)
