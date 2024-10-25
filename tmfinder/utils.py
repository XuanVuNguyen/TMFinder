from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO

symm_ax = calc_symmetry_axis(model)
symm_ax = symm_ax / np.linalg.norm(symm_ax)
com = np.mean(get_CA_coords(model), axis=0)

atom1 = Atom(name="NA1", coord=com, bfactor=0, occupancy=1, altloc=" ", fullname="NA1", element="NA", serial_number=0)
atom2 = Atom(name="NA2", coord=com + symm_ax*15, bfactor=0, occupancy=1, altloc=" ", fullname="NA2", element="NA", serial_number=1)
chain = Chain("I")
res = Residue((' ', 1, ' '), "ION", 1)
res.add(atom1)
res.add(atom2)
chain.add(res)
model.add(chain)
# model.add(atom2)

io = PDBIO()
io.set_structure(struct)
io.save("data/4qi1_rotate.pdb")