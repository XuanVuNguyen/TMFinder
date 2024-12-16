import pytest


@pytest.fixture
def model():
    pdb_file = "data/4qi1_in.pdb"
    parser = PDBParser()
    structure = parser.get_structure("4qi1", pdb_file)
    return structure[0]


def test_remove_hetatm(model):
    remove_hetatm(model)
    for chain in model:
        assert len(chain) == 228
        hetres = [res for res in chain.get_residues() if res.get_id()[0] != " "]
        assert len(hetres) == 0
