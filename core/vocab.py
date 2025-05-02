import rdkit.Chem as Chem
from typing import List, Tuple, Dict
from .utils import smiles2mol
from collections import defaultdict

class Vocab:
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vmap = {item: idx for idx, item in enumerate(vocab_list)}

    def __getitem__(self, item):
        return self.vmap[item]

    def get_smiles(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return len(self.vocab_list)


_vocab_data = {
    "ATOM_SYMBOL"      : ['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F'],
    "ATOM_ISAROMATIC"  : [True, False],
    "ATOM_FORMALCHARGE": ["*", -1, 0, 1, 2, 3],
    "ATOM_NUMEXPLICITHS" : ["*", 0, 1, 2, 3],
    "ATOM_NUMIMPLICITHS" : ["*", 0, 1, 2, 3],
}

for name, lst in _vocab_data.items():
    # e.g. ATOM_SYMBOL_VOCAB = Vocab([...])
    globals()[f"{name}_VOCAB"]    = Vocab(lst)
    # e.g. INV_ATOM_SYMBOL_VOCAB = {0: '*', 1: 'N', …}
    globals()[f"INV_{name}_VOCAB"] = {i: v for i, v in enumerate(lst)}

ATOM_FEATURES = [
    ATOM_SYMBOL_VOCAB,
    ATOM_ISAROMATIC_VOCAB,
    ATOM_FORMALCHARGE_VOCAB,
    ATOM_NUMEXPLICITHS_VOCAB,
    ATOM_NUMIMPLICITHS_VOCAB,
]

BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_VOCAB = Vocab(BOND_LIST)


__all__ = [
    *(f"{name}_VOCAB" for name in _vocab_data),
    *(f"INV_{name}_VOCAB" for name in _vocab_data),
    "ATOM_FEATURES",
    "BOND_LIST",
    "BOND_VOCAB",
]

class MotifVocab(object):

    def __init__(self, pair_list: List[Tuple[str, str]]):
        self.motif_smiles_list = [motif for _, motif in pair_list]
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))

        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        conn_dict: Dict[int, Tuple[int, int]] = {}
        bond_type_motifs_dict = defaultdict(list)
        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            motif = smiles2mol(motif_smiles)
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))

            cur_orders = []
            vocab_conn_dict[motif_idx] = {}
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    bond_type = atom.GetBonds()[0].GetBondType()
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])
                    cur_orders.append(ranks[atom.GetIdx()])
                    bond_type_motifs_dict[bond_type].append(conn_offset)
                    nodes_idx.append(node_offset)
                    conn_offset += 1
                node_offset += 1
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()
        self.vocab_conn_dict = vocab_conn_dict
        self.conn_dict = conn_dict
        self.nodes_idx = nodes_idx
        self.num_atoms_dict = num_atoms_dict
        self.bond_type_conns_dict = bond_type_motifs_dict


    def __getitem__(self, smiles: str) -> int:
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1
    
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx
    
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]


    



