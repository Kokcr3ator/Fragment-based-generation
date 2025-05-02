from core import MolGraph
import networkx as nx
from rdkit import Chem
from functools import lru_cache
from typing import Dict
from core.vocab import (
        INV_ATOM_SYMBOL_VOCAB,
        INV_ATOM_ISAROMATIC_VOCAB,
        INV_ATOM_FORMALCHARGE_VOCAB,
        INV_ATOM_NUMEXPLICITHS_VOCAB,
        )

# ----------------------------------------------------------------------------
# Molecule reconstruction from star graph (WIP: reconstruction directly from fragment graph)
# ----------------------------------------------------------------------------

class MoleculeReconstructor:
    """Rebuild an RDKit Mol (SMILES) from a star graph."""

    def __init__(self, vocab: Dict[int, str]):
        self.vocab = vocab

    @lru_cache(maxsize=None)
    def _motif_graph(self, motif_smiles: str) -> nx.Graph:
        return MolGraph.motif_to_graph(motif_smiles)[0]
    
    # this is dogshit, but it works. i'll work on it later
    def from_star_graph(self, star_graph: nx.Graph) -> nx.Graph:
        """
        From the star graph, builds the graph where each node is an atom and the edges are the bonds between the atoms.
        """
        fragments = {}

        total_len = 0
        for node in star_graph.nodes:
            label = '_'.join(node.split('_')[:2])
            motif_idx = int(label.split('_')[0])
            if label not in fragments:
                fragments[label] = None
                motif_smiles = self.vocab[motif_idx]
                fragment_graph = self._motif_graph(motif_smiles)
                total_len += fragment_graph.number_of_nodes()      
        offset = total_len
        offsets = {}
        for fragment in star_graph.nodes:
            label = '_'.join(fragment.split('_')[:2])
            motif_idx = int(label.split('_')[0])
            if label not in offsets:
                fragment_graph = MolGraph.motif_to_graph(self.vocab[motif_idx])[0]
                offsets[label] = offset
                offset += fragment_graph.number_of_nodes()
                fragment_graph = nx.relabel_nodes(fragment_graph, lambda n: n + offsets[label] , copy=True)
                fragments[label] = fragment_graph
                        
        fragments_list = list(fragments.values())
        # union the fragments using union
        G = fragments_list[0]
        for i in range(1, len(fragments_list)):
            G = nx.union(G, fragments_list[i])
        to_delete = set()
        for edge in star_graph.edges(data=True):
            source, dest, data = edge
            source_label = '_'.join(source.split('_')[:2])
            dest_label = '_'.join(dest.split('_')[:2])
            
            source_star = int(source.split('_')[2]) + offsets[source_label]

            source_idx = list(G.neighbors(source_star))
            assert len(source_idx) == 1, f"Expected exactly one neighbor, got {len(source_idx)}"
            source_idx = source_idx[0]
            
            dest_star = int(dest.split('_')[2]) + offsets[dest_label]
            dest_idx = list(G.neighbors(dest_star))
            assert len(dest_idx) == 1, f"Expected exactly one neighbor, got {len(dest_idx)}"
            dest_idx = dest_idx[0]

            bondtype = data['bondtype']
            G.add_edge(source_idx, dest_idx,
                        bondtype=bondtype
                        )
            to_delete.add(source_star)
            to_delete.add(dest_star)
        G.remove_nodes_from(to_delete)
            
        return G
    
    def to_smiles(self, atom_graph: nx.Graph) -> str:
        """
        Converts the atom graph the the corresponding molecule's SMILES string.
        """
        mol = Chem.RWMol()
        node_to_idx = {}
        h_info = {} 

        for node, data in atom_graph.nodes(data=True):
            labels = data['label']
            symbol = INV_ATOM_SYMBOL_VOCAB[labels[0]]
            IsAromatic = INV_ATOM_ISAROMATIC_VOCAB[labels[1]]
            FormalCharge = INV_ATOM_FORMALCHARGE_VOCAB[labels[2]]
            NumExplicitHs = INV_ATOM_NUMEXPLICITHS_VOCAB[labels[3]]

            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(FormalCharge)

            atom.SetIsAromatic(IsAromatic)

            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
            h_info[idx] = NumExplicitHs 

        for u, v, edge_data in atom_graph.edges(data=True):
            bondtype = edge_data.get('bondtype')
            mol.AddBond(node_to_idx[u], node_to_idx[v], bondtype)

        for idx, num_h in h_info.items():
            mol.GetAtomWithIdx(idx).SetNumExplicitHs(num_h)

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print("[!] Sanitization failed:", e)

        return Chem.MolToSmiles(mol, canonical=True)   