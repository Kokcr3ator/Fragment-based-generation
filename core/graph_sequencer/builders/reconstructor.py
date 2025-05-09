from core import MolGraph
import networkx as nx
from rdkit import Chem
from functools import lru_cache
from typing import Dict

# ----------------------------------------------------------------------------
# Molecule reconstruction from fragment graph
# ----------------------------------------------------------------------------

class MoleculeReconstructor:
    """Rebuild an RDKit Mol (SMILES) from the fragment graph."""

    def __init__(self, vocab: Dict[int, str]):
        self.vocab = vocab

    @lru_cache(maxsize=None)
    def _motif_graph(self, motif_smiles: str) -> nx.Graph:
        return MolGraph.motif_to_graph(motif_smiles)
    
    def from_fragment_graph(self, fragment_graph: nx.MultiDiGraph) -> nx.Graph:
        """
        Reconstructs the atom graph from the fragment graph. The atom graph is the graph where each node is an atom
        and the edges are the bonds between them.
        How it works:
        1. For each fragment in the fragment graph, build the corresponding motif graph. The node indices in the motif graph
           are the internal rank of the atoms in the fragment.

        2. The nodes of each motif graph are relabeled to be idx + offset, where offset is a counter of the number total atoms
           added until now. This is done to perform the union of the motif graphs which need to be disjoint.

        3. fragment2atoms is a dict that maps the the node index of the fragment graph to the node indices of the atoms of the
           corresponding motif graph. This is done in order to retrieve the edges between the fragments and add them to the atom graph.
        
        4. Add the edges between the fragments to the atom graph.

        5. Remove the connection sites from the atom graph.
        """

        fragment2atoms = {}
        offset = 0
        atom_graph = nx.Graph()
        for fragment in fragment_graph.nodes:
            fragment_label = fragment_graph.nodes[fragment]['label']
            motif_graph = self._motif_graph(self.vocab[fragment_label])

            n_atoms = motif_graph.number_of_nodes()
            fragment2atoms[fragment] = list(range(offset, offset + n_atoms))
            motif_graph = nx.relabel_nodes(motif_graph, lambda n: n + offset, copy=True)
            offset += n_atoms
            atom_graph = nx.union(atom_graph, motif_graph)

        connection_sites = set()
        for edge in fragment_graph.edges(data=True):           
            source, dest, data = edge

            source_rank = data['source_rank']
            dest_rank = data['dest_rank']
            bondtype = data['bondtype']

            source_star_idx = fragment2atoms[source][source_rank]
            connection_sites.add(source_star_idx)
            source_anchor = atom_graph.nodes[source_star_idx]['anchor']
            # the anchor corresponds to the rank of the atom connected to the connection site
            source_idx = fragment2atoms[source][source_anchor]

            dest_star_idx = fragment2atoms[dest][dest_rank]
            connection_sites.add(dest_star_idx)
            dest_anchor = atom_graph.nodes[dest_star_idx]['anchor']
            dest_idx = fragment2atoms[dest][dest_anchor]

            atom_graph.add_edge(source_idx, dest_idx,
                                bondtype=bondtype
                                )
        # remove the connection sites from the graph
        atom_graph.remove_nodes_from(connection_sites)
        return atom_graph

    
    def to_smiles(self, atom_graph: nx.Graph) -> str:
        """
        Converts the atom graph the the corresponding molecule's SMILES string.
        """
        mol = Chem.RWMol()
        node_to_idx = {}
        h_info = {} 

        for node, data in atom_graph.nodes(data=True):
            labels = data['label']
            symbol, IsAromatic, FormalCharge, NumExplicitHs =  labels

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