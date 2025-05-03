from core import MolGraph
import networkx as nx

# ----------------------------------------------------------------------------
# Build fragment-level graph
# ----------------------------------------------------------------------------
class FragmentGraphBuilder:

    def build(self, mol: MolGraph) -> nx.MultiDiGraph:
        """
        Given a MolGraph, creates the fragment graph. The fragment graph is defined as multi‐directed graph where the nodes
        are the fragments and the edges are the bonds between the fragments. The edges are directed from the source fragment 
        to the destination fragment and contain the following attributes:

        - source_rank: the rank (internal order index of the fragment) of the connection site in the source fragment
        - dest_rank: the rank of the connection site in the destination fragment
        - bondtype: the bond type between the two connection sites
        """
        bpe_graph = mol.merging_graph
        atom_graph = mol.mol_graph
        fragment_graph = nx.MultiDiGraph()
        self._add_nodes(bpe_graph, fragment_graph)
        self._add_edges(bpe_graph, atom_graph, fragment_graph)
        return fragment_graph
    
    def _add_nodes(self, bpe_graph: nx.Graph, fragment_graph: nx.MultiDiGraph) -> None:
        for node, data in bpe_graph.nodes(data=True):
            fragment_graph.add_node(node, label=data["label"])
    
    def _add_edges(self, bpe_graph: nx.Graph, atom_graph: nx.Graph, fragment_graph: nx.MultiDiGraph) -> None:
        for n, d in bpe_graph.nodes(data=True):
            source_bpe_node = n
            # ordermap is a dict {source_mol_idx: source_rank} where source_mol_idx is the index of the atom in the mol_graph
            for source_mol_idx, source_rank in d['ordermap'].items():

                dest_mol_idx = atom_graph.nodes[source_mol_idx]['merge_targ']
                dest_bpe_node = atom_graph.nodes[dest_mol_idx]['bpe_node']

                if dest_bpe_node not in fragment_graph.nodes:
                    # label is the fragment label an integer which corresponds to a certain motif
                    label = bpe_graph.nodes[dest_bpe_node].get('label')
                    fragment_graph.add_node(dest_bpe_node, label = label)

                dest_rank = bpe_graph.nodes[dest_bpe_node]['ordermap'][dest_mol_idx]
                # actually source_mol_idx and dest_mol_idx correspond to special atoms denoted by '*' which 
                # are the connection sites of the fragments and are not the atoms of the molecule
                # so to get the bond type we need to go to the atoms connected to the connection sites
                # which is going to be only one atom per connection site and is denoted by the 'anchor' attribute.
                # the targ_atom is the atom connected to the destination connection site

                anchor      = atom_graph.nodes[source_mol_idx]['anchor']
                targ_atom = atom_graph.nodes[source_mol_idx]['targ_atom']
                btype = atom_graph[anchor][targ_atom]['bondtype']

                # add the edge from the source fragment to the destination fragment
                if source_mol_idx < dest_mol_idx:
                    fragment_graph.add_edge(source_bpe_node, dest_bpe_node, source_rank = source_rank, dest_rank = dest_rank, bondtype = btype)
