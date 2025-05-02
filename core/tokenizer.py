from .mol_graph import MolGraph
import networkx as nx
from rdkit import Chem
from collections import deque
from typing import Tuple, List, Any, Union
from config import config
from .utils import bond_type_2_bond_token, bond_token_2_bond_type
from .vocab import (
        INV_ATOM_SYMBOL_VOCAB,
        INV_ATOM_ISAROMATIC_VOCAB,
        INV_ATOM_FORMALCHARGE_VOCAB,
        INV_ATOM_NUMEXPLICITHS_VOCAB,
        )

NodeToken = Tuple[int, int]
# A node token is defined as (fragment_label, node_id)
EdgeToken = Tuple[int, int, int, int, str]
# An edge token is defined as (source_node_id, dest_node_id, source_rank, dest_rank, bondtype)
SpecialToken = str
# A special token is defined as a string, e.g. "(SOG)", "(EOG)"
GraphSequence = List[Union[SpecialToken, NodeToken, EdgeToken]]

class Tokenizer:

    def __init__(self, vocab_path: str = config['vocab_path'], 
                 operation_path: str = config['operation_path'], 
                 num_operations: int = config['num_operations']):

        if not MolGraph._is_vocab_loaded():
            MolGraph.load_vocab(vocab_path)
            pair_list = [line.strip("\r\n").split() for line in open(vocab_path)]
            motif_smiles_list = [motif for _, motif in pair_list]
            self.vocab = {i: motif for i, motif in enumerate(motif_smiles_list)}

        if not MolGraph._is_operations_loaded():
            MolGraph.load_operations(operation_path, num_operations)

        
    def create_fragment_graph(self, mol: MolGraph) -> nx.MultiDiGraph:
        """
        Given a MolGraph, creates the fragment graph. The fragment graph is defined as multi‐directed graph where the nodes
        are the fragments and the edges are the bonds between the fragments. The edges are directed from the source fragment 
        to the destination fragment and contain the following attributes:

        - source_rank: the rank (internal order index of the fragment) of the connection site in the source fragment
        - dest_rank: the rank of the connection site in the destination fragment
        - bondtype: the bond type between the two connection sites
        """
        # mol_graph is the graph in which nodes are the atoms and edges are the bonds 
        # bpe_graph is the graph in which nodes are the fragments but the edges only say if the fragments are connected but not how
        mol_graph, bpe_graph =  mol.mol_graph, mol.merging_graph
        fragment_graph = nx.MultiDiGraph()

        for n, d in bpe_graph.nodes(data=True):
            source_bpe_node = n
            label = d['label']
            if label not in fragment_graph.nodes:
                fragment_graph.add_node(source_bpe_node, label = label)
            # ordermap is a dict {source_mol_idx: source_rank} where source_mol_idx is the index of the atom in the mol_graph
            for source_mol_idx, source_rank in d['ordermap'].items():

                dest_mol_idx = mol_graph.nodes[source_mol_idx]['merge_targ']
                dest_bpe_node = mol_graph.nodes[dest_mol_idx]['bpe_node']

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

                anchor      = mol_graph.nodes[source_mol_idx]['anchor']
                targ_atom = mol_graph.nodes[source_mol_idx]['targ_atom']
                btype = mol_graph[anchor][targ_atom]['bondtype']

                # add the edge from the source fragment to the destination fragment
                if source_mol_idx < dest_mol_idx:
                    fragment_graph.add_edge(source_bpe_node, dest_bpe_node, source_rank = source_rank, dest_rank = dest_rank, bondtype = btype)
        
        self.fragment_graph = fragment_graph
        return fragment_graph
    
    def fragment_graph2star_graph(self, fragment_graph: nx.MultiDiGraph) -> nx.Graph:
        """
        From the fragment graph, creates the corresponding star graph.
        The star graph is only used as an intermediate representation useful for the detokenization process.
        The star graph is defined as a graph where the nodes are the connection sites of the fragments and the edges are the bonds between the connection sites.
        In particular in the star graph the index of the nodes are "fragmentlabel_fragmentcount_rank" where:
        - fragmentlabel is the label of the fragment in the vocabulary

        - fragmentcount represents a counter of the number of fragments with the same label e.g there can be 2 fragments with label 420
          so the rank 0 connection site of the first fragment will be 420_0_0 and the rank 0 connection site of the second fragment will be 420_1_0
        
        - rank is the rank of the connection site in the fragment
        """

        fragments_counter = {}
        fragments_number = {}
        star_graph = nx.Graph()
        for n, d in fragment_graph.nodes(data=True):
            label = d['label']
            if label not in fragments_counter:
                fragments_counter[label] = 0
            else:
                fragments_counter[label] += 1
            fragments_number[n] = str(label) + '_' + str(fragments_counter[label])
        
        for source, dest, d in fragment_graph.edges(data=True):
            source_rank = d['source_rank']
            dest_rank = d['dest_rank']
            btype = d['bondtype']
            source_node = fragments_number[source] + '_' + str(source_rank)
            dest_node = fragments_number[dest] + '_' + str(dest_rank)
            star_graph.add_edge(source_node, dest_node, bondtype = btype)
        
        return star_graph
    
    def fragment_graph2sequence(self, fragment_graph: nx.MultiDiGraph) -> GraphSequence:
        """
        Convert a fragment graph to an ordered sequence of tokens. The order is defined as a BFS traversal 
        of the graph (see: self._order_BFS_nodes_and_edges). The sequence starts with the special token "(SOG)" and ends with "(EOG)".
        The sequence is a list of tuples where each tuple is either a node or an edge. The node tuples are of the form (label, index)
        where label is the label of the fragment and index is the index of the node in the BFS order. The edge tuples are of the form
        (source_index, dest_index, source_rank, dest_rank, bond_token) where source_index and dest_index are the indices of the source and destination nodes in the BFS order,
        source_rank and dest_rank are the ranks of the connection sites in the source and destination fragments and bondtype is the bond type between the two connection sites.
        """
        seq = ["(SOG)"]
        order = self._order_BFS_nodes_and_edges(fragment_graph)
        index = 0
        order2index = {}
        for elem in order:
            # if node
            if isinstance(elem, int):
                label = fragment_graph.nodes[elem]['label']
                seq.append((label, index))
                order2index[elem] = index
                index += 1
            # if edge
            elif isinstance(elem, set):
                i, j = list(elem)
                edges_ij = fragment_graph.get_edge_data(i,j)
                if edges_ij is not None:
                    for edge in edges_ij.values():
                        source_rank = edge['source_rank']
                        dest_rank = edge['dest_rank']
                        bondtype = edge['bondtype']
                        bondtype = bond_type_2_bond_token(bondtype)
                        seq.append((order2index[i], order2index[j], source_rank, dest_rank, bondtype))
                
                edges_ji = fragment_graph.get_edge_data(j,i)
                if edges_ji is not None:
                    for edge in edges_ji.values():
                        source_rank = edge['source_rank']
                        dest_rank = edge['dest_rank']
                        bondtype = edge['bondtype']
                        bondtype = bond_type_2_bond_token(bondtype)
                        seq.append((order2index[j], order2index[i], source_rank, dest_rank, bondtype))
                
        seq.append("(EOG)")
        return seq
    
    def sequence2fragment_graph(self, seq: List[Any]) -> nx.MultiDiGraph:
        """
        Reconstruct a MultiDiGraph from a sequence of the form:
        ["(SOG)",
        (label1, idx1),
        (label2, idx2),
        ...,
        (i_idx, j_idx, source_rank, dest_rank, bond_token),
        ...,
        "(EOG)"
        ]
        Parameters
        ----------
        seq : list
            Your token sequence. First element must be "(SOG)", last "(EOG)".
        Returns
        -------
        G : nx.MultiDiGraph
            Reconstructed graph with integer nodes and edge attrs:
            - 'source_rank'
            - 'dest_rank'
            - 'bondtype'
        """
        G = nx.MultiDiGraph()
        # 1) add nodes
        for elem in seq:
            if elem in ("(SOG)", "(EOG)"):
                continue
            # node entries are 2‐tuples: (label, index)
            if isinstance(elem, tuple) and len(elem) == 2:
                label, idx = elem
                G.add_node(idx, label=label)
        # 2) add edges
   
            if isinstance(elem, tuple) and len(elem) == 5:
                i_idx, j_idx, src_rank, dst_rank, bond_token = elem
                bondtype = bond_token_2_bond_type(bond_token)
                G.add_edge(
                    i_idx,
                    j_idx,
                    source_rank=src_rank,
                    dest_rank=dst_rank,
                    bondtype=bondtype
                )
        return G
    
    def star_graph2atom_graph(self, star_graph: nx.Graph) -> nx.Graph:
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
                fragment_graph = MolGraph.motif_to_graph(self.vocab[motif_idx])[0]
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
                
    def _order_BFS_nodes_and_edges(self, G: nx.Graph, root: int = None) -> list:
        """Return a list of nodes in BFS order and edges starting from start_node."""
        G = G.to_undirected()
        if nx.is_connected(G):
            assert "The graph is not connected"
        if root is None:
            # start from the node with the highest degree
            root = max(G.degree, key=lambda x: x[1])[0]
        visited_nodes = set()
        queue = deque([root])
        order = []
        visited_edges = []

        while queue:
            node = queue.popleft()
            if node not in visited_nodes:
                visited_nodes.add(node)
                order.append(node)
                # add neighbors that have not been visited
                for neighbor in G.neighbors(node):
                    if set([neighbor, node]) not in visited_edges:
                        if node in visited_nodes and neighbor in visited_nodes:
                            visited_edges.append(set([node, neighbor]))
                            order.append(set([node, neighbor]))
                    if neighbor not in visited_nodes:
                        queue.append(neighbor)
        return order
    
    def mol_from_atom_graph(self, graph: nx.Graph) -> Chem.Mol:
        """
        Converts the atom graph the the corresponding molecule's SMILES string.
        """
        mol = Chem.RWMol()
        node_to_idx = {}
        h_info = {} 

        for node, data in graph.nodes(data=True):
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

        for u, v, edge_data in graph.edges(data=True):
            bondtype = edge_data.get('bondtype')
            mol.AddBond(node_to_idx[u], node_to_idx[v], bondtype)

        for idx, num_h in h_info.items():
            mol.GetAtomWithIdx(idx).SetNumExplicitHs(num_h)

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print("[!] Sanitization failed:", e)

        return Chem.MolToSmiles(mol, canonical=True)   
    
    def tokenize(self, smiles: str) -> GraphSequence:
        mol_graph = MolGraph(smiles, tokenizer="motif")
        fragment_graph = self.create_fragment_graph(mol_graph)
        graph_sequence = self.fragment_graph2sequence(fragment_graph)
        return graph_sequence
    
    
    def detokenize(self, seq: list) -> str:
        fragment_graph = self.sequence2fragment_graph(seq)
        try:
            star_graph = self.fragment_graph2star_graph(fragment_graph)
            atom_graph = self.star_graph2atom_graph(star_graph)
            smiles = self.mol_from_atom_graph(atom_graph)

        except Exception as e:
            print("[!] Detokenization failed:", e)
            smiles = None

        return smiles
    
    

