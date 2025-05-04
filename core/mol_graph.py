from typing import Set, Tuple
import networkx as nx
import rdkit.Chem as Chem
from .utils import (fragment2smiles, get_conn_list, graph2smiles, smiles2mol, merge_nodes)
from config import config

class MolGraph(object):
    _vocab_loaded = False
    _operations_loaded = False
    

    @classmethod
    def load_operations(cls, operation_path: str = config.tokenization_config['operation_path'] , num_operations: int = config.tokenization_config['num_operations']):
        if not cls._operations_loaded:
            cls.NUM_OPERATIONS = num_operations
            with open(operation_path) as f:
                lines = list(f)
                cls.OPERATIONS = [code.strip('\r\n') for code in lines[:num_operations]]
            cls._operations_loaded = True

    @classmethod
    def load_vocab(cls, vocab_path:str = config.tokenization_config['vocab_path']):
        if not cls._vocab_loaded:
            with open(vocab_path, 'r') as f:
                # each line: "<index> <smiles>"
                cls.INDEX_2_MOTIF = {i: line.split()[1].strip() for i, line in enumerate(f)}
                cls.MOTIF_2_INDEX = {v: k for k, v in cls.INDEX_2_MOTIF.items()}
            cls._vocab_loaded = True
    
    @staticmethod
    def _is_vocab_loaded():
        if MolGraph._vocab_loaded:
            return True
        else:
            return False
        
    @staticmethod
    def _is_operations_loaded():
        if MolGraph._operations_loaded:
            return True
        else:
            return False
        
    def __init__(self,
        smiles: str,
        tokenizer: str="motif",
    ):  
        assert tokenizer in ["graph", "motif"], \
            "The variable `process_level` should be 'graph' or 'motif'. "
        self.smiles = smiles
        self.mol = smiles2mol(smiles, sanitize=True)
        self.mol_graph = self.get_mol_graph()
        
        if tokenizer == "motif":
            self.merging_graph = self.get_merging_graph()
            self.refragment()
            self.motifs = self.get_motifs()
            self.relabel()

    def get_mol_graph(self) -> nx.Graph:
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        for atom in self.mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['smarts'] = atom.GetSmarts()
            graph.nodes[atom.GetIdx()]['atom_indices'] = set([atom.GetIdx()])
            graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)

        for bond in self.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()

        return graph
    
    def get_merging_graph(self) -> nx.Graph:
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        merging_graph = mol_graph.copy()
        for code in self.OPERATIONS:
            for (node1, node2) in mol_graph.edges:
                if not merging_graph.has_edge(node1, node2):
                    continue
                atom_indices = merging_graph.nodes[node1]['atom_indices'].union(merging_graph.nodes[node2]['atom_indices'])
                pattern = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
                if pattern == code:
                    merge_nodes(merging_graph, node1, node2)
            mol_graph = merging_graph.copy()
        return nx.convert_node_labels_to_integers(merging_graph)

    def refragment(self) -> None:
        mol_graph = self.mol_graph.copy()
        merging_graph = self.merging_graph

        for node in merging_graph.nodes:
            atom_indices = self.merging_graph.nodes[node]['atom_indices']
            merging_graph.nodes[node]['motif_no_conn'] = fragment2smiles(self.mol, atom_indices)
            for atom_idx in atom_indices:
                mol_graph.nodes[atom_idx]['bpe_node'] = node

        for node1, node2 in self.mol_graph.edges:
            bpe_node1, bpe_node2 = mol_graph.nodes[node1]['bpe_node'], mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 != bpe_node2:
                conn1 = len(mol_graph)
                mol_graph.add_node(conn1)
                mol_graph.add_edge(node1, conn1)

                conn2 = len(mol_graph)
                mol_graph.add_node(conn2)
                mol_graph.add_edge(node2, conn2)
                
                mol_graph.nodes[conn1]['smarts'] = '*'
                mol_graph.nodes[conn1]['targ_atom'] = node2
                mol_graph.nodes[conn1]['merge_targ'] = conn2
                mol_graph.nodes[conn1]['anchor'] = node1
                mol_graph.nodes[conn1]['bpe_node'] = bpe_node1
                mol_graph[node1][conn1]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                merging_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                merging_graph.nodes[bpe_node2]['atom_indices'].add(conn2)
                mol_graph.nodes[conn2]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

        for node in merging_graph.nodes:
            atom_indices = merging_graph.nodes[node]['atom_indices']
            motif_graph = mol_graph.subgraph(atom_indices)
            merging_graph.nodes[node]['motif'] = graph2smiles(motif_graph)

        self.mol_graph = mol_graph

    def get_motifs(self) -> Set[str]:
        return [(self.merging_graph.nodes[node]['motif_no_conn'], self.merging_graph.nodes[node]['motif']) for node in self.merging_graph.nodes]

    def relabel(self):
        mol_graph = self.mol_graph
        bpe_graph = self.merging_graph

        for node in bpe_graph.nodes:
            bpe_graph.nodes[node]['internal_edges'] = []
            atom_indices = bpe_graph.nodes[node]['atom_indices']
            
            fragment_graph = mol_graph.subgraph(atom_indices)
            motif_smiles_with_idx = graph2smiles(fragment_graph, with_idx=True)
            motif_with_idx = smiles2mol(motif_smiles_with_idx)
            conn_list, ordermap = get_conn_list(motif_with_idx, use_Isotope=True)

            bpe_graph.nodes[node]['conn_list'] = conn_list
            bpe_graph.nodes[node]['ordermap'] = ordermap
            bpe_graph.nodes[node]['label'] = MolGraph.MOTIF_2_INDEX[ bpe_graph.nodes[node]['motif'] ]
            bpe_graph.nodes[node]['num_atoms'] = len(atom_indices)

        for node1, node2 in bpe_graph.edges:
            self.merging_graph[node1][node2]['label'] = 0
        edge_dict = {}
        for edge, (node1, node2, attr) in enumerate(mol_graph.edges(data=True)):
            edge_dict[(node1, node2)] = edge_dict[(node2, node1)] = edge
            bpe_node1 = mol_graph.nodes[node1]['bpe_node']
            bpe_node2 = mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 == bpe_node2:
                bpe_graph.nodes[bpe_node1]['internal_edges'].append(edge)
        
        for node, attr in mol_graph.nodes(data=True): 
            if attr['smarts'] == '*':
                anchor = attr['anchor']
                targ_atom = attr['targ_atom']
                mol_graph.nodes[node]['edge_to_anchor'] = edge_dict[(node, anchor)]
                mol_graph.nodes[node]['merge_edge'] = edge_dict[(anchor, targ_atom)]

    @staticmethod
    def get_atom_features(atom: Chem.rdchem.Atom=None, IsConn: bool=False, BondType: Chem.rdchem.BondType=None) -> Tuple[int, int, int, int, int]:
        if IsConn:
            Symbol, FormalCharge, NumExplicitHs = '*', '*', '*'      
            IsAromatic = True if BondType == Chem.rdchem.BondType.AROMATIC else False
            IsAromatic = IsAromatic
        else:
            Symbol = atom.GetSymbol()
            IsAromatic = atom.GetIsAromatic()
            FormalCharge = atom.GetFormalCharge()
            NumExplicitHs = atom.GetNumExplicitHs()
        return (Symbol, IsAromatic, FormalCharge, NumExplicitHs)

    @staticmethod
    def motif_to_graph(smiles: str,
                    ) -> nx.Graph:
        """
        Converts a connection aware motif SMILES string to a graph. The node indices correspond to the RDkit canonical ranking.
        This graph also contains the connection sites and for each connection site the corresponding anchor atom.
        """
        motif = smiles2mol(smiles, sanitize= False)
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(motif))
        ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=True))
        rank_mapping = {idx: rank for idx, rank in enumerate(ranks)}
        
        for atom in motif.GetAtoms():
            idx = atom.GetIdx()
            graph.nodes[idx]['smarts'] = atom.GetSmarts()
            graph.nodes[idx]['motif']  = smiles
            # if the atom is a connection site, we set the anchor atom
            if atom.GetSymbol() == '*':

                neighbors = list(graph.neighbors(idx))
                if len(neighbors) != 1:
                    raise ValueError(f"A connection site should have only one neighbor, but got {len(neighbors)}")
                # the anchor is mapped with the ranking since after this step the graph is relabeled given the ranking
                graph.nodes[idx]['anchor'] = rank_mapping[neighbors[0]]
            else:
                graph.nodes[idx]['label'] = MolGraph.get_atom_features(atom)

        # add all the internal bonds
        for bond in motif.GetBonds():
            i, j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            graph[i][j]['bondtype'] = bond.GetBondType()

        graph = nx.relabel_nodes(graph, rank_mapping, copy=True)

        return graph