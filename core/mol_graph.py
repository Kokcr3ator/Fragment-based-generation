"""For molecular graph processing."""
from typing import List, Optional, Set, Tuple

import networkx as nx
import rdkit.Chem as Chem

from .utils import (fragment2smiles, get_conn_list, graph2smiles, smiles2mol, merge_nodes)
from .vocab import MotifVocab
from config import config
from .vocab import (ATOM_SYMBOL_VOCAB, 
                    ATOM_ISAROMATIC_VOCAB, 
                    ATOM_FORMALCHARGE_VOCAB,
                    ATOM_NUMEXPLICITHS_VOCAB,
                    ATOM_NUMIMPLICITHS_VOCAB,
                    ATOM_FEATURES,
                    BOND_LIST,
                    BOND_VOCAB)

class MolGraph(object):
    _vocab_loaded = False
    _operations_loaded = False
    

    @classmethod
    def load_operations(cls, operation_path: str = config['operation_path'] , num_operations: int = config['num_operations']):

        if not cls._operations_loaded:
            cls.NUM_OPERATIONS = num_operations
            with open(operation_path) as f:
                lines = list(f)
                cls.OPERATIONS = [code.strip('\r\n') for code in lines[:num_operations]]
            cls._operations_loaded = True

    @classmethod
    def load_vocab(cls, vocab_path:str = config['vocab_path']):
        if not cls._vocab_loaded:
            with open(vocab_path) as f:
                pair_list = [line.strip("\r\n").split() for line in f]
            cls.MOTIF_VOCAB = MotifVocab(pair_list)
            cls.MOTIF_LIST = cls.MOTIF_VOCAB.motif_smiles_list
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
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

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
                mol_graph[node1][conn1]['label'] = mol_graph[node1][node2]['label']
                merging_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node2][conn2]['label'] = mol_graph[node1][node2]['label']
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
            bpe_graph.nodes[node]['label'] = MolGraph.MOTIF_VOCAB[ bpe_graph.nodes[node]['motif'] ]
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
            Symbol, FormalCharge, NumExplicitHs, NumImplicitHs = 0, 0, 0, 0       
            IsAromatic = True if BondType == Chem.rdchem.BondType.AROMATIC else False
            IsAromatic = ATOM_ISAROMATIC_VOCAB[IsAromatic]
        else:
            Symbol = ATOM_SYMBOL_VOCAB[atom.GetSymbol()]
            IsAromatic = ATOM_ISAROMATIC_VOCAB[atom.GetIsAromatic()]
            FormalCharge = ATOM_FORMALCHARGE_VOCAB[atom.GetFormalCharge()]
            NumExplicitHs = ATOM_NUMEXPLICITHS_VOCAB[atom.GetNumExplicitHs()]
            NumImplicitHs = ATOM_NUMIMPLICITHS_VOCAB[atom.GetNumImplicitHs()]
        return (Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs)

    @staticmethod
    def motif_to_graph(smiles: str,
                    motif_list: Optional[List[str]] = None
                    ) -> Tuple[nx.Graph, List[int], List[int]]:
        motif = smiles2mol(smiles, sanitize= False)
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(motif))
        
        for atom in motif.GetAtoms():
            idx = atom.GetIdx()
            graph.nodes[idx]['smarts'] = atom.GetSmarts()
            graph.nodes[idx]['motif']  = smiles
            if atom.GetSymbol() == '*':
                bondtype = atom.GetBonds()[0].GetBondType()
                graph.nodes[idx]['dummy_bond_type'] = bondtype
                graph.nodes[idx]['label'] = MolGraph.get_atom_features(IsConn=True,
                                                                        BondType=bondtype)
            else:
                graph.nodes[idx]['label'] = MolGraph.get_atom_features(atom)

        ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=True))

        mapping = {}
        for idx, rank in enumerate(ranks):
            graph.nodes[idx]['rank'] = rank
            mapping[idx] = rank

        for idx in mapping:
            graph.nodes[idx]['orig_idx'] = idx

        dummy_idxs = [atom.GetIdx() for atom in motif.GetAtoms() if atom.GetSymbol() == '*']
        dummy_idxs.sort(key=lambda idx: ranks[idx])
        for bond in motif.GetBonds():
            i, j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            graph[i][j]['bondtype'] = bond.GetBondType()
            graph[i][j]['label']    = BOND_VOCAB[bond.GetBondType()]

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        dummy_idxs = [mapping[i] for i in dummy_idxs]

        return graph, dummy_idxs, ranks