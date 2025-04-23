from .mol_graph import MolGraph
import networkx as nx
from .vocab import Vocab
import networkx as nx
from rdkit import Chem
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import config

ATOM_SYMBOL_VOCAB = Vocab(['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F']).vmap
ATOM_ISAROMATIC_VOCAB = Vocab([True, False]).vmap
ATOM_FORMALCHARGE_VOCAB = Vocab(["*", -1, 0, 1, 2, 3]).vmap
ATOM_NUMEXPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3]).vmap
ATOM_NUMIMPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3]).vmap

INV_ATOM_SYMBOL_VOCAB = {v: k for k, v in ATOM_SYMBOL_VOCAB.items()}
INV_ISAROMATIC_VOCAB = {v: k for k, v in ATOM_ISAROMATIC_VOCAB.items()}
INV_FORMALCHARGE_VOCAB = {v: k for k, v in ATOM_FORMALCHARGE_VOCAB.items()}
INV_NUMEXPLICITHS_VOCAB = {v: k for k, v in ATOM_NUMEXPLICITHS_VOCAB.items()}
INV_NUMIMPLICITHS_VOCAB = {v: k for k, v in ATOM_NUMIMPLICITHS_VOCAB.items()}

class Tokenizer:
    _vocab_loaded = False
    _operations_loaded = False
    def __init__(self, vocab_path: str = config['vocab_path'], operation_path: str = config['operation_path'], num_operations: int = config['num_operations']):
        if not Tokenizer._vocab_loaded:
            MolGraph.load_vocab(vocab_path)
            pair_list = [line.strip("\r\n").split() for line in open(vocab_path)]
            motif_smiles_list = [motif for _, motif in pair_list]
            self.vocab = {i: motif for i, motif in enumerate(motif_smiles_list)}
            Tokenizer._vocab_loaded = True

        if not Tokenizer._operations_loaded:
            MolGraph.load_operations(operation_path, num_operations)
            Tokenizer._operations_loaded = True
    
    def create_star_graph(self, mol: MolGraph) -> nx.Graph:

        mol.relabel()
        mol_graph, bpe_graph =  mol.mol_graph, mol.merging_graph

        connectors = [n for n,d in mol_graph.nodes(data=True) if d.get('smarts') == '*']

        star_graph = nx.Graph()
        fragments_count= {}
        bpe_node_map = {}
        for n,d in bpe_graph.nodes(data=True):
            label = str(d['label'])
            if label not in fragments_count:
                fragments_count[label] = 0
            else:
                fragments_count[label] += 1
            ordermap = d.get('ordermap')
            bpe_node_map[n] = label + '_' + str(fragments_count[label])

            for idx, rank in ordermap.items():
                node = label + '_' + str(fragments_count[label]) + '_' + str(rank)
                connection_site = label + '_' + str(rank)
                anchor = mol_graph.nodes[idx]['anchor']
                targ_atom = mol_graph.nodes[idx]['targ_atom']
                smiles = d.get('motif')
                star_graph.add_node(node, anchor = anchor, targ_atom = targ_atom, motif = smiles, connection_site = connection_site)
                
        for n in connectors:
            m = mol_graph.nodes[n]['merge_targ']

            if n < m:
                source_bpe_node = mol_graph.nodes[n]['bpe_node']

                source_node = bpe_node_map[source_bpe_node] + '_' + str(bpe_graph.nodes[source_bpe_node]['ordermap'][n])

                dest_bpe_node = mol_graph.nodes[m]['bpe_node']
                dest_node = bpe_node_map[dest_bpe_node] + '_' + str(bpe_graph.nodes[dest_bpe_node]['ordermap'][m])

                anchor    = mol_graph.nodes[n]['anchor']
                targ_atom = mol_graph.nodes[n]['targ_atom']

                btype = mol_graph[anchor][targ_atom]['bondtype']
                star_graph.add_edge(source_node, dest_node, bondtype=btype)

        return star_graph
    
    def reconnect_star_graph(self, star_graph: nx.Graph) -> nx.Graph:
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
            G.add_edge(source_idx, dest_idx, bondtype=bondtype)
            to_delete.add(source_star)
            to_delete.add(dest_star)
        G.remove_nodes_from(to_delete)
            
        return G

    def mol_from_feature_graph(self, graph: nx.Graph) -> Chem.Mol:
        mol = Chem.RWMol()
        node_to_idx = {}
        h_info = {}  # Store hydrogen info to set later

        # First pass: add atoms with symbol, charge, aromaticity
        for node, data in graph.nodes(data=True):
            labels = data['label']
            symbol = INV_ATOM_SYMBOL_VOCAB[labels[0]]
            IsAromatic = INV_ISAROMATIC_VOCAB[labels[1]]
            FormalCharge = INV_FORMALCHARGE_VOCAB[labels[2]]
            NumExplicitHs = INV_NUMEXPLICITHS_VOCAB[labels[3]]
            # NumImplicitHs is usually computed automatically by RDKit

            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(FormalCharge)

            atom.SetIsAromatic(IsAromatic)

            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
            h_info[idx] = NumExplicitHs  # Save explicit H info for later

        # Second pass: add bonds
        for u, v, edge_data in graph.edges(data=True):
            bondtype = edge_data.get('bondtype')
            mol.AddBond(node_to_idx[u], node_to_idx[v], bondtype)

        # Now that the molecule is built, apply NumExplicitHs
        for idx, num_h in h_info.items():
            mol.GetAtomWithIdx(idx).SetNumExplicitHs(num_h)

        # Final sanitization and SMILES conversion
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print("[!] Sanitization failed:", e)

        return Chem.MolToSmiles(mol, canonical=True)    
    

    def tokenize(self, smiles: str) -> nx.Graph:
        mol_graph = MolGraph(smiles, tokenizer="motif")
        star_graph = self.create_star_graph(mol_graph)
        return star_graph
    
    def detokenize(self, graph: nx.Graph) -> str:
        try:
            star_graph = self.reconnect_star_graph(graph)
            smiles = self.mol_from_feature_graph(star_graph)

        except Exception as e:
            print("[!] Detokenization failed:", e)
            smiles = None

        return smiles
