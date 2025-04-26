from .mol_graph import MolGraph
import networkx as nx
from .vocab import Vocab
import networkx as nx
from rdkit import Chem
from pathlib import Path
import sys
from rdkit.Chem.rdchem import BondType
from itertools import combinations
from collections import defaultdict
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import config
random.seed(420)

elements = ['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F']
INV_ATOM_SYMBOL_VOCAB = {i: e for i, e in enumerate(elements)}
isaromatic = [True, False]
INV_ISAROMATIC_VOCAB = {i: e for i, e in enumerate(isaromatic)}
formalcharge = ['*', -1, 0, 1, 2, 3]
INV_FORMALCHARGE_VOCAB = {i: e for i, e in enumerate(formalcharge)}
numexplicitHs = ['*', 0, 1, 2, 3]
INV_NUMEXPLICITHS_VOCAB = {i: e for i, e in enumerate(numexplicitHs)}
numimplicitHs = ['*', 0, 1, 2, 3]
INV_NUMIMPLICITHS_VOCAB = {i: e for i, e in enumerate(numimplicitHs)}


class Tokenizer:
    _operations_loaded = False
    def __init__(self, vocab_path: str = config['vocab_path'], operation_path: str = config['operation_path'], num_operations: int = config['num_operations'], ordering: str = 'BFS', add_internal_bonds: bool = True):
        self._vocab_loaded = False
        self._operations_loaded = False
        self.ordering = ordering
        self.add_internal_bonds = add_internal_bonds
        if not self._vocab_loaded:
            MolGraph.load_vocab(vocab_path)
            pair_list = [line.strip("\r\n").split() for line in open(vocab_path)]
            motif_smiles_list = [motif for _, motif in pair_list]
            self.vocab = {i: motif for i, motif in enumerate(motif_smiles_list)}
            self._vocab_loaded = True

        if not self._operations_loaded:
            MolGraph.load_operations(operation_path, num_operations)
            self._operations_loaded = True

        self.order_dict = {
            'BFS': self._order_BFS,
            'DFS': self._order_DFS,
            'RANDOM': self._order_RANDOM,
            'VOCAB_ORDER': self._order_VOCAB_ORDER}
        
    def create_star_graph(self, mol: MolGraph) -> nx.Graph:
        """
        Creates the star graph i.e. the graph whose nodes are the connection sites of the fragments
        """
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
        """
        From the star graph, reconnects the fragments, add internal bonds in the fragments,
        removes the connection sites (star nodes)
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
            G.add_edge(source_idx, dest_idx, bondtype=bondtype)
            to_delete.add(source_star)
            to_delete.add(dest_star)
        G.remove_nodes_from(to_delete)
            
        return G

    def mol_from_feature_graph(self, graph: nx.Graph) -> Chem.Mol:
        mol = Chem.RWMol()
        node_to_idx = {}
        h_info = {} 

        for node, data in graph.nodes(data=True):
            labels = data['label']
            symbol = INV_ATOM_SYMBOL_VOCAB[labels[0]]
            IsAromatic = INV_ISAROMATIC_VOCAB[labels[1]]
            FormalCharge = INV_FORMALCHARGE_VOCAB[labels[2]]
            NumExplicitHs = INV_NUMEXPLICITHS_VOCAB[labels[3]]

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
    
    def _bond_type_2_bond_token(self, bondtype: BondType) -> str:
        if bondtype == BondType.SINGLE:   return "(*)"
        if bondtype == BondType.DOUBLE:   return "(=)"
        if bondtype == BondType.TRIPLE:   return "(#)"
        if bondtype == BondType.AROMATIC: return "(:)"
        if bondtype == 'INTERNAL' :       return "(-)"
        raise ValueError("unknown bond")
    
    def _bond_token_2_bond_type(self, bond_token: str) -> BondType:
        if  bond_token == "(*)": return BondType.SINGLE
        elif bond_token == "(=)": return BondType.DOUBLE
        elif bond_token == "(#)": return BondType.TRIPLE
        elif bond_token == "(:)": return BondType.AROMATIC
        elif bond_token == "(-)": return 'INTERNAL'
        else:
            raise ValueError(f"Unknown bond token: {bond_token}")
                
    def _add_internal_bonds(self, star_graph: nx.Graph) -> nx.Graph:

        groups = defaultdict(list)
        for node in star_graph.nodes():
            number, order, *_ = node.split('_')
            key = (number, order)
            groups[key].append(node)

        for nodes_with_same_pair in groups.values():
            for u, v in combinations(nodes_with_same_pair, 2):
                if not star_graph.has_edge(u, v):
                    star_graph.add_edge(u, v, bondtype='INTERNAL')
        return star_graph
    

    def _group_connection_sites(self, star_graph: nx.Graph) -> nx.Graph:
        groups = defaultdict(list)
        for node in star_graph.nodes():
            number, order, *_ = node.split('_')
            groups[(number, order)].append(node)

        mapping = {}
        for members in groups.values():
            fused_name = "/".join(sorted(members))
            for n in members:
                mapping[n] = fused_name

        H = nx.relabel_nodes(star_graph, mapping, copy=True)
        H.remove_edges_from(nx.selfloop_edges(H))

        return H
                

    def _order_BFS(self, G: nx.Graph) -> list:
        if self.add_internal_bonds:
            ordering = nx.bfs_tree(G, source=list(G.nodes())[0])
            return ordering
        else:
            grouped_connection_sites = self._group_connection_sites(G)
            grouped_ordering = nx.bfs_tree(grouped_connection_sites, source=list(grouped_connection_sites.nodes())[0])
            ordering = []
            for elem in grouped_ordering:
                ordering.extend(elem.split('/'))
            return ordering

    
    def _order_DFS(self, G: nx.Graph) -> list:
        if self.add_internal_bonds:
            ordering = nx.dfs_tree(G, source=list(G.nodes())[0])
            
        else:
            grouped_connection_sites = self._group_connection_sites(G)
            grouped_ordering = nx.dfs_tree(grouped_connection_sites, source=list(grouped_connection_sites.nodes())[0])
            ordering = []
            for elem in grouped_ordering:
                ordering.extend(elem.split('/'))
            
        return ordering
    def _order_RANDOM(self, G: nx.Graph) -> list:

        if self.add_internal_bonds:
            ordering = list(G.nodes())
            random.shuffle(ordering)
            return ordering
        else:
            grouped_connection_sites = self._group_connection_sites(G)
            grouped_ordering = list(grouped_connection_sites.nodes())
            random.shuffle(grouped_ordering)
            ordering = []
            for elem in grouped_ordering:
                ordering.extend(elem.split('/'))
            return ordering

    def _order_VOCAB_ORDER(self, G: nx.Graph) -> list:
        def parse_key(node):
            i, j, k = map(int, node.split('_'))
            return (i, j, k)
        ordering = sorted(list(G.nodes()), key=parse_key)
        return ordering

    def _order_nodes(self, G: nx.Graph) ->list:
        if self.add_internal_bonds:
            G = self._add_internal_bonds(G)
        ordering = self.order_dict[self.ordering](G)
        return ordering
        
    def star_graph_to_sequence(self, star_graph: nx.Graph) -> list:
        if self.add_internal_bonds:
            star_graph = self._add_internal_bonds(star_graph)
        ordering = self._order_nodes(star_graph)
        id_map = {node: i+1 for i, node in enumerate(ordering)}

        seq = ["(SOG)"]
        for node in ordering:
            label, _, site = node.split("_")
            seq.append(f"({label}_{site})")
            seq.append(f"({id_map[node]})")
        seq.append("(αΔ)")

        edges = []
        for u, v, data in star_graph.edges(data=True):
            i, j = id_map[u], id_map[v]
            if i > j: i, j = j, i
            edges.append((i, j, self._bond_type_2_bond_token(data["bondtype"])))
        edges.sort(key=lambda x: (x[0], x[1]))

        for i, j, bt in edges:
            seq.extend([f"({i})", f"({j})", bt])
        seq.append("(EOG)")
        return seq

    def tokenize(self, smiles: str) -> nx.Graph:
        mol_graph = MolGraph(smiles, tokenizer="motif")
        star_graph = self.create_star_graph(mol_graph)
        seq = self.star_graph_to_sequence(star_graph)
        return seq
    
    def build_fragments_disconnected_graph(self, edge_tokens: list) -> nx.Graph:
        G = nx.Graph()
        for i in range(0, len(edge_tokens), 3):
            u_tok, v_tok, b_tok = edge_tokens[i:i+3]

            u = int(u_tok.strip("()"))
            v = int(v_tok.strip("()"))
            if u not in G:
                G.add_node(u)
            if v not in G:
                G.add_node(v)
            if b_tok == "(-)":
                G.add_edge(u, v)
        return G

    
    def sequence_to_star_graph(self, tokens:list) -> nx.Graph:
        # 1) sanity checks & split into node‐block and edge‐block
        if tokens[0] != "(SOG)" or tokens[-1] != "(EOG)":
            raise ValueError("Sequence must start with (SOG) and end with (EOG)")
        try:
            α_idx = tokens.index("(αΔ)")
        except ValueError:
            raise ValueError("Missing start‐of‐edges token “(αΔ)”")

        node_tokens = tokens[1:α_idx]
        edge_tokens = tokens[α_idx+1:-1]
        if len(node_tokens) % 2 != 0:
            raise ValueError("Node block is not an even number of tokens")

        if self.add_internal_bonds:

            fragments_disconnected_graph = self.build_fragments_disconnected_graph(edge_tokens)
            id_2_connection_site = {}
            for i in range(0, len(node_tokens), 2):
                ntok = node_tokens[i]   # e.g. "(103_0)"
                itok = node_tokens[i+1] # e.g. "(4)"
                connection_site = ntok.strip("()")
                node_id     = int(itok.strip("()"))
                id_2_connection_site[node_id] = connection_site
            fragments_counter = {}
            id_to_type = {}
            fragments = list(nx.connected_components(fragments_disconnected_graph))
            for fragment in fragments:
                fragment_idx = id_2_connection_site[next(iter(fragment))].split("_")[0]
                if fragment_idx not in fragments_counter:
                    fragments_counter[fragment_idx] = 0
                else:
                    fragments_counter[fragment_idx] += 1
                
                for node_id in fragment:
                    connection_site = id_2_connection_site[node_id].split("_")[1]
                    label_site = f"{fragment_idx}_{fragments_counter[fragment_idx]}_{connection_site}"
                    id_to_type[node_id] = label_site

        else:

            id_to_type = {}
            fragments_counter = {}
            for i in range(0, len(node_tokens), 2):
                ntok = node_tokens[i]   # e.g. "(103_0)"
                itok = node_tokens[i+1] # e.g. "(4)"
                label_site = ntok.strip("()")
                node_id     = int(itok.strip("()"))
                if label_site not in fragments_counter:
                    fragments_counter[label_site] = 0
                else:
                    fragments_counter[label_site] += 1

                fragment_idx, connection_site = label_site.split("_")
                label_site = f"{fragment_idx}_{fragments_counter[label_site]}_{connection_site}"
                id_to_type[node_id] = label_site

        G = nx.Graph()
        for nid in id_to_type.keys():
            G.add_node(nid)

        if len(edge_tokens) % 3 != 0:
            raise ValueError("Edge block is not a multiple of 3 tokens")
        for i in range(0, len(edge_tokens), 3):
            u_tok, v_tok, b_tok = edge_tokens[i:i+3]
            u = int(u_tok.strip("()"))
            v = int(v_tok.strip("()"))
            bond = self._bond_token_2_bond_type(b_tok)
            if bond != 'INTERNAL':
                G.add_edge(u, v, bondtype=bond)

        nx.relabel_nodes(G, id_to_type, copy=False)

        return G
    
    def detokenize(self, seq: list) -> str:
        star_graph = self.sequence_to_star_graph(seq)
        try:
            star_graph = self.reconnect_star_graph(star_graph)
            smiles = self.mol_from_feature_graph(star_graph)

        except Exception as e:
            print("[!] Detokenization failed:", e)
            smiles = None

        return smiles
