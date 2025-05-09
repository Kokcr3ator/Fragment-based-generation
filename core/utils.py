from typing import Dict, List, Tuple
import networkx as nx
import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

def merge_nodes(graph: nx.Graph, node1: int, node2: int) -> None:
    neighbors = [n for n in graph.neighbors(node2)]
    atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[node2]["atom_indices"])
    for n in neighbors:
        if node1 != n and not graph.has_edge(node1, n):
            graph.add_edge(node1, n)
        graph.remove_edge(node2, n)
    graph.remove_node(node2)
    graph.nodes[node1]["atom_indices"] = atom_indices

def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        try:
            out = Chem.MolFromSmiles(smiles)
            return out
        except:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            AllChem.SanitizeMol(mol, sanitizeOps=0)
    else:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol

def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol()
    node2idx = {}
    for node in fragment_graph.nodes:
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node)
        node2idx[node] = idx
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
    return Chem.MolToSmiles(motif, allBondsExplicit=True)


def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)

def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=True))
    if use_Isotope:
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))
    if not symm:
        conn_atoms = list(ordermap.keys())
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)
    return conn_atoms, ordermap

def bond_type_2_bond_token(bondtype: BondType) -> str:
    if bondtype == BondType.SINGLE:   return "(*)"
    if bondtype == BondType.DOUBLE:   return "(=)"
    if bondtype == BondType.TRIPLE:   return "(#)"
    if bondtype == BondType.AROMATIC: return "(:)"
    raise ValueError("unknown bond")

def bond_token_2_bond_type(bond_token: str) -> BondType:
    if  bond_token == "(*)": return BondType.SINGLE
    elif bond_token == "(=)": return BondType.DOUBLE
    elif bond_token == "(#)": return BondType.TRIPLE
    elif bond_token == "(:)": return BondType.AROMATIC
    else:
        raise ValueError(f"Unknown bond token: {bond_token}")
    

 

