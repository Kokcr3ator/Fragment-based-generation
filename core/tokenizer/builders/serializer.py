import networkx as nx
from collections import deque
from typing import Tuple, List, Union
from core.utils import bond_type_2_bond_token, bond_token_2_bond_type

NodeToken = Tuple[int, int]
# A node token is defined as (fragment_label, node_id)
EdgeToken = Tuple[int, int, int, int, str]
# An edge token is defined as (source_node_id, dest_node_id, source_rank, dest_rank, bondtype)
SpecialToken = str
# A special token is defined as a string, e.g. "(SOG)", "(EOG)"
GraphSequence = List[Union[SpecialToken, NodeToken, EdgeToken]]

# ----------------------------------------------------------------------------
# Convert fragment graph <-> sequential tokens
# ----------------------------------------------------------------------------

class SequenceSerializer:
    """Convert between fragment graphs and token sequences."""

    SOG = "(SOG)"
    EOG = "(EOG)"

    def to_sequence(self, fragment_graph: nx.MultiDiGraph) -> GraphSequence:
        """
        Convert a fragment graph to an ordered sequence of tokens. The order is defined as a BFS traversal 
        of the graph (see: self._BFS_order). The sequence starts with the special token "(SOG)" and ends with "(EOG)".
        The sequence is a list of tuples where each tuple is either a node or an edge. The node tuples are of the form (label, index)
        where label is the label of the fragment and index is the index of the node in the BFS order. The edge tuples are of the form
        (source_index, dest_index, source_rank, dest_rank, bond_token) where source_index and dest_index are the indices of the source and destination nodes in the BFS order,
        source_rank and dest_rank are the ranks of the connection sites in the source and destination fragments and bondtype is the bond type between the two connection sites.
        """
        seq = [self.SOG]
        order = self._BFS_order(fragment_graph)
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
                i,j = list(elem)
                for u, v in ((i, j), (j, i)):
                    # get_edge_data returns None if no edges exist, so or {} makes .values() empty
                    for e in (fragment_graph.get_edge_data(u, v) or {}).values():
                        seq.append((
                            order2index[u],
                            order2index[v],
                            e['source_rank'],
                            e['dest_rank'],
                            bond_type_2_bond_token(e['bondtype'])
                        ))

        seq.append(self.EOG)
        return seq
    
    def from_sequence(self, seq: GraphSequence) -> nx.MultiDiGraph:
        """
        Reconstructs the fragment graph from GraphSequence.
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
            # edge entries are 5‐tuples: (source_index, dest_index, source_rank, dest_rank, bond_token)
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
    
    def _BFS_order(self, G: nx.MultiDiGraph, root: int = None) -> List[Union[int, set]]:
        """
        Return a list of nodes and edges in BFS order starting from a root node.
        """     
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