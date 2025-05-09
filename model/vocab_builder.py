from config import config
import ast
import json
import os
from collections import OrderedDict

class GraphVocabBuilder():

    def __init__(self):
        self.tokens = set()
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:

        self.tokens.add('(PAD)')
        self.tokens.add('(SOG)')
        self.tokens.add('(EOG)')
    

    def _process_one_sequence(self, seq: str):
        tokens = seq.split(';')
        if tokens[0] != '(SOG)' and tokens[-1] != '(EOG)':
            raise ValueError("Sequence must start with (SOG) and end with (EOG).")
        
        for token in tokens[1:-1]:
            token = ast.literal_eval(token)
            # if node
            if len(token) == 2:
                node_label, node_id = token
                self.tokens.add(node_label)
                self.tokens.add(node_id)
            # if edge
            elif len(token) == 5:
                src_node_str, dst_node_str, src_rank_str, dst_rank_str, bond_token = token
                self.tokens.add(src_node_str)
                self.tokens.add(dst_node_str)
                self.tokens.add(src_rank_str)
                self.tokens.add(dst_rank_str)
                self.tokens.add(bond_token)
    
    def _process_sequences(self, graph_sequences_path: str = config.sequencing_config.sequences_path) -> None:
        with open(graph_sequences_path, 'r') as f:
            for line in f:
                self._process_one_sequence(line.strip())
    
    def output_vocab(self) -> None:
        vocab_path = os.path.join(config.sequencing_config.config_path, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=4)

        print(f"Vocabulary saved to {vocab_path}")

    def build_vocab(self, graph_sequences_path: str = config.sequencing_config.sequences_path) -> OrderedDict:

        self._process_sequences(graph_sequences_path)
        ordered_keys = ["(PAD)", "(SOG)", "(EOG)"]

        # Sort nodes
        node_keys = sorted([k for k in self.tokens if k.startswith("node_")], key=lambda x: int(x.split("_")[1]))
        # Sort ranks
        rank_keys = sorted([k for k in self.tokens if k.startswith("rank_")], key=lambda x: int(x.split("_")[1]))
        # Sort fragments
        fragment_keys = sorted([k for k in self.tokens if k.startswith("frag_")], key=lambda x: int(x.split("_")[1]))

        edge_types = ["(:)", "(=)", "(*)", "(#)"]
        edge_keys = [k for k in edge_types if k in self.tokens]

        # Merge all keys in desired order
        ordered_keys += node_keys + rank_keys + edge_keys + fragment_keys

        self.vocab = OrderedDict()
        for i, k in enumerate(ordered_keys):
            self.vocab[k] = i

        self.output_vocab()
        return self.vocab

    