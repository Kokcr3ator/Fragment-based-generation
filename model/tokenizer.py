from config import config
import os
import json
import ast 

class Tokenizer:
    def __init__(self, vocab_path: str = os.path.join(config.sequencing_config.config_path, 'vocab.json')):
        self.vocab_path = vocab_path
        self.vocab = {}
        self.reverse_vocab = {}
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str) -> None:
        """
        Load the vocabulary from a JSON file.
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def _encode_one_sequence(self, seq: str) -> list:
        """
        Tokenize a single sequence.
        """
        seq = seq.split(';')
        if seq[0] != '(SOG)' or seq[-1] != '(EOG)':
            raise ValueError("Sequence must start with (SOG) and end with (EOG).")
        seq = seq[1:-1]
        tokens = []
        pad_token = self.vocab['(PAD)']
        tokens.append([self.vocab['(SOG)'], pad_token])
        for token in seq:
            token = token.strip()
            token = ast.literal_eval(token)
            # if node
            if len(token) == 2:
                node_label, node_id = token
                if node_label in self.vocab and node_id in self.vocab:
                    tokens.append([self.vocab[node_label], self.vocab[node_id]])
                else:
                    raise ValueError(f"Node label {node_label} or node_id {node_id} not found in vocabulary.")
            # if edge
            elif len(token) == 5:
                src_node_str, dst_node_str, src_rank_str, dst_rank_str, bond_token = token
                if src_node_str in self.vocab and dst_node_str in self.vocab and \
                   src_rank_str in self.vocab and dst_rank_str in self.vocab and \
                   bond_token in self.vocab:
                    tokens.append([self.vocab[src_node_str], self.vocab[dst_node_str],
                                   self.vocab[src_rank_str], self.vocab[dst_rank_str], self.vocab[bond_token]])
                else:
                    raise ValueError(f"Edge tokens {token} not found in vocabulary.")
            else:
                raise ValueError("Invalid token format.")
            
        # Append end of graph token
        tokens.append([self.vocab['(EOG)'], pad_token])
        return tokens
    
    def __call__(self, sequences: list) -> list:
        """
        Tokenize a list of sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        tokenized_sequences = [self._encode_one_sequence(seq) for seq in sequences if isinstance(seq, str)]
        return tokenized_sequences

    def _decode_one_sequence(self, seq: list) -> str:
        """
        Decode a single sequence.
        """
        if self.reverse_vocab[seq[0][0]] != '(SOG)' or self.reverse_vocab[seq[-1][0]] != '(EOG)':
            raise ValueError("Sequence must start with (SOG) and end with (EOG).")
        seq = seq[1:-1]
        decoded_tokens = []
        decoded_tokens.append('(SOG)')
        for token in seq:
            if isinstance(token, tuple):
                decoded_tokens.append(tuple(self.reverse_vocab.get(t, t) for t in token))
        
        decoded_tokens.append('(EOG)')
        return ';'.join(map(str, decoded_tokens))
    
    def decode(self, sequences: list) -> list:
        """
        Decode a list of sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        decoded_sequences = [self._decode_one_sequence(seq) for seq in sequences if isinstance(seq, list)]
        return decoded_sequences