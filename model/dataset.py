import torch
from torch.utils.data import Dataset
from typing import List, Union, Dict
import ast
from core.tokenizer.vocab_loader import VocabLoader
from config import config
from core.tokenizer.tokenizer import Tokenizer
import re
import os
import logging
import json


class GraphDataset(Dataset):
    def __init__(self, tokenized_mol_path: str = config.tokenization_config['tokenized_mols_path']):
        """
        Args:
            path: Path to .txt file with 1 sequence per line
            max_seq_len: Maximum number of tokens per sequence
        """
        self.tokenized_mol_path = tokenized_mol_path
        self.node_vocab = VocabLoader.load_vocab(config.tokenization_config['vocab_path'], return_vocab= True)
        # the node vocab is the same vocab used for the nodes so the mapping is node_label : node_label
        self.node_vocab = {idx:idx for idx,_ in self.node_vocab.items()}

    def build(self) -> None:
        if not os.path.isfile(self.tokenized_mol_path):
            self._tokenize()
            logging.info(f"Tokenized molecules and saved to {self.tokenized_mol_path}")

        self.max_node_id = self._calculate_max_node_id()
        self.max_rank = self._calculate_max_rank()
        self.max_seq_len = self._calculate_max_seq_len()

        self.vocab = self.node_vocab.copy()
        self.node_id_vocab = {}
        self.rank_vocab = {}

        special_tokens = ['(SOG)', '(EOG)', '(PAD)']
        self.special_tokens_vocab = {}

        for idx, token in zip(range(len(self.vocab),len(self.vocab) + len(special_tokens)), special_tokens):
            self.vocab[idx] = token
            self.node_vocab[token] = idx
            self.special_tokens_vocab[token] = idx

        for idx, node_id in zip(range(len(self.vocab),len(self.vocab) + self.max_node_id), range(self.max_node_id)):
            self.vocab[idx] = node_id
            self.node_id_vocab[node_id] = idx
        
        for idx, rank in zip(range(len(self.vocab),len(self.vocab) + self.max_rank), range(self.max_node_id)):
            self.vocab[idx] = rank
            self.rank_vocab[rank] = idx

        bond_labels= ['(*)', '(=)', '(#)', '(:)']
        self.bond_vocab = {}

        for idx, bond in zip(range(len(self.vocab),len(self.vocab) + len(bond_labels)), bond_labels):
            self.vocab[idx] = bond
            self.bond_vocab[bond] = idx

        self.inv_node_id_vocab = {v: k for k, v in self.node_id_vocab.items()}
        self.inv_node_vocab    = {v: k for k, v in self.node_vocab.items()}
        self.inv_rank_vocab    = {v: k for k, v in self.rank_vocab.items()}
        self.inv_bond_vocab    = {v: k for k, v in self.bond_vocab.items()}

        self._output_config()
        logging.info(f"Dataset config saved to {self.dataset_config_path}")
        self._output_vocab()
        logging.info(f"Vocabulary saved to {self.vocab_path}")
        
    def _tokenize(self) -> None:

        train_path = config.tokenization_config['train_path']
        with open(train_path, 'r') as f:
            lines = f.readlines()
            smiles_list = [line.strip() for line in lines]
        tokenizer = Tokenizer()
        tokenized_mols = tokenizer.tokenize(smiles_list)
        with open(self.tokenized_mol_path, 'w') as f:
            for seq in tokenized_mols:
                for tok in seq:
                    f.write(f"{tok};")
                # delete the last semicolon
                f.seek(f.tell() - 1, 0)
                f.truncate()
                f.write("\n")

    def _calculate_max_node_id(self) -> int:
        pair_pattern = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')

        with open(self.tokenized_mol_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Find all (label, id) pairs
        pairs = pair_pattern.findall(text)
        if not pairs:
            raise ValueError("No two‐element tuples found in the file.")

        node_ids = [int(node_id) for _, node_id in pairs]

        return max(node_ids)
    
    def _calculate_max_seq_len(self) -> int:
        with open(self.tokenized_mol_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            max_len = max(len(line.split(';')) for line in lines)
        return max_len
    
    def _calculate_max_rank(self) -> int:
        edge_pattern = re.compile(
        r"\(\s*(\d+)\s*,\s*"    # node_id 
        r"(\d+)\s*,\s*"         # dest_id 
        r"(\d+)\s*,\s*"         # source_rank
        r"(\d+)\s*,\s*"
        r"'[^']*'\s*"           # edge type
        r"\)"
        )

        with open(self.tokenized_mol_path, 'r', encoding='utf-8') as f:
            text = f.read()

        matches = edge_pattern.findall(text)
        if not matches:
            raise ValueError("No five‐element edge tuples found in the file.")

        source_ranks = [int(src) for _, _, src, _ in matches]
        dest_ranks   = [int(dst) for _, _, _, dst in matches]

        max_source = max(source_ranks)
        max_dest   = max(dest_ranks)
        overall_max = max(max_source, max_dest)
        return overall_max

    def _parse_sequence(self, line: str) -> List[Union[str, tuple]]:
        return list(ast.literal_eval("[" + line + "]"))

    def _output_config(self) -> None:
        dataset_config  = {
            "vocab_size": len(self.vocab),
            "max_node_id": self.max_node_id,
            "max_rank": self.max_rank,
            "max_seq_len": self.max_seq_len,
            "num_edge_types": len(self.bond_vocab),
            "node_vocab_size": len(self.node_vocab),
        }
        # write to config yaml file
        config_path = config.tokenization_config['config_path']
        self.dataset_config_path = os.path.join(config_path, 'dataset_config.yaml')
        with open(self.dataset_config_path, 'w') as f:
            for key, value in dataset_config.items():
                f.write(f"{key}: {value}\n")

    def _output_vocab(self) -> None:
        self.vocab_path = os.path.join(config.tokenization_config['config_path'], 'vocab.json')
        with open(self.vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
