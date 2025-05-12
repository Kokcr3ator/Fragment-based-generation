import torch
from torch.utils.data import Dataset
from config import config
import os
import json
from .tokenizer import Tokenizer
import numpy as np

class GraphDataset(Dataset):

    def __init__(self, graph_sequences_path: str, vocab_path: str = os.path.join(config.sequencing_config['config_path'], 'vocab.json')):
        self.vocab_path = vocab_path
        self.vocab = self._load_vocab(vocab_path)
        self.tokenizer = Tokenizer(vocab_path)
        self.graph_sequences_path = graph_sequences_path
        self._build()
        self._output_config()

    @staticmethod
    def _load_vocab(vocab_path: str) -> None:
        """
        Load the vocabulary from a JSON file.
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        return vocab

    def _calculate_stats(self) -> None:
        """
        Calculate statistics for the dataset (these are used to build the architecture of the model).
        """
        ranks = [int(k.split("_")[1]) for k in self.vocab if k.startswith("rank_")]
        fragments = [int(k.split("_")[1]) for k in self.vocab if k.startswith("frag_")]
        node_ids = [int(k.split("_")[1]) for k in self.vocab if k.startswith("node_")]
        edge_types = ["(:)", "(=)", "(*)", "(#)"]
        special_tokens = ["(PAD)", "(SOG)", "(EOG)"]
        self.stats = {}
        self.stats['max_rank'] = max(ranks)
        self.stats['max_fragment'] = max(fragments)
        self.stats['max_node_id'] = max(node_ids)
        self.stats['num_edge_types'] = len(edge_types)
        self.stats['num_special_tokens'] = len(special_tokens)
        self.stats['max_num_nodes'] = max([len(seq) for seq in self.node_seqs])
        self.stats['max_num_edges'] = max([len(seq) for seq in self.edge_seqs])
        self.stats['max_seq_len'] = max([len(seq) for seq in self.routing_seqs])
        self.stats['vocab_size'] = len(self.vocab)

    def _preprocess_one_sequence(self, tokenized_seq: list):
        node_seq = []
        edge_seq = []
        routing_seq = []
        for token in tokenized_seq:
            if len(token) == 2:
                node_seq.append(token)
                routing_seq.append(0)

            elif len(token) == 5:
                edge_seq.append(token)
                routing_seq.append(1)
        return node_seq, edge_seq, routing_seq
    
    def _output_config(self) -> None:
        config_path = config.sequencing_config['config_path']
        self.dataset_config_path = os.path.join(config_path, 'dataset_config.yaml')
        with open(self.dataset_config_path, 'w') as f:
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
        
    def _build(self) -> None:
        """
        Build the dataset by loading the vocabulary and calculating statistics.
        """
        with open(self.graph_sequences_path, 'r') as f:
            seqs = [line.strip() for line in f]
        tokenized_seqs = self.tokenizer(seqs)
        self.node_seqs, self.edge_seqs, self.routing_seqs = zip(*[self._preprocess_one_sequence(seq) for seq in tokenized_seqs])
        del tokenized_seqs
        self._calculate_stats()

    def __len__(self):
        return len(self.node_seqs)

    def __getitem__(self, idx):
        return (self.node_seqs[idx], self.edge_seqs[idx], self.routing_seqs[idx])

def collate_fn(batch):
    PAD = 0
    ROUT_PAD = config.model_config.rout_PAD_token
    node_seqs, edge_seqs, routing_seqs = zip(*batch)
    max_n_nodes_per_seq = config.dataset_config['max_num_nodes']
    max_n_edges_per_seq = config.dataset_config['max_num_edges']
    max_routing_seq_len = max_n_nodes_per_seq + max_n_edges_per_seq
    nodes = [torch.LongTensor(seq) for seq in node_seqs]
    edges = [torch.LongTensor(seq) for seq in edge_seqs]
    routs = [torch.LongTensor(seq) for seq in routing_seqs]
    
    nodes = [torch.cat([x, torch.full((max_n_nodes_per_seq - x.size(0), 2), PAD, dtype=torch.long)]) 
                     if x.size(0) < max_n_nodes_per_seq else x for x in nodes]
    
    edges = [torch.cat([x, torch.full((max_n_edges_per_seq - x.size(0), 5), PAD, dtype=torch.long)])
                     if x.size(0) < max_n_edges_per_seq else x for x in edges]

    routs = [torch.cat([x, torch.full((max_routing_seq_len - x.size(0),), ROUT_PAD, dtype=torch.long)])
                     if x.size(0) < max_routing_seq_len else x for x in routs]

    nodes = torch.stack(nodes)
    edges = torch.stack(edges)
    routs = torch.stack(routs)
    routs_next = torch.roll(routs, shifts=-1, dims=1)
    routs_next[:, -1] = ROUT_PAD
    node_mask = ~nodes.ne(PAD)
    edge_mask = ~edges.ne(PAD)
    rout_mask = ~routs.ne(ROUT_PAD)


    return {
            "nodes": nodes,
            "edges": edges,
            "routing": routs,
            "routing_next": routs_next,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "routing_mask": rout_mask,
            "node_flattened_attention_mask": flattened_attention_mask(max_n_nodes_per_seq*2, 2),
            "edge_flattened_attention_mask": flattened_attention_mask(max_n_edges_per_seq*5, 5),
            "seq_flattened_attention_mask": seq_flattened_attention_mask(routs),
            "seq_flattened_padding_mask": seq_flattened_padding_mask(routs),
            "node_head_attention_mask": node_head_attention_mask(routs),
            "edge_head_attention_mask": edge_head_attention_mask(routs),
        }

def flattened_attention_mask(seq_len, block_size) -> torch.Tensor:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be a multiple of block_size")
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    for i in range(1 , int(seq_len/block_size)):
        mask[:(i+1)*block_size, i*block_size:(i+1)*block_size] = False
    return mask

def seq_flattened_attention_mask(routing):
    B, _ = routing.shape
    max_nodes = config.dataset_config['max_num_nodes']
    max_edges = config.dataset_config['max_num_edges']
    num_tokens = max_nodes *2 + max_edges * 5
    mask = torch.zeros((B, num_tokens, num_tokens), dtype=torch.bool)
    for b in range(B):
        ptr = 0
        for i in range(len(routing[b]) -1):
            # if node
            if routing[b][i] == 0:
                mask[b, :ptr, ptr:ptr +2] = True    
                ptr +=2
            # if edge
            elif routing[b][i] == 1:
                mask[b, :ptr, ptr:ptr+5] = True
                ptr += 5
            else:
                continue
    return mask

def seq_flattened_padding_mask(routing):
    B, _ = routing.shape
    max_nodes = config.dataset_config['max_num_nodes']
    max_edges = config.dataset_config['max_num_edges']
    num_tokens = max_nodes *2 + max_edges * 5
    mask = torch.ones((B, num_tokens), dtype=torch.bool)
    for b in range(B):
        tokens_dict = {key: val for (key,val) in zip(*np.unique(routing[b], return_counts=True))}
        n_not_padded = tokens_dict[0] * 2 + tokens_dict[1] * 5
        mask[b, :n_not_padded] = False
    return mask

def node_head_attention_mask(routing):
    B, _ = routing.shape
    max_nodes = config.dataset_config['max_num_nodes']
    max_edges = config.dataset_config['max_num_edges']
    num_tokens = max_nodes *2 + max_edges * 5
    mask = torch.zeros((B, max_nodes, num_tokens), dtype=torch.bool)
    for b in range(B):
        q_ptr = 0
        k_ptr = 0
        for i in range(len(routing[b]) -1):
            # if node
            if routing[b][i] == 0:
                mask[b, :q_ptr, k_ptr:k_ptr +2] = True
                k_ptr +=2
                q_ptr += 1
            # if edge
            elif routing[b][i] == 1:
                mask[b, :q_ptr, k_ptr:k_ptr +5] = True
                k_ptr += 5
            else:
                continue
    return mask

def edge_head_attention_mask(routing):
    B, _ = routing.shape
    max_nodes = config.dataset_config['max_num_nodes']
    max_edges = config.dataset_config['max_num_edges']
    num_tokens = max_nodes *2 + max_edges * 5
    mask = torch.zeros((B, max_edges, num_tokens), dtype=torch.bool)
    for b in range(B):
        q_ptr = 0
        k_ptr = 0
        for i in range(len(routing[b]) -1):
            # if node
            if routing[b][i] == 0:
                mask[b, :q_ptr, k_ptr:k_ptr +2] = True
                k_ptr +=2
            # if edge
            elif routing[b][i] == 1:
                mask[b, :q_ptr, k_ptr:k_ptr +5] = True
                k_ptr += 5
                q_ptr += 1
            else:
                continue
    return mask
