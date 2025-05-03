from core import MolGraph
from typing import Tuple, List, Any, Union
from pathlib import Path
from config import config
from .vocab_loader import VocabLoader
from .builders import FragmentGraphBuilder, SequenceSerializer, MoleculeReconstructor


NodeToken = Tuple[int, int]
# A node token is defined as (fragment_label, node_id)
EdgeToken = Tuple[int, int, int, int, str]
# An edge token is defined as (source_node_id, dest_node_id, source_rank, dest_rank, bondtype)
SpecialToken = str
# A special token is defined as a string, e.g. "(SOG)", "(EOG)"
GraphSequence = List[Union[SpecialToken, NodeToken, EdgeToken]]


# ----------------------------------------------------------------------------
# Main Tokenizer 
# ----------------------------------------------------------------------------
class Tokenizer:
    """SMILES from/to GraphSequence"""

    def __init__(
        self,
        vocab_path: str = config['vocab_path'],
        operation_path: str = config['operation_path'],
        num_operations: int = config['num_operations'],
    ):
        vocab = VocabLoader.load_vocab(Path(vocab_path))
        VocabLoader.load_operations(Path(operation_path), num_operations)

        self._frag_builder = FragmentGraphBuilder()
        self._serializer = SequenceSerializer()
        self._reconstructor = MoleculeReconstructor(vocab)

    def tokenize(self, smiles: str) -> GraphSequence:
        """SMILES to GraphSequence"""
        mg = MolGraph(smiles, tokenizer="motif")
        fg = self._frag_builder.build(mg)
        return self._serializer.to_sequence(fg)

    def detokenize(self, seq: GraphSequence) -> str:
        """GraphSequence to SMILES"""
        fg = self._serializer.from_sequence(seq)
        atom_graph = self._reconstructor.from_fragment_graph(fg)
        smiles = self._reconstructor.to_smiles(atom_graph)
        return smiles
