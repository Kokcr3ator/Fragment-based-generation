from core import MolGraph
from typing import Tuple, List, Union
from pathlib import Path
from config import config
from .vocab_loader import VocabLoader
from .builders import FragmentGraphBuilder, SequenceSerializer, MoleculeReconstructor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


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
        vocab_path: str = config.tokenization_config['vocab_path'],
        operation_path: str = config.tokenization_config['operation_path'],
        num_operations: int = config.tokenization_config['num_operations'],
    ):
        vocab = VocabLoader.load_vocab(Path(vocab_path), return_vocab = True)
        VocabLoader.load_operations(Path(operation_path), num_operations)

        self._frag_builder = FragmentGraphBuilder()
        self._serializer = SequenceSerializer()
        self._reconstructor = MoleculeReconstructor(vocab)
        self._n_workers = 4

    def _tokenize_one(self, smiles: str) -> GraphSequence:
        """Tokenize one SMILES string."""
        mg = MolGraph(smiles, tokenizer="motif")
        fg = self._frag_builder.build(mg)
        return self._serializer.to_sequence(fg)

    def tokenize(
        self, 
        smiles: Union[str, List[str]]
    ) -> Union[GraphSequence, List[GraphSequence]]:
        """
        If `smiles` is a single string, return one GraphSequence.
        If it's a list (or tuple) of strings, tokenize them in parallel
        and return a list of GraphSequence.
        """
        if isinstance(smiles, str):
            return self._tokenize_one(smiles)

        if isinstance(smiles, (list, tuple)):
            if not smiles:
                return []

            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                iterator = executor.map(self._tokenize_one, smiles)
                results = list(
                    tqdm(
                        iterator,
                        total=len(smiles),
                        desc="Tokenizing SMILES",
                        unit="mol"
                    )
                )
            return results

        raise TypeError(f"Expected str or list/tuple of str, got {type(smiles)}")

    def _detokenize_one(self, seq: GraphSequence) -> str:
        """Detokenize one GraphSequence to SMILES string."""

        fg = self._serializer.from_sequence(seq)
        atom_graph = self._reconstructor.from_fragment_graph(fg)
        smiles = self._reconstructor.to_smiles(atom_graph)
        return smiles
    
    def detokenize(self, seq: GraphSequence) -> str:
        """GraphSequence to SMILES"""

        if isinstance(seq, str):
            return self._detokenize_one(seq)
        elif isinstance(seq, (list, tuple)):
            return [self._detokenize_one(s) for s in seq]
        else:
            raise TypeError(f"Expected str or list of str, got {type(seq)}")
