from core import MolGraph
from typing import Tuple, List, Union
from pathlib import Path
from config import config
from .fragment_vocab_loader import VocabLoader
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


class GraphSequencer:
    """SMILES from/to GraphSequence"""

    def __init__(
        self,
        vocab_path: str = config.sequencing_config['vocab_path'],
        operation_path: str = config.sequencing_config['operation_path'],
        num_operations: int = config.sequencing_config['num_operations'],
    ):
        vocab = VocabLoader.load_vocab(Path(vocab_path), return_vocab = True)
        VocabLoader.load_operations(Path(operation_path), num_operations)

        self._frag_builder = FragmentGraphBuilder()
        self._serializer = SequenceSerializer()
        self._reconstructor = MoleculeReconstructor(vocab)
        self._n_workers = 4

    def _graph2sequence_single(self, smiles: str) -> GraphSequence:
        """Tokenize one SMILES string."""
        mg = MolGraph(smiles, tokenizer="motif")
        fg = self._frag_builder.build(mg)
        return self._serializer.to_sequence(fg)

    def graph2sequence(
        self, 
        smiles: Union[str, List[str]]
    ) -> Union[GraphSequence, List[GraphSequence]]:
        """
        If `smiles` is a single string, return one GraphSequence.
        If it's a list (or tuple) of strings, tokenize them in parallel
        and return a list of GraphSequence.
        """
        if isinstance(smiles, str):
            return self._graph2sequence_single(smiles)

        if isinstance(smiles, (list, tuple)):
            if not smiles:
                return []

            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                iterator = executor.map(self._graph2sequence_single, smiles)
                results = list(
                    tqdm(
                        iterator,
                        total=len(smiles),
                        desc="Sequencing SMILES",
                        unit="mol"
                    )
                )
            return results

        raise TypeError(f"Expected str or list/tuple of str, got {type(smiles)}")

    def _sequence2graph_one(self, seq: GraphSequence) -> str:
        """Detokenize one GraphSequence to SMILES string."""

        fg = self._serializer.from_sequence(seq)
        atom_graph = self._reconstructor.from_fragment_graph(fg)
        smiles = self._reconstructor.to_smiles(atom_graph)
        return smiles
    
    def sequence2graph(self, seq: Union[List[Tuple[str, ...]], List[List[Tuple[str,...]]]]) -> str:
        """GraphSequence to SMILES"""
        if isinstance(seq, list):
            if isinstance(seq[0], str):
                return self._sequence2graph_one(seq)
            elif isinstance(seq[0], list):
                return [self._sequence2graph_one(s) for s in seq]
        raise TypeError("Provide a list of sequences or a single sequence.")

    def output_sequences(self, train_path: str = config.sequencing_config['train_path'], output_path: str = config.sequencing_config.sequences_path ) -> None:

        with open(train_path, 'r') as f:
            lines = f.readlines()
            smiles_list = [line.strip() for line in lines]
            
        sequenced_mols = self.graph2sequence(smiles_list)
    
        with open(output_path, 'w') as f:
            for seq in sequenced_mols:
                for tok in seq:
                    f.write(f"{tok};")
                # delete the last semicolon
                f.seek(f.tell() - 1, 0)
                f.truncate()
                f.write("\n")
