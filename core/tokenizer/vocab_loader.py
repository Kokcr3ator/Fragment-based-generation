from core import MolGraph
from pathlib import Path
from typing import Dict

# ----------------------------------------------------------------------------
# Vocab & Operations Loader
# ----------------------------------------------------------------------------
class VocabLoader:
    """Load and cache motif vocabulary and operations for MolGraph."""
    _vocab: Dict[int, str] = {}
    _vocab_loaded = False
    _ops_loaded = False

    @classmethod
    def load_vocab(cls, vocab_path: Path) -> Dict[int, str]:
        if not cls._vocab_loaded:
            with vocab_path.open() as f:
                # each line: "<index> <smiles>"
                cls._vocab = {i: line.split()[1].strip() for i, line in enumerate(f)}
            MolGraph.load_vocab(str(vocab_path))
            cls._vocab_loaded = True
        return cls._vocab

    @classmethod
    def load_operations(cls, op_path: Path, num_ops: int) -> None:
        if not cls._ops_loaded:
            MolGraph.load_operations(str(op_path), num_ops)
            cls._ops_loaded = True