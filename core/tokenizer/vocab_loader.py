from typing import Dict, List, Union
from pathlib import Path
from core import MolGraph


class VocabLoader:
    _vocab: Dict[int, str] = {}
    _vocab_loaded = False
    _ops_loaded = False

    @classmethod
    def load_vocab(cls, vocab_path: Path, return_vocab: bool = False) -> Union[None, Dict[int, str]]:
        if not cls._vocab_loaded:
            with open(vocab_path, 'r') as f:
                cls._vocab = {i: line.split()[1].strip() for i, line in enumerate(f)}
            MolGraph.load_vocab(str(vocab_path))
            cls._vocab_loaded = True

        if return_vocab:
            return cls._vocab

    @classmethod
    def load_operations(cls, op_path: Path, num_ops: int) -> None:
        if not cls._ops_loaded:
            MolGraph.load_operations(str(op_path), num_ops)
            cls._ops_loaded = True

