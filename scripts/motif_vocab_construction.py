import multiprocessing as mp
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import config
from core.mol_graph import MolGraph
import argparse


def apply_operations(batch: List[Tuple[int, str]]) -> Counter:
    vocab = Counter()
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            mol = MolGraph(smi, tokenizer="motif")
            vocab = vocab + Counter(mol.motifs)
            pbar.update()
    return vocab

def motif_vocab_construction(
    train_path: str,
    vocab_path: str,
    operation_path: str,
    num_operations: int,
    num_workers: int,
):

    print(f"[{datetime.now()}] Construcing motif vocabulary from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.")

    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(train_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f"Total: {len(data_set)} molecules.\n")

    print(f"Processing...")
    vocab = Counter()
    MolGraph.load_operations(operation_path, num_operations)
    func = partial(apply_operations)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(func, batches):
            vocab = vocab + batch_vocab

    atom_list = [x for (x, _) in vocab.keys() if x not in MolGraph.OPERATIONS]
    atom_list.sort()
    new_vocab = []
    full_list = atom_list + MolGraph.OPERATIONS
    for (x, y), value in vocab.items():
        assert x in full_list
        new_vocab.append((x, y, value))
        
    index_dict = dict(zip(full_list, range(len(full_list))))
    sorted_vocab = sorted(new_vocab, key=lambda x: index_dict[x[0]])
    with open(vocab_path, "w") as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")
    
    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_path', type=str, default=config['train_path'])
    parser.add_argument('--vocab_path', type=str, default=config['vocab_path'])
    parser.add_argument('--operation_path', type=str, default=config['operation_path'])
    parser.add_argument('--num_workers', type=int, default=config['num_workers'])
    parser.add_argument('--num_operations', type=int, default=config['num_operations'])

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()


    motif_vocab_construction(
        train_path = args.train_path,
        vocab_path = args.vocab_path,
        operation_path = args.operation_path,
        num_operations = args.num_operations,
        num_workers = args.num_workers,
    )

    
    
    