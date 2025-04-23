## Installation

To install the required dependencies, run:

```bash
conda env create -f env.yml
```

## Usage

### Step 1: Calculate the Most Common Motifs

```bash
python scripts/merging.py
```

### Step 2: Create the Connection-Aware Motif Vocabulary

```bash
python scripts/motif_vocab_construction.py
```

### Step 3: Tokenize a Molecule

```python
from core.tokenizer import Tokenizer

tokenizer = Tokenizer()
smi = 'COc1ccc(C2=NN(C(=O)COC(=O)c3c(C)noc3N)[C@H](c3ccco3)C2)cc1'
star_graph = tokenizer.tokenize(smi)
```

### To go back from the star_graph to the smiles, run
```python
reconstructed_smi = tokenizer.detokenize(star_graph)
```

## Configuration

All parameters for vocabulary creation and file paths are defined in:

```text
config/config.yaml
```