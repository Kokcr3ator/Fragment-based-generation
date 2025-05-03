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

### Step 3: Tokenize a molecule into a sequence

```python
from core.tokenizer import Tokenizer

tokenizer = Tokenizer()
smi = 'COC(=O)c1c(NC(=O)c2ccc(OC)cc2)sc(C)c1C'
sequence = tokenizer.tokenize(smi)
print(sequence)
```
```
['(SOG)', (1077, 0), (1191, 1), (1, 0, 0, 1, '(*)'), (1699, 2), (0, 2, 0, 0, '(*)'), '(EOG)']
```

### To go back from the sequence to the smiles, run
```python
reconstructed_smi = tokenizer.detokenize(sequence)
```

## Configuration

All parameters for vocabulary creation and file paths are defined in:

```text
config/config.yaml
```
