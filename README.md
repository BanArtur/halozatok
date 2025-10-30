# Assignment 2

Assignment for paper https://ilanrcohen.droppages.com/pdfs/stochasticLimited.pdf

## Structure overview:

# graph.py
Graph class lives here, we assume directed tree structures

# serialize.py
JSON serializing graphs

# distribution.py
Uniform and Normal distribution for edge costs

# dataset.py:
We can generate, save and load datasets.
We have spider and bounded depth trees.

Spider:
```python
from dataset import SpiderDataset

dataset = SpiderDataset(
    reward_distribution=NormalDistribution(0.4, 1.3), # Sampling vertex rewards from this
    min_legs=2,
    max_legs=10,
    min_lenght=1,
    max_length=5,
)

dataset.generate_dataset(1000)
dataset.save_dataset("base")
...
dataset.load_dataset("base")
for graph in dataset.graphs:
    ... # process graphs
```

Bounded depth trees:
```python
dataset = BoundedTreeDataset(
    reward_distribution=NormalDistribution(0.4, 1.3),
    max_depth=6,
    halt_prob_min=0.25,
    halt_prob_max=0.75,
    B=6, # B parameter from page 12, (beta, B) trees
    beta=8,
)

dataset.generate_dataset(1000)
dataset.save_dataset("base")
...
dataset.load_dataset("base")
for graph in dataset.graphs:
    ... # process graphs
```

## Setup

Create virtual environment:
```bash
python3 -m venv venv
```

Activate:
```bash
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Testing

```bash
python3 -m pytest tests/
```