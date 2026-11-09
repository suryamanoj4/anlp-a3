# ANLP Assignment 3

## Setup

```
uv sync
cp .env.example .env
```

## Part 1

```
uv run python -m src.part1.sanity
bash scripts/part1/test_training.sh
bash scripts/part1/test_training.sh config/part1/mamba.yaml
bash scripts/part1/test_training.sh config/part1/probe.yaml
bash scripts/part1/test_assn.sh
bash scripts/part1/test_assn.sh --heldout
```

## Part 2

```
bash ./scripts/part2/test_gnn.sh
bash ./scripts/part2/test_gnn.sh --ta
```
