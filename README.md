# SD-MoSE-V: Scientific Discovery - Mixture of Scientific Experts (Versioned Agentic System)

SD-MoSE-V is a GRAIL-V compliant agentic framework for climate equation discovery. It uses a core agent to orchestrate symbolic reasoning, memory-augmented regime tracking, and scientific verification.

## Project Structure

- `data/`: Raw and processed data.
- `sdmose/`: Main package.
  - `agent/`: **Core Cognition** (Perception, Reasoning, Verification, Control).
  - `experts/`: **Regime Experts** (Symbolic, Gating, Ensembles).
  - `memory/`: **Dynamics & Storage** (Transition matrices, Viterbi decoding).
  - `science/`: **Domain Knowledge** (Chemistry laws, Physics constraints).
  - `learning/`: **Optimization** (EM algorithm, Belief updates).
- `configs/`: Hydra configuration files.
- `notebooks/`: Jupyter notebooks for analysis.
- `scripts/`: Executable scripts.

## Agent Loop

The `sdmose.agent.Agent` orchestrates the following loop:
1. **Perception**: Encodes raw data into observations.
2. **Retrieval**: Fetches relevant priors and historical regime patterns.
3. **Reasoning**: Proposes symbolic hypotheses for the current regime.
4. **Verification**: Validates hypotheses against scientific constraints.
5. **Learning**: Updates beliefs and regime experts using EM.

## Installation

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sdmose
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

Run the agent:
```bash
python scripts/run_agent.py
```
