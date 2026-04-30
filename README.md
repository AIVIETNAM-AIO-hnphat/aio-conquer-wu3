# AIO Conquer for Warmup 3

# Project structure
```
aio-conquer-wu3/
│
├── data/
│   ├── raw/
│   │   └── RawData.csv
│   └── processed/
│
├── src/
│   ├── data/
│   │
│   ├── models/
│   │
│   └── utils/
│
├── notebooks/
│
├── configs/
│   └── config.yaml
│
├── experiments/
├── requirements.txt
├── run_pipeline.sh
├── README.md
└── .gitignore
```
## Setup

### 1. Clone repository
```bash
git clone https://github.com/AIVIETNAM-AIO-hnphat/aio-conquer-wu3.git 
cd aio-conquer-wu3
```

### 2. Create environment
```bash
python -m venv .venv

# Linux / Mac
source .venv/bin/activate


```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Weights & Biases (W&B) setup

```python
wandb login
```

Then paste your API key from https://wandb.ai/

## Run pipeline
```bash
bash run_pipeline.sh
```

Pipeline includes:

1. Data preprocessing
2. Model training
3. 

## Configuration

All parameters are defined in:

`configs/config.yaml`

Includes:

- training hyperparameters
- model settings
- data paths
- random seed

## Fixed random seed for reproducibility
- Fixed seed: 42
- Deterministic training (numpy, torch)

## Expected outputs
- Trained model: `model.pt`
- Logs: Weights & Biases dashboard

## Notes
- Data should be placed in data/raw/
- Do not commit large data files
## Branching conventions

- main: stable, production-ready
- develop: integration branch

Supporting branches:
- feature/<name>: new features
- experiment/<name>: experiments
- fix/<name>: bug fixes





