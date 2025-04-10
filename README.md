# HIV Treatment Optimization via Reinforcement Learning

## Project Overview

This project implements a reinforcement learning solution for optimizing HIV treatment strategies. The goal is to develop a control policy that maintains patient health while minimizing drug administration. The agent interacts with a simulated HIV patient model and learns to make treatment decisions based on the patient's immune system state variables.

## Problem Background

HIV treatment optimization is a challenging medical control problem. Continuous administration of antiretroviral drugs can lead to:
- Drug resistance
- Pharmaceutical side effects
- Weakening of the patient's natural immune response

This project implements a Structured Treatment Interruption (STI) strategy, where the agent intelligently decides when to administer drugs based on the patient's immune state.

## Environment

The environment simulates the dynamics of HIV infection using a system of deterministic non-linear equations with 6 state variables:
- `T1`: Healthy type 1 cells (CD4+ T-lymphocytes)
- `T1star`: Infected type 1 cells
- `T2`: Healthy type 2 cells (macrophages)
- `T2star`: Infected type 2 cells
- `V`: Free virus particles
- `E`: HIV-specific cytotoxic cells (CD8 T-lymphocytes)

At each time step (representing 5 days), the agent must choose one of 4 actions:
- Prescribe nothing (0)
- Prescribe reverse transcriptase inhibitors only (1)
- Prescribe protease inhibitors only (2)
- Prescribe both drugs (3)

The reward function encourages high values of `E` and low values of `V`, while penalizing drug administration.

## Implementation

### Approach

The agent uses a Fitted Q-Iteration (FQI) approach with XGBoost regression models:
- One XGBoost model per action to predict Q-values
- Epsilon-greedy exploration strategy with decay
- Bootstrapping for robust model training
- Feature standardization to improve regression performance

### Files Structure

```
├── README.md              # Project documentation
├── requirements.txt       # Required dependencies
├── best_model.pt          # Saved model weights (XGBoost models and scalers)
└── src/
    ├── env_hiv.py         # HIV patient environment
    ├── evaluate.py        # Evaluation functions
    ├── fast_env_py.py     # Optimized environment
    ├── grading.py         # Grading utilities
    ├── interface.py       # Agent interface definition
    ├── main.py            # Entry point for evaluation
    └── train.py           # Implementation of the ProjectAgent and training logic
```

## Agent Architecture

The agent uses XGBoost regression models to approximate the Q-function. Key components:

- **Models**: One XGBoost regressor for each possible action (4 total)
- **Scalers**: StandardScaler for normalizing input features for each model
- **Training**: Fitted Q-Iteration (FQI) with bootstrapping
- **Exploration**: Epsilon-greedy with exponential decay schedule

### Training Process

The training process follows these steps:

1. **Initial Exploration**: Collect a large number of transitions (30,000) using purely random actions to build an initial dataset.

2. **Fitted Q-Iteration (FQI)**: 
   - For each epoch (6 total), collect more transitions using the current policy
   - Bootstrap a training dataset by randomly sampling with replacement from all collected transitions
   - For each action, train a separate XGBoost model to predict Q-values
   - Use early stopping based on validation performance to prevent overfitting
   - Models are trained to minimize squared error between predicted Q-values and target Q-values

3. **Model Selection**:
   - After each epoch, evaluate the agent on the environment
   - Save the best-performing model based on evaluation rewards
   - The best model is stored in `best_model.pt` using joblib serialization

4. **Hyperparameters**:
   - Discount factor (gamma): 0.995
   - XGBoost parameters optimized for this specific task
   - Exploration decay from 1.0 to 0.01 with a decay rate of 0.995

## How to Run

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- gymnasium
- numpy
- xgboost
- scikit-learn
- joblib
- tqdm

### Training

To train the agent:

```bash
python src/train.py
```

Training hyperparameters can be modified in the `train.py` file.

### Evaluation

To evaluate a trained agent:

```bash
python src/main.py
```

This will load the agent from `best_model.pt` and run it on the default HIV patient.

## Results

The agent achieves consistent performance across both the default patient and randomized patient populations. The learned policy demonstrates a structured treatment interruption approach that maintains healthy T-cell counts while minimizing drug administration.

Key performance indicators:
- Maintains virus levels below critical thresholds
- Preserves and enhances immune response (E cells)
- Reduces drug usage compared to continuous treatment

### Model Storage

The best performing model is automatically saved during training as `best_model.pt`. This file contains:
- The trained XGBoost models for each action
- The StandardScaler objects for feature normalization

The saved model is loaded automatically by the evaluation script through the agent's `load()` method.
