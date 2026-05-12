# PSO-Optimized Artificial Neural Network for Concrete Strength Prediction

## Overview
This project implements and evaluates a **Particle Swarm Optimization (PSO)** based **Artificial Neural Network (ANN)** for predicting concrete compressive strength using the UCI Concrete Compressive Strength dataset.

Both the ANN and PSO algorithms were developed completely from scratch in Python without using deep learning frameworks. The project investigates how ANN architectures, PSO hyperparameters, and optimization strategies influence regression performance.

---

## Project Objectives
- Build a fully connected ANN from scratch
- Implement PSO from scratch based on metaheuristic optimization principles
- Integrate PSO with ANN training
- Predict concrete compressive strength as a regression task
- Evaluate the effect of ANN and PSO hyperparameters
- Compare different PSO configurations and optimization strategies

---

## Dataset
The project uses the Concrete Compressive Strength Dataset from the UCI Machine Learning Repository.

### Dataset Information
- **Instances:** 1030
- **Input Features:** 8
  - Cement
  - Blast Furnace Slag
  - Fly Ash
  - Water
  - Superplasticizer
  - Coarse Aggregate
  - Fine Aggregate
  - Age
- **Target Variable:** Concrete compressive strength (MPa)

### Preprocessing
- 70% Training / 30% Testing split
- Min-Max normalization applied to features and target values

---

# Implementation

## Artificial Neural Network (ANN)
The ANN implementation includes:
- Configurable hidden layers and neurons
- Activation functions:
  - ReLU
  - Tanh
  - Sigmoid
  - Linear
- Forward propagation
- Prediction functionality
- Parameter flattening for PSO optimization

Example architecture:

```python
[8, 10, 1]
```

---

## Particle Swarm Optimization (PSO)
The PSO implementation includes:
- Particle initialization
- Velocity and position updates
- Personal best and global best tracking
- Informant network topology
- Boundary handling methods
- Constriction factor
- Early stopping mechanism

PSO is used to optimize ANN weights and biases instead of backpropagation.

---

## PSO-ANN Coupling
Each particle in PSO represents a flattened vector of ANN weights and biases.

The optimization objective is minimizing:
- Mean Absolute Error (MAE)

The final optimized parameters are loaded back into the ANN for evaluation.

---

# Experimental Investigations

## 1. Effect of ANN Architecture

### Hyperparameters Tested
- Hidden neurons: 5, 10, 15
- Hidden layers: 1 vs 2
- Activations:
  - ReLU
  - Tanh

### Key Findings
- Smaller architectures performed better on the dataset
- `[8, 5, 1]` achieved the best performance
- ReLU outperformed Tanh overall
- Larger networks caused overfitting

---

## 2. Effect of PSO Evaluation Budget

### Configurations Tested
- 10 × 50
- 20 × 25
- 25 × 20
- 50 × 10

### Key Findings
- More iterations with fewer particles performed best
- `10 particles × 50 iterations` achieved the lowest MAE
- Large swarms with very few iterations performed poorly

---

## 3. Effect of PSO Acceleration Coefficients

### Coefficients Tested
- (2.0, 0.5)
- (1.49, 1.49)
- (0.5, 2.0)
- (2.5, 2.5)

### Key Findings
- Balanced coefficients improved stability
- Very high coefficients caused instability
- Higher cognitive influence improved exploration

---

# Extended Experiments

## Baseline ANN vs PSO-Trained ANN

| Model | Test MAE |
|---|---|
| Random ANN | 0.435 |
| PSO-Trained ANN | 0.108 |

PSO reduced prediction error by approximately **75%**.

---

## Constriction Factor
Using the PSO constriction factor improved convergence stability and reduced MAE.

---

## Boundary Handling Methods
The **absorb** method achieved the best performance.

---

## Informant Network Size

Best performance was achieved with:

```python
k = 3 informants
```

---

## Early Stopping
Early stopping reduced computational cost by approximately **33%** while maintaining strong performance.

---

# Best Configuration

## Optimal Setup

### ANN Architecture
```python
[8, 5, 1]
```

### Activation
```python
ReLU + Linear Output
```

### PSO Configuration
- 10 particles
- 50 iterations
- Cognitive coefficient (β) = 2.0
- Social coefficient (γ) = 0.5
- Constriction factor enabled
- Absorb boundary handling
- k = 3 informants

### Best Result
- Test MAE ≈ 0.089
- Approximate real-world error ≈ 7.14 MPa

---

# Project Structure

```bash
├── ann.py                  # ANN implementation
├── pso.py                  # PSO implementation
├── train_ann.py            # Training pipeline
├── data/
│   └── concrete.csv        # Dataset
├── results/
│   └── experiment_outputs
├── README.md
```

---

# How to Run

## Install Requirements

```bash
pip install numpy pandas scikit-learn
```

## Run Training

```bash
python train_ann.py
```

---

# Results
The project demonstrates that:
- PSO can effectively train ANNs without backpropagation
- Smaller ANN architectures generalize better on small datasets
- PSO hyperparameters strongly affect convergence quality
- Informant topology and boundary handling significantly influence optimization performance

---

# Future Improvements
Possible extensions include:
- Adaptive PSO parameters
- Hybrid PSO + gradient descent optimization
- Alternative swarm topologies
- Feature engineering
- K-fold cross-validation
- Multi-objective optimization

---

# References
1. UCI Concrete Compressive Strength Dataset  
2. Kennedy, J. & Eberhart, R. (1995) Particle Swarm Optimization  
3. Clerc, M. & Kennedy, J. (2002) The Particle Swarm  
4. Luke, S. (2013) Essentials of Metaheuristics  
5. Ratnaweera, A. et al. (2004) Self-organizing Hierarchical PSO  

---

# Author
Bilgah Johnson  
B.E. Computer Science Graduate | MSc Data Science 
