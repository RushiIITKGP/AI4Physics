# Physics-Informed Neural Networks: Internal Representation Analysis

Investigating how PINNs learn physics-aware representations compared to standard neural networks through systematic experimentation, linear probing, and ablation studies.

## 🎯 Project Overview

This research project investigates the fundamental differences between **Physics-Informed Neural Networks (PINNs)** and **standard neural networks** in how they internally encode physical laws. Through rigorous experimentation on oscillator systems, we analyze whether physics-aware representations lead to better generalization and extrapolation capabilities.

## 🏗️ Project Structure

```
AI4Physics-project/
├── simple_oscillator.py                   # Simple Harmonic Oscillator experiments
├── duffing_oscillator.py                  # Duffing Oscillator experiments (enhanced)
```

## 🔬 Experimental Design

### Two-Stage Approach

**Stage 1: Simple Harmonic Oscillator** ([simple_oscillator.py](simple_oscillator.py))
- System: `mx'' + cx' + kx = 0`
- Parameters: m=1.0, c=1.0, k=20.0
- Purpose: Baseline testing and proof of concept
- Training: 1000 epochs, 80/20 train/test split

**Stage 2: Duffing Oscillator** ([duffing_oscillator.py](duffing_oscillator.py))
- System: `mx'' + cx' + kx + βx³ = F·cos(Ωt)`
- Parameters: m=1.0, c=0.25, k=1.0, β=5.0, F=0.5, Ω=1.2
- Purpose: Enhanced testing on nonlinear system
- Training: 5000 epochs, 20/80 train/test split (sparse data)

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy
```

### Running Experiments

**Simple Harmonic Oscillator:**
```bash
python simple_oscillator.py
```

**Duffing Oscillator (Enhanced):**
```bash
python duffing_oscillator.py
```

### Core Components

1. **Data Generation**: Analytical solutions using `scipy.integrate.solve_ivp`
2. **Model Architecture**: Shared MLP backbone (1→32→32→1) with Tanh activations
3. **Training Regimes**:
   - Normal MLP: Pure data fitting (MSE loss)
   - PINN: Data + Physics loss (weighted combination)
4. **Analysis Tools**:
   - Linear probing of hidden layers
   - Orthogonal projection ablation
   - Extrapolation testing
   - Physics violation metrics


## 🛠️ Technologies Used

- **PyTorch**: Neural network framework and automatic differentiation
- **SciPy**: ODE solving for ground truth data generation
- **Scikit-learn**: Linear probing and regression analysis
- **Matplotlib**: Visualization and plotting
- **NumPy**: Numerical computations
- **Pandas**: Data management and CSV operations

## 🔮 Future Directions

### Potential Extensions

1. **More Complex Systems**
   - Coupled oscillators
   - Partial differential equations
   - Multi-physics problems

2. **Advanced Analysis**
   - Nonlinear probing techniques
   - Representation visualization
   - Layer-wise analysis

3. **Architecture Improvements**
   - Adaptive physics weighting
   - Specialized network architectures
   - Hybrid approaches

4. **Applications**
   - Real-world physical systems
   - Engineering design optimization
   - Scientific discovery

## 📖 References

This project builds on foundational work in:
- Physics-Informed Neural Networks (Raissi et al., 2019)
- Neural network representation learning
- Linear probing in deep learning
- Ablation studies in neural networks

## 🤝 Contributing

This is a research project. Suggestions for improvements and extensions are welcome:

1. More complex physical systems
2. Advanced probing techniques
3. Alternative architectures
4. Real-world applications



**Physics-aware representation learning is the future of scientific AI.**

*Investigating the intersection of deep learning and physics for better scientific computing*
