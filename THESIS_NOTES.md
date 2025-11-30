## Demo 02 — VQE Toy Energy Minimization (PennyLane)

- **Framework:** PennyLane
- **Hamiltonian:** H = -1.05 Z₀ - 0.39 Z₁ + 0.1 X₀X₁ (toy H₂-like system)
- **Ansatz:** Two RY rotations + CNOT
- **Optimizer:** GradientDescentOptimizer, step size 0.4
- **Iterations:** 80
- **Outputs:**
  - `results/demo02/results_demo02_vqe.json`
  - `results/demo02/energy_convergence_demo02.png`
- **Purpose:** Demonstrates a core variational quantum algorithm (VQE).
  Shows how hybrid optimization can approximate the ground-state energy.

## Demo 03 — QAOA MaxCut (Cirq)

**Framework:** Cirq  
**Graph:** 5-node ring  
**Algorithm:** QAOA with 1 layer, gradient ascent  
**Hamiltonian:** ZZ terms for each edge  
**Mixer:** RX on each qubit  
**Optimizer:** Finite-difference gradient ascent  
**Iterations:** 15  
**Outputs:**
- `results/demo03/results_demo03_qaoa.json`
- `results/demo03/cost_convergence_demo03.png`

**Purpose:**  
Demonstrates how QAOA solves combinatorial optimization problems such as MaxCut.  
This will appear in the Variational Algorithms chapter and hybrid quantum-classical optimization section.

## Demo 04 — QSVM Anomaly Detection (Qiskit Machine Learning)

- **Dataset:** Synthetic anomaly detection (2D features, 300 samples)
- **Quantum Component:** Fidelity-based Quantum Kernel (FidelityQuantumKernel)
- **Feature Map:** 2-parameter RY-based circuit with entanglement
- **SVM:** Classical SVC with precomputed quantum kernel
- **Metrics:**
  - Accuracy: 1.000
  - AUC: 1.000
- **Outputs:**
  - results/demo04/results_demo04_qsvm.json
  - results/demo04/roc_demo04.png
- **Purpose:** Demonstrates quantum-enhanced anomaly detection using kernel
  methods. Fully compatible with legacy Qiskit ML, with parameterized feature
  map matching classical feature dimension.

## Demo 05 — Noise-Robust HQNN (Qiskit)

- **Purpose:**  
  Evaluate hybrid quantum neural network robustness under noise and compare against 
  classical baseline. This supports the thesis claim that hybrid architectures can 
  maintain meaningful performance despite noise.

- **Tracks:**
  - Noise-free HQNN  
  - Noisy HQNN (depolarizing noise p=0.05)  
  - HQNN + ZNE (if available)  
  - Classical baseline (MLP)

- **Results:**
  - Noiseless HQNN accuracy: 0.50  
  - Noisy HQNN accuracy: 0.55  
  - ZNE: Not available  
  - Classical baseline accuracy: 0.883  

- **Outputs:**
  - `results/demo05/results_demo05.json`
  - `results/demo05/accuracy_demo05.png`

- **Thesis Relevance:**  
  Demonstrates the fragility of quantum models and the effectiveness of classical baselines. 
  Sets the stage for Demo 06 & 07, which introduce cross-framework robustness and 
  mitigation techniques.
