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
