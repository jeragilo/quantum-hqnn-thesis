Demo 01 — HQNN Toy Classifier (Qiskit)

Purpose:
Introduce the simplest hybrid quantum neural network (HQNN) pipeline.
Demo 01 acts as the baseline demonstration that your later demos expand on.
It shows how a shallow variational circuit + classical optimizer can classify a toy dataset.

Dataset:

2D synthetic dataset (two classes)

200 total samples (100 per class)

Standard train/test split (80/20)

Quantum Component:

Feature Map: RY encoding of each classical feature into a qubit

Variational Layer: RX + RZ rotation block

Entanglement: CNOT (0 → 1)

Measurement: parity expectation ⟨Z₀Z₁⟩ mapped to class probability

Optimizer:

COBYLA or SPSA (depending on run)

Learning rate ~0.2

40–60 optimization iterations

Outputs:

results/demo01/results_demo01_hqnn.json

results/demo01/accuracy_demo01.png

results/demo01/decision_boundary_demo01.png (optional)

Results:

Accuracy ranges between 0.45–0.60, depending on initialization

Demonstrates:

hybrid loop behavior

sampling noise

optimizer fluctuation

shallow circuit expressiveness limits

Significance:

Establishes the baseline HQNN architecture used in later demos

Connects directly to the early chapters of your thesis

Provides the simplest controlled environment to examine hybrid workflows

Used in Chapter 6 (Demonstration Ecosystem) as the “introductory hybrid model”

Helps motivate why more advanced demos (03–13) are necessary

## Demo 02 — VQE Toy Energy Minimization (PennyLane)

Framework: PennyLane

Hamiltonian: H = –1.05 Z₀ – 0.39 Z₁ + 0.1 X₀X₁

Ansatz: Two RY rotations + one CNOT

Optimizer: GradientDescentOptimizer (step 0.4)

Iterations: 80

Outputs:

results/demo02/results_demo02_vqe.json

results/demo02/energy_convergence_demo02.png

Purpose:
Implements a core VQE loop to approximate ground-state energy of a toy H₂-like Hamiltonian. Demonstrates hybrid optimization and energy convergence behavior.

Demo 03 — QAOA MaxCut (Cirq)

Framework: Cirq

Graph: 5-node ring

Algorithm: QAOA (p=1)

Mixer: RX on each qubit

Optimizer: Finite-difference gradient ascent

Iterations: 15

Outputs:

results/demo03/results_demo03_qaoa.json

results/demo03/cost_convergence_demo03.png

Purpose:
Shows how QAOA solves combinatorial optimization problems such as MaxCut. Supports the Variational Algorithms and Hybrid Optimization chapter.

Demo 04 — QSVM Anomaly Detection (Qiskit ML)

Dataset: 300 samples, 2D synthetic anomaly dataset

Quantum Kernel: FidelityQuantumKernel

Feature Map: RY-based 2-parameter entangled circuit

SVM: Classical SVC (precomputed kernel)

Metrics:

Accuracy: 1.000

AUC: 1.000

Outputs:

results/demo04/results_demo04_qsvm.json

results/demo04/roc_demo04.png

Purpose:
Demonstrates quantum kernel–based anomaly detection. Validation for QML chapter and hybrid reliability.

Demo 05 — Noise-Robust HQNN (Qiskit)

Tracks:

Noise-free HQNN

Noisy HQNN (depolarizing p=0.05)

HQNN + ZNE (if available)

Classical baseline (MLP)

Results:

Noiseless HQNN: 0.50

Noisy HQNN: 0.55

Classical baseline: 0.883

Outputs:

results/demo05/results_demo05.json

results/demo05/accuracy_demo05.png

Purpose:
Shows fragility of HQNN vs classical model. Supports thesis motivation for hybrid reliability.

Demo 06 — Cross-Framework Noise Benchmark (Qiskit, Cirq, PennyLane)

Circuit: 2-qubit RY → CNOT parity circuit

Frameworks: Qiskit, Cirq, PennyLane

Noise: depolarizing, amplitude damping

Results:

Qiskit: noticeable drift

Cirq: monotonic degradation

PennyLane: zero drift

Outputs:

results/demo06/results_demo06.json

results/demo06/parity_demo06.png

Significance:
Shows simulator discrepancies and hybrid reliability challenges. Crucial for cross-framework validation sections.

Demo 07 — Cross-Platform Parity Consistency

Circuit: H → RY(θ) → CNOT

Angles: θ = π/4

Expectation values:
⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₀Z₁⟩, ⟨X₀X₁⟩

Findings:

ZZ and XX match across all frameworks

Z₀, Z₁ differ due to endianness

Qiskit: little-endian

Cirq/PennyLane: big-endian

Outputs:

results/demo07/results_demo07.json

results/demo07/parity_demo07.png

Purpose:
Shows critical indexing differences across frameworks.

Demo 08 — Hybrid HQNN Training Loop (Qiskit + SPSA)

Feature Map: RY

Variational Layer: RX/RZ

Optimizer: SPSA

Loss: Binary cross-entropy

Results:

Loss: ~0.69

Accuracy: ~0.35–0.52

Outputs:

results/demo08/results_demo08_training.json

results/demo08/training_curves_demo08.png

Significance:
Core hybrid training loop evidence for thesis.

Demo 09 — Medical Risk Classification (HQNN vs Classical)

Dataset: Synthetic 4D medical features

Models: HQNN vs Logistic Regression

Optimizer: SPSA

Results:

HQNN: ~0.49

Classical: ~0.99

Outputs:

results/demo09/results_demo09_medical.json

results/demo09/accuracy_demo09.png

Interpretation:
HQNN struggles on linearly separable data → supports thesis reliability claims.

Demo 10 — Energy Grid Optimization (Cirq QAOA)

Task: Microgrid unit-commitment / cost-minimization

Graph: 4-node weighted grid

Algorithm: QAOA p=1

Results:

Classical optimum: 7.0

QAOA estimate: ~4.53

Outputs:

results/demo10/results_demo10_energy.json

results/demo10/qaoa_energy_plot.png

Demo 11 — Cybersecurity Anomaly Detection (QSVM + HQNN)

Dataset: Synthetic 4-feature network traffic

Results:

QSVM: 0.844 (AUC: 0.906)

HQNN: 0.544

Classical baseline: 0.978

Outputs:

results/demo11/results_demo11_cyber.json

results/demo11/cyber_roc_demo11.png

results/demo11/cyber_accuracy_demo11.png

Demo 12 — HQNN Explainability & Trustworthiness

Metrics: Parameter sensitivity, feature importance, stability

Results:

Parameter sensitivity non-uniform

Feature 0 dominant

Stability is non-monotonic

Outputs:

results/demo12/results_demo12_explainability.json

results/demo12/parameter_sensitivity_demo12.png

results/demo12/feature_sensitivity_demo12.png

results/demo12/stability_demo12.png

Demo 13 — Cross-Noise Robustness Heatmap (Qiskit, Cirq, PennyLane)

Noise levels: 0, 0.02, 0.05, 0.10, 0.20

Circuits: Parity circuits

Noise Behavior:

Qiskit: non-monotonic

Cirq: monotonic

PennyLane: flat

Noise Matrix:
Qiskit: [0.0187, -0.0020, 0.0073, 0.0253, -0.0180]
Cirq: [0.0000, -0.0120, -0.0227, 0.0040, -0.0180]
PennyLane: ~0.0 across all noise

Outputs:

noise_matrix_demo13.json

noise_heatmap_demo13.png

noise_curves_demo13.png
