#!/usr/bin/env python
"""
Demo 04 — QSVM Anomaly Detection (Legacy Qiskit ML Kernel API)

This version:
- Uses a PARAMETERIZED quantum feature map (REQUIRED for your version)
- Uses FidelityQuantumKernel with NO arguments except feature_map
- Fully compatible with older Qiskit ML 0.5.x (your environment)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
def generate_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=1.6,
        random_state=42,
    )
    return X, y


# ------------------------------------------------------------
# PARAMETERIZED FEATURE MAP  (THIS FIXES YOUR VERSION)
# ------------------------------------------------------------
def build_feature_map():
    """A 2-parameter circuit required by older FidelityQuantumKernel."""
    x0 = Parameter("x0")
    x1 = Parameter("x1")

    qc = QuantumCircuit(2)

    qc.ry(x0, 0)
    qc.ry(x1, 1)
    qc.cx(0, 1)

    return qc


# ------------------------------------------------------------
# QSVM runner
# ------------------------------------------------------------
def run_qsvm(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )

    # Build PARAMETERIZED feature map
    fm = build_feature_map()

    # Build kernel (legacy mode)
    print("[Demo04] Creating Legacy FidelityQuantumKernel...")
    kernel = FidelityQuantumKernel(feature_map=fm)

    # Kernel matrices
    print("[Demo04] Evaluating Kernel...")
    K_train = kernel.evaluate(X_train, X_train)
    K_test = kernel.evaluate(X_test, X_train)

    # Train classical SVM
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    accuracy = accuracy_score(y_test, y_pred)

    # ROC / AUC
    scores = clf.decision_function(K_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    # Save results
    json_path = os.path.join(output_dir, "results_demo04_qsvm.json")
    with open(json_path, "w") as f:
        json.dump(
            {"accuracy": float(accuracy), "auc": float(roc_auc),
             "fpr": fpr.tolist(), "tpr": tpr.tolist()},
            f, indent=2)

    # Plot ROC
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Demo 04 — Legacy QSVM Anomaly Detection")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "roc_demo04.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[Demo04] Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
    print(f"Saved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/demo04")
    args = parser.parse_args()

    run_qsvm(args.output_dir)
