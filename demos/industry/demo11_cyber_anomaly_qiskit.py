#!/usr/bin/env python
"""
Demo 11 — Cybersecurity Anomaly Detection (QSVM + HQNN)

This demo:
- Builds a synthetic network-traffic anomaly dataset (4 features)
- Trains:
    1. Quantum Kernel SVM (FidelityQuantumKernel)
    2. Hybrid HQNN (Qiskit Aer + SPSA)
    3. Classical Logistic Regression baseline
- Compares ROC curves and accuracy
- Saves JSON + PNG for thesis integration

Outputs:
- results/demo11/results_demo11_cyber.json
- results/demo11/cyber_roc_demo11.png
- results/demo11/cyber_accuracy_demo11.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# Synthetic Cyber Dataset
# ============================================================

def generate_cyber_data():
    """
    4 features (synthetic but cyber-relevant):
    - packet interval variance
    - byte entropy
    - port randomness
    - flag sequence irregularity
    """
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=11
    )
    return X, y


# ============================================================
# Quantum Kernel Feature Map (PARAMETERIZED, 4 features)
# ============================================================

def feature_map():
    """
    Parameterized 2-qubit feature map for 4 classical features.
    Required by your Qiskit ML version: num_parameters == n_features.
    """
    x = ParameterVector("x", 4)  # 4 parameters for 4 features
    qc = QuantumCircuit(2)

    # Embed the features via RY rotations
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)
    qc.ry(x[2], 0)
    qc.ry(x[3], 1)

    qc.cx(0, 1)
    return qc


# ============================================================
# HQNN Components
# ============================================================

def build_feature_map_hqnn(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc

def build_var_layer_hqnn(num_qubits, w):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(w[i], i)
        qc.rz(w[num_qubits+i], i)
    for i in range(num_qubits):
        qc.cz(i, (i+1)%num_qubits)
    return qc

def build_hqnn_circuit(num_qubits, x, w):
    fm = build_feature_map_hqnn(num_qubits, x)
    var = build_var_layer_hqnn(num_qubits, w)
    qc = fm.compose(var)
    qc.measure_all()
    return qc

def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0
    for bits, c in counts.items():
        parity = bits.count("1") % 2
        sign = 1 if parity==0 else -1
        exp += sign * c/shots
    return exp

def predict_prob_hqnn(sim, num_qubits, w, x):
    x_pad = np.zeros(num_qubits)
    x_pad[:len(x)] = x
    qc = build_hqnn_circuit(num_qubits, x_pad, w)
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()
    exp = parity_expval(counts)
    return (1-exp)/2


# ============================================================
# SPSA for HQNN training
# ============================================================

def loss_fn(sim, num_qubits, w, X, y):
    eps=1e-10
    preds = np.array([predict_prob_hqnn(sim,num_qubits,w,x) for x in X])
    return float(-np.mean(y*np.log(preds+eps)+(1-y)*np.log(1-preds+eps)))

def spsa_step(sim, num_qubits, w, X, y, alpha=0.15, c=0.15):
    dim=len(w)
    delta = 2*np.random.randint(0,2,dim)-1
    wplus  = w + c*delta
    wminus = w - c*delta
    loss_p = loss_fn(sim,num_qubits,wplus,X,y)
    loss_m = loss_fn(sim,num_qubits,wminus,X,y)
    ghat = (loss_p-loss_m)/(2*c*delta)
    return w - alpha*ghat


# ============================================================
# MAIN DEMO
# ============================================================

def run_demo(output_dir, epochs=10):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    X, y = generate_cyber_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    # --------------------------------------------------------
    # 2. Quantum Kernel SVM (QSVM)
    # --------------------------------------------------------
    fm = feature_map()  # PARAMETERIZED feature map with 4 parameters
    kernel = FidelityQuantumKernel(feature_map=fm)
    print("[QSVM] Computing quantum kernel...")

    K_train = kernel.evaluate(X_train, X_train)
    K_test  = kernel.evaluate(X_test, X_train)

    qsvm = SVC(kernel="precomputed")
    qsvm.fit(K_train, y_train)
    y_pred_qsvm = qsvm.predict(K_test)

    qsvm_acc = accuracy_score(y_test,y_pred_qsvm)
    scores = qsvm.decision_function(K_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    qsvm_auc = auc(fpr,tpr)

    # --------------------------------------------------------
    # 3. HQNN Training (SPSA)
    # --------------------------------------------------------
    num_qubits=4
    w = np.random.uniform(-np.pi,np.pi,2*num_qubits)
    sim = AerSimulator()

    for ep in range(epochs):
        w = spsa_step(sim,num_qubits,w,X_train,y_train)
        print(f"[HQNN][Epoch {ep}] Loss={loss_fn(sim,num_qubits,w,X_train,y_train):.4f}")

    preds_hqnn = np.array([predict_prob_hqnn(sim,num_qubits,w,x) for x in X_test])
    y_pred_hqnn = (preds_hqnn>=0.5).astype(int)
    hqnn_acc = accuracy_score(y_test,y_pred_hqnn)

    # --------------------------------------------------------
    # 4. Classical baseline
    # --------------------------------------------------------
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train,y_train)
    y_pred_cl = clf.predict(X_test)
    cl_acc = accuracy_score(y_test,y_pred_cl)

    # --------------------------------------------------------
    # 5. Save JSON
    # --------------------------------------------------------
    summary={
        "qsvm_accuracy": float(qsvm_acc),
        "qsvm_auc": float(qsvm_auc),
        "hqnn_accuracy": float(hqnn_acc),
        "classical_accuracy": float(cl_acc),
        "epochs": epochs
    }

    json_path=os.path.join(output_dir,"results_demo11_cyber.json")
    with open(json_path,"w") as f: json.dump(summary,f,indent=2)

    # --------------------------------------------------------
    # 6. Save ROC plot for QSVM
    # --------------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(fpr,tpr,label=f"QSVM (AUC={qsvm_auc:.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Demo 11 — QSVM ROC Curve (Cyber Anomaly Detection)")
    plt.grid(True)
    plt.legend()
    roc_path=os.path.join(output_dir,"cyber_roc_demo11.png")
    plt.savefig(roc_path)
    plt.close()

    # --------------------------------------------------------
    # 7. Accuracy comparison bar chart
    # --------------------------------------------------------
    labels=["QSVM","HQNN","Classical"]
    accs=[qsvm_acc,hqnn_acc,cl_acc]

    plt.figure(figsize=(7,5))
    plt.bar(labels,accs,color=["purple","green","gray"])
    plt.ylim(0,1)
    plt.title("Demo 11 — Cybersecurity Anomaly Detection Accuracy")
    plt.tight_layout()
    acc_path=os.path.join(output_dir,"cyber_accuracy_demo11.png")
    plt.savefig(acc_path)
    plt.close()

    print("\n===== DEMO 11 SUMMARY =====")
    print(summary)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved ROC Plot: {roc_path}")
    print(f"Saved Accuracy Plot: {acc_path}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",default="results/demo11")
    parser.add_argument("--epochs",type=int,default=10)
    args = parser.parse_args()
    run_demo(args.output_dir,epochs=args.epochs)
