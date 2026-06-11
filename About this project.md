# FEM Analysis of a Circular Tapered Bar Under Axial Load

## Overview

This project presents a **Finite Element Method (FEM) implementation** for analyzing the axial deformation of a **circular tapered bar** subjected to a tensile load. The program compares numerical FEM solutions with the exact analytical solution and evaluates convergence behavior as the number of finite elements increases.

The assignment was developed for **Finite Element Analysis (FEA)** coursework and demonstrates:

* Analytical derivation of displacement
* Direct stiffness method implementation
* Global stiffness matrix assembly
* Boundary condition application
* Numerical solution of nodal displacements
* Stress and internal force evaluation
* Mesh convergence analysis
* Advanced graphical visualization

---

## Problem Description

A circular tapered bar of length **L** is fixed at one end and subjected to an axial tensile load **P** at the free end.

The diameter varies linearly along the bar:

[
d(x)=d_1+\frac{(d_2-d_1)}{L}x
]

Cross-sectional area:

[
A(x)=\frac{\pi}{4}[d(x)]^2
]

The objective is to determine the free-end displacement using:

1. Exact analytical solution
2. Finite Element Method (FEM)

and compare the results for different mesh densities.

---

## Features

### Analytical Solution

* Closed-form displacement equation
* Exact free-end displacement calculation
* Exact displacement field along the bar

### FEM Solver

* Direct stiffness formulation
* Linear 1D bar elements
* Midpoint area approximation
* Automatic global stiffness assembly
* Fixed-end boundary condition implementation
* Numerical displacement solution

### Post-Processing

* Nodal displacements
* Element stresses
* Internal axial forces
* Support reaction calculation
* Percentage error estimation

### Convergence Study

* Supports any number of elements (**n ≥ 1**)
* Error tracking against analytical solution
* Convergence visualization

### Advanced Visualization

* Deformed shape representation
* Cross-sectional variation
* Stress distribution plots
* Element stiffness comparison
* Convergence curves

---

## Mathematical Formulation

### Element Stiffness Matrix

For each element:

[
k_e=\frac{EA_e}{l_e}
\begin{bmatrix}
1 & -1\
-1 & 1
\end{bmatrix}
]

where

[
A_e=\frac{\pi}{4}d_e^2
]

and (d_e) is evaluated at the element midpoint.

### Exact Free-End Displacement

The analytical solution yields:

[
\delta_{exact}
==============

\frac{4PL}
{\pi E d_1 d_2}
]

which serves as the benchmark for FEM validation.

---

## Project Structure

```text
.
├── fem_tapered_bar_ASS1.py
├── tapered_bar_fem_v2.png
└── README.md
```

---

## Requirements

### Python Packages

```bash
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Running the Program

### Assignment Demonstration Mode

```bash
python fem_tapered_bar_ASS1.py
```

or

```bash
python fem_tapered_bar_ASS1.py --demo
```

This mode:

* Uses predefined assignment values
* Computes analytical displacement
* Evaluates FEM results for multiple meshes
* Generates convergence tables
* Produces graphical outputs

---

### Interactive Mode

```bash
python fem_tapered_bar_ASS1.py
```

Select:

```text
[2] Interactive mode
```

Enter:

```text
Bar length L
Diameter d1
Diameter d2
Young's modulus E
Applied load P
Number of elements n
```

The program automatically performs the FEM analysis and displays results.

---

## Example Parameters

```text
Length (L)          = 1.0 m
Fixed Diameter (d1) = 50 mm
Free Diameter (d2)  = 25 mm
Young's Modulus (E) = 200 GPa
Load (P)            = 10 kN
```

---

## Example Output

### FEM vs Analytical Comparison

| Elements (n) | FEM Displacement | Exact Displacement | Error (%) |
| ------------ | ---------------- | ------------------ | --------- |
| 1            | Computed         | Exact              | Error     |
| 2            | Computed         | Exact              | Error     |
| 3            | Computed         | Exact              | Error     |

As the number of elements increases, the FEM solution converges toward the analytical solution.

---

## Visualization Dashboard

The program generates a comprehensive engineering dashboard containing:

### Panel A

Deformed shape of the tapered bar (exaggerated scale)

### Panel B

Cross-sectional diameter variation

### Panel C

Element stress distribution

### Panel D

Element stiffness comparison

### Panel E

Mesh convergence study

---

## Engineering Concepts Demonstrated

* Finite Element Method (FEM)
* Direct Stiffness Method
* Structural Mechanics
* Axial Deformation Analysis
* Numerical Methods
* Matrix Assembly Techniques
* Mesh Convergence Analysis
* Scientific Visualization

---

## Academic Context

This project was developed as part of:

**Finite Element Analysis (FEA)**

**M.Sc. Mechanical Engineering**
(Design and Manufacturing)

**Institute of Engineering (IOE)**
**Tribhuvan University, Nepal**

---

## Author

**Dixanta Parajuli**

M.Sc. Mechanical Engineering (Design & Manufacturing)

Thapathali Campus

Institute of Engineering (IOE)

Tribhuvan University

---

## Caution
There is an effective use of ai for coding for this assignment.Replit and Claude were the ai chat used for this coding.
