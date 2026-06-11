"""
FEM SOLVER — Circular Tapered Bar Under Axial Load
====================================================
Fixed at left end, axial force P at free end.
Diameter varies linearly from d1 (fixed) to d2 (free).

Assignment covers:
  Part 1 — Analytical solution (closed form)
  Parts 2-4 — FEM solution for any number of elements n
"""

import numpy as np

# ─────────────────────────────────────────────
# PARAMETERS — change these as needed
# ─────────────────────────────────────────────
L  = 1.0      # Length of bar (m)
P  = 10000    # Axial load at free end (N)
d1 = 0.05     # Diameter at fixed end (m)
d2 = 0.03     # Diameter at free end (m)
E  = 200e9    # Young's modulus (Pa) — 200 GPa for steel


# ─────────────────────────────────────────────
# PART 1 — ANALYTICAL SOLUTION
# ─────────────────────────────────────────────
# Derived by integrating: du = P dx / [E * A(x)]
# Result: delta = 4PL / (pi * E * d1 * d2)

def analytical_solution(L, P, d1, d2, E):
    return (4 * P * L) / (np.pi * E * d1 * d2)


# ─────────────────────────────────────────────
# PART 2, 3, 4 — FEM SOLUTION (generalised for any n)
# ─────────────────────────────────────────────

def fem_solver(L, P, d1, d2, E, n):
    """
    Solves the tapered bar problem using FEM.

    Steps:
      1. Divide bar into n equal elements
      2. For each element, find diameter at its midpoint
      3. Compute element stiffness matrix ke
      4. Assemble all ke into global stiffness matrix K
      5. Apply boundary condition: fix left end (u = 0)
      6. Apply load P at right end
      7. Solve for displacements
    """

    le = L / n          # Length of each element (all equal)
    ndof = n + 1        # Number of nodes = number of elements + 1

    # Step 1 — Create empty global stiffness matrix (ndof x ndof)
    K = np.zeros((ndof, ndof))

    # Step 2 & 3 — Loop over each element
    for i in range(n):

        # Midpoint x-coordinate of element i
        x_mid = (i + 0.5) * le

        # Diameter at midpoint using linear variation formula
        # d(x) = d1 + (d2 - d1) / L * x
        de = d1 + (d2 - d1) / L * x_mid

        # Cross-sectional area at midpoint
        Ae = (np.pi / 4) * de**2

        # Element stiffness matrix (2x2)
        # ke = (E * Ae / le) * [[1, -1], [-1, 1]]
        ke = (E * Ae / le) * np.array([[ 1, -1],
                                        [-1,  1]])

        # Step 4 — Assemble ke into global K
        # Element i connects node i and node i+1
        K[i,   i  ] += ke[0, 0]
        K[i,   i+1] += ke[0, 1]
        K[i+1, i  ] += ke[1, 0]
        K[i+1, i+1] += ke[1, 1]

    # Step 5 — Apply boundary condition: node 0 is fixed (u0 = 0)
    # Remove row 0 and column 0 from K
    K_reduced = K[1:, 1:]          # shape: (n x n)

    # Step 6 — Force vector: P applied at last node (free end)
    F_reduced = np.zeros(n)
    F_reduced[-1] = P

    # Step 7 — Solve K_reduced * u = F_reduced
    u = np.linalg.solve(K_reduced, F_reduced)

    # Free-end displacement is the last entry
    delta_fem = u[-1]
    return delta_fem


# ─────────────────────────────────────────────
# RESULTS — Print comparison table
# ─────────────────────────────────────────────

delta_exact = analytical_solution(L, P, d1, d2, E)

print("=" * 60)
print("  CIRCULAR TAPERED BAR — FEM vs ANALYTICAL")
print("=" * 60)
print(f"  Parameters: L={L} m, P={P} N, d1={d1} m, d2={d2} m, E={E:.0e} Pa")
print(f"  Analytical solution: delta = 4PL / (pi*E*d1*d2)")
print(f"  Analytical result  : {delta_exact * 1e3:.6f} mm")
print("=" * 60)
print(f"  {'n':>4}  {'FEM delta (mm)':>16}  {'Exact (mm)':>12}  {'Error (%)':>10}")
print("-" * 60)

for n in [1, 2, 3, 5, 10, 20, 50, 100]:
    delta_fem = fem_solver(L, P, d1, d2, E, n)
    error = abs(delta_fem - delta_exact) / delta_exact * 100
    print(f"  {n:>4}  {delta_fem * 1e3:>16.6f}  {delta_exact * 1e3:>12.6f}  {error:>10.5f}")

print("=" * 60)
print("\nObservation: Error decreases as n increases (convergence).")
print("Even n=3 gives a very good approximation.")
