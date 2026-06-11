"""
╔══════════════════════════════════════════════════════════════════════╗
║   FEM Analysis — Circular Tapered Bar Under Axial Load  (v2)        ║
║   Architecture : Class-based solver + Visualizer                    ║
║   Author       : Assignment-I, FEA, MSc ME (Design & Mfg)          ║
║   Tribhuvan University, Institute of Engineering                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import LogLocator, LogFormatter
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.collections import PatchCollection
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BarGeometry:
    """Physical description of the tapered bar."""
    L:  float
    d1: float
    d2: float
    E:  float
    P:  float

    def diameter_at(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.d1 + (self.d2 - self.d1) * x / self.L

    def area_at(self, x: float | np.ndarray) -> float | np.ndarray:
        d = self.diameter_at(x)
        return (math.pi / 4.0) * d ** 2

    def analytical_displacement(self, x: float | np.ndarray) -> float | np.ndarray:
        """Closed-form displacement field u(x) under tip load P."""
        if abs(self.d2 - self.d1) < 1e-12:
            return self.P * x / (self.E * self.area_at(0.0))
        dx = self.diameter_at(x)
        return (4.0 * self.P * self.L) / (math.pi * self.E * (self.d2 - self.d1)) \
               * (1.0 / self.d1 - 1.0 / dx)

    def delta_exact(self) -> float:
        return float(self.analytical_displacement(self.L))


@dataclass
class FEMResults:
    """All outputs from a single FEM solve."""
    n:            int
    h:            float
    x_nodes:      np.ndarray
    d_nodes:      np.ndarray
    x_mid:        np.ndarray
    d_mid:        np.ndarray
    A_mid:        np.ndarray
    ke:           np.ndarray
    K_global:     np.ndarray
    u:            np.ndarray
    sigma:        np.ndarray
    force:        np.ndarray
    delta_fem:    float
    delta_exact:  float
    error_pct:    float
    reaction:     float


# ══════════════════════════════════════════════════════════════════════════════
#  SOLVER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class TaperedBarFEM:
    """
    Direct-stiffness FEM solver for a fixed-free circular tapered bar.

    Usage
    -----
    solver = TaperedBarFEM(geometry)
    result = solver.solve(n=3)
    """

    def __init__(self, geometry: BarGeometry) -> None:
        self.geo = geometry

    # ── element geometry ──────────────────────────────────────────────────────
    def _element_data(self, n: int) -> dict:
        g   = self.geo
        h   = g.L / n
        idx = np.arange(n + 1)
        x_nodes = idx * h
        d_nodes = g.diameter_at(x_nodes)

        e_idx = np.arange(1, n + 1)
        x_mid = (e_idx - 0.5) * h
        d_mid = g.diameter_at(x_mid)
        A_mid = (math.pi / 4.0) * d_mid ** 2
        ke    = g.E * A_mid / h
        return dict(h=h, x_nodes=x_nodes, d_nodes=d_nodes,
                    x_mid=x_mid, d_mid=d_mid, A_mid=A_mid, ke=ke)

    # ── global stiffness assembly ─────────────────────────────────────────────
    @staticmethod
    def _assemble(n: int, ke: np.ndarray) -> np.ndarray:
        K = np.zeros((n + 1, n + 1))
        for e in range(n):
            K[e:e+2, e:e+2] += ke[e] * np.array([[1, -1], [-1, 1]])
        return K

    # ── boundary conditions + solve ───────────────────────────────────────────
    @staticmethod
    def _solve(K: np.ndarray, P: float, n: int) -> np.ndarray:
        K_red = K[1:, 1:].copy()
        f_red = np.zeros(n)
        f_red[-1] = P
        u_free = np.linalg.solve(K_red, f_red)
        return np.concatenate(([0.0], u_free))

    # ── post-process ──────────────────────────────────────────────────────────
    @staticmethod
    def _stress_force(E: float, h: float,
                      u: np.ndarray, A_mid: np.ndarray):
        strain = (u[1:] - u[:-1]) / h
        sigma  = E * strain
        force  = sigma * A_mid
        return sigma, force

    # ── public API ────────────────────────────────────────────────────────────
    def solve(self, n: int) -> FEMResults:
        g   = self.geo
        ed  = self._element_data(n)
        K   = self._assemble(n, ed["ke"])
        u   = self._solve(K, g.P, n)
        sig, frc = self._stress_force(g.E, ed["h"], u, ed["A_mid"])
        d_fem    = float(u[-1])
        d_ex     = g.delta_exact()
        err      = abs(d_fem - d_ex) / d_ex * 100.0
        rxn      = float(-ed["ke"][0] * u[1])

        return FEMResults(
            n=n, h=ed["h"],
            x_nodes=ed["x_nodes"], d_nodes=ed["d_nodes"],
            x_mid=ed["x_mid"], d_mid=ed["d_mid"],
            A_mid=ed["A_mid"], ke=ed["ke"],
            K_global=K, u=u,
            sigma=sig, force=frc,
            delta_fem=d_fem, delta_exact=d_ex,
            error_pct=err, reaction=rxn,
        )

    def convergence_study(self, n_max: int = 50) -> tuple[list[int], list[float]]:
        ns     = list(range(1, n_max + 1))
        errors = [self.solve(ni).error_pct for ni in ns]
        return ns, errors


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_SEP  = "─" * 70
_DSEP = "═" * 70

def _hdr(title: str):
    print(f"\n{_DSEP}\n  {title}\n{_DSEP}")

def _tbl_row(*vals, widths, aligns):
    parts = []
    for v, w, a in zip(vals, widths, aligns):
        s = str(v)
        parts.append(s.rjust(w) if a=="r" else s.center(w) if a=="c" else s.ljust(w))
    print("  " + "  ".join(parts))


def print_full_report(r: FEMResults, geo: BarGeometry) -> None:
    n = r.n

    _hdr("ELEMENT PROPERTIES")
    _tbl_row("Elem","x_mid","d_mid","A_mid","kₑ","Nodes",
             widths=[5,12,12,14,18,8], aligns=["c","r","r","r","r","c"])
    print("  " + _SEP)
    for e in range(n):
        _tbl_row(e+1, f"{r.x_mid[e]:.5f}", f"{r.d_mid[e]:.6f}",
                 f"{r.A_mid[e]:.8f}", f"{r.ke[e]:.4f}", f"{e+1}↔{e+2}",
                 widths=[5,12,12,14,18,8], aligns=["c","r","r","r","r","c"])

    _hdr("GLOBAL STIFFNESS MATRIX  K")
    K = r.K_global
    w = max(14, max(len(f"{v:.2f}") for v in K.flat) + 2)
    print("     " + "".join(f"{'u'+str(i+1):>{w}}" for i in range(n+1)))
    print("  " + _SEP)
    for i in range(n+1):
        row = f"  f{i+1:<3}|"
        for j in range(n+1):
            row += f"{K[i,j]:>{w}.2f}"
        print(row)

    _hdr("NODAL DISPLACEMENTS")
    _tbl_row("Node","x","u (FEM)","u (Exact)","Diff",
             widths=[5,12,18,18,14], aligns=["c","r","r","r","r"])
    print("  " + _SEP)
    xs = r.x_nodes
    u_ex = geo.analytical_displacement(xs)
    for i in range(n+1):
        tag = "(fixed)" if i==0 else ("← FREE END" if i==n else "")
        diff = abs(r.u[i] - float(u_ex[i]))
        _tbl_row(i+1, f"{xs[i]:.4f}", f"{r.u[i]:.10f}",
                 f"{float(u_ex[i]):.10f}", f"{diff:.3e}",
                 widths=[5,12,18,18,14], aligns=["c","r","r","r","r"])
        if tag:
            print(f"{'':>57}  {tag}")

    _hdr("ELEMENT STRESSES & FORCES")
    _tbl_row("Elem","σ (Pa)","F (N)","A_mid (m²)",
             widths=[5,22,18,16], aligns=["c","r","r","r"])
    print("  " + _SEP)
    for e in range(n):
        _tbl_row(e+1, f"{r.sigma[e]:.6e}", f"{r.force[e]:.4f}",
                 f"{r.A_mid[e]:.8f}",
                 widths=[5,22,18,16], aligns=["c","r","r","r"])

    print(f"\n  Reaction at wall  : {r.reaction:.6f} N   (expected −P = {-geo.P:.6f} N)")

    _hdr("FEM vs ANALYTICAL")
    print(f"  δ_exact  =  4PL / (π E d1 d2)  =  {r.delta_exact:.10e} m")
    print(f"  δ_FEM    =  {r.delta_fem:.10e} m   (n = {n})")
    print(f"  Error    =  {r.error_pct:.6f} %")
    qual = ("✔  Excellent" if r.error_pct < 0.5
            else "✔  Good" if r.error_pct < 2.0
            else "⚠  Coarse mesh")
    print(f"\n  {qual}  (n={n})")


def print_convergence_table(solver: TaperedBarFEM, ns: list[int]) -> None:
    _hdr("CONVERGENCE TABLE")
    d_ex = solver.geo.delta_exact()
    _tbl_row("n","δ_FEM (m)","δ_FEM (×10⁻⁵)","Error (%)",
             widths=[6,22,18,14], aligns=["c","r","r","r"])
    print("  " + _SEP)
    for ni in ns:
        r = solver.solve(ni)
        _tbl_row(ni, f"{r.delta_fem:.10f}",
                 f"{r.delta_fem*1e5:.6f}", f"{r.error_pct:.5f}",
                 widths=[6,22,18,14], aligns=["c","r","r","r"])
    print("  " + _SEP)
    _tbl_row("Exact", f"{d_ex:.10f}", f"{d_ex*1e5:.6f}", "—",
             widths=[6,22,18,14], aligns=["c","r","r","r"])


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISER CLASS  — dark theme, 5-panel layout
# ══════════════════════════════════════════════════════════════════════════════

class FEMVisualiser:
    """
    Generates a 5-panel dark-theme figure:
      Panel A (top-left wide)  : Deformed shape (exaggerated) + original
      Panel B (top-right)      : Circular cross-section comparison (d1 vs d2)
      Panel C (mid-left wide)  : Element axial stress distribution (step + fill)
      Panel D (mid-right)      : Element stiffness — horizontal bar chart
      Panel E (bottom full)    : Convergence — dual y-axis (error% + δ_FEM)
    """

    # ── colour tokens ──────────────────────────────────────────────────────────
    C_CYAN    = "#00E5FF"
    C_AMBER   = "#FFB300"
    C_LIME    = "#76FF03"
    C_MAGENTA = "#FF4081"
    C_VIOLET  = "#CE93D8"
    C_ORANGE  = "#FF6D00"
    C_BG      = "#0D1117"
    C_AX      = "#161B22"
    C_GRID    = "#30363D"
    C_TEXT    = "#E6EDF3"
    C_MUTED   = "#8B949E"

    def __init__(self, geo: BarGeometry, result: FEMResults) -> None:
        self.geo = geo
        self.r   = result

    # ── helpers ───────────────────────────────────────────────────────────────
    def _style_ax(self, ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_facecolor(self.C_AX)
        ax.set_title(title, color=self.C_TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=self.C_MUTED, fontsize=8)
        ax.set_ylabel(ylabel, color=self.C_MUTED, fontsize=8)
        ax.tick_params(colors=self.C_MUTED, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.C_GRID)
        ax.grid(True, color=self.C_GRID, linewidth=0.5, alpha=0.6)

    # ── Panel A : deformed shape ───────────────────────────────────────────────
    def _panel_deformed(self, ax) -> None:
        self._style_ax(ax, "Deformed Shape  (displacement exaggerated)",
                       "x  (position along bar)", "y  (radius)")
        geo = self.geo
        r   = self.r
        L   = geo.L

        xs = np.linspace(0, L, 500)
        d_orig = geo.diameter_at(xs)

        # Scale factor for exaggeration
        max_u  = r.delta_fem
        scale  = 0.12 * L / max(max_u, 1e-30)

        # Original bar (muted)
        ax.fill_between(xs,  d_orig/2, -d_orig/2,
                        color=self.C_MUTED, alpha=0.18, linewidth=0)
        ax.plot(xs,  d_orig/2, color=self.C_MUTED, lw=1.0, ls=":", alpha=0.6)
        ax.plot(xs, -d_orig/2, color=self.C_MUTED, lw=1.0, ls=":", alpha=0.6)

        # Deformed bar: interpolate u onto fine grid
        u_fine    = np.interp(xs, r.x_nodes, r.u)
        xs_def    = xs + u_fine * scale       # x shifts due to axial deformation
        d_def_top = d_orig / 2
        d_def_bot = -d_orig / 2

        ax.fill_between(xs_def, d_def_top, d_def_bot,
                        color=self.C_CYAN, alpha=0.22, linewidth=0)
        ax.plot(xs_def,  d_def_top, color=self.C_CYAN, lw=2.0, label="Deformed")
        ax.plot(xs_def,  d_def_bot, color=self.C_CYAN, lw=2.0)

        # FEM node markers on deformed shape
        u_nodes = r.u
        xd_nodes = r.x_nodes + u_nodes * scale
        d_at_nodes = geo.diameter_at(r.x_nodes)
        ax.scatter(xd_nodes,  d_at_nodes/2, color=self.C_AMBER,
                   s=40, zorder=5, label="FEM nodes", edgecolors=self.C_BG, linewidths=0.6)
        ax.scatter(xd_nodes, -d_at_nodes/2, color=self.C_AMBER,
                   s=40, zorder=5, edgecolors=self.C_BG, linewidths=0.6)

        # Fixed wall
        ax.fill_betweenx([-geo.d1*0.9, geo.d1*0.9], -0.025*L, 0,
                         color="#455A64", alpha=0.7, zorder=3)
        ax.axvline(0, color=self.C_TEXT, lw=2.5, zorder=4)

        # Load arrow
        tip_x = xs_def[-1]
        ax.annotate("", xy=(tip_x + 0.08*L, 0), xytext=(tip_x, 0),
                    arrowprops=dict(arrowstyle="-|>", color=self.C_MAGENTA,
                                   lw=2.0, mutation_scale=16))
        ax.text(tip_x + 0.10*L, 0.005*geo.d1, "P",
                color=self.C_MAGENTA, fontsize=12, fontweight="bold", va="bottom")

        # Scale label
        ax.text(0.02*L, -geo.d1*0.78,
                f"Scale ×{scale:.0f}", color=self.C_MUTED, fontsize=7)

        muted_patch = mpatches.Patch(color=self.C_MUTED, alpha=0.5, label="Original")
        ax.legend(handles=[muted_patch,
                            mpatches.Patch(color=self.C_CYAN, alpha=0.5, label="Deformed"),
                            plt.Line2D([],[],marker="o",color="none",
                                       markerfacecolor=self.C_AMBER,ms=6,label="FEM nodes")],
                  facecolor=self.C_BG, edgecolor=self.C_GRID,
                  labelcolor=self.C_TEXT, fontsize=8, loc="upper left")
        ax.set_xlim(-0.07*L, L * 1.22)

    # ── Panel B : cross-section circles ───────────────────────────────────────
    def _panel_cross_sections(self, ax) -> None:
        self._style_ax(ax, "Cross-Sections (Fixed → Free end)")
        geo = self.geo
        r   = self.r
        ax.set_aspect("equal")
        ax.grid(False)

        # Draw n cross-sections spaced vertically
        n_show = min(r.n, 6)
        xs_show = np.linspace(0, geo.L, n_show + 2)[1:-1]
        diams   = geo.diameter_at(xs_show)
        max_d   = max(geo.d1, geo.d2)

        cmap  = matplotlib.colormaps["cool"]
        norms = (diams - diams.min()) / (diams.max() - diams.min() + 1e-30)

        spacing = max_d * 1.5
        for i, (x, d, nv) in enumerate(zip(xs_show, diams, norms)):
            cx = i * spacing
            circle = plt.Circle((cx, 0), d / 2,
                                 color=cmap(0.1 + 0.8 * nv), alpha=0.85, zorder=3)
            ax.add_patch(circle)
            # Dashed outline
            outer = plt.Circle((cx, 0), d / 2,
                                fill=False, edgecolor=self.C_TEXT,
                                lw=0.8, ls="--", alpha=0.5, zorder=4)
            ax.add_patch(outer)
            ax.text(cx, -d/2 - max_d*0.18,
                    f"x={x:.2f}\nd={d*1000:.1f}mm",
                    color=self.C_MUTED, fontsize=6.5, ha="center", va="top")

        ax.set_xlim(-max_d, (n_show - 1) * spacing + max_d)
        ax.set_ylim(-max_d * 1.1, max_d * 1.2)

        # Colourbar annotation (vertical — avoids stealing row height)
        sm = cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(diams.min()*1000, diams.max()*1000))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, orientation="vertical",
                          pad=0.04, fraction=0.06, shrink=0.75, aspect=18)
        cb.set_label("Diameter (mm)", color=self.C_MUTED, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, colors=self.C_MUTED)
        cb.outline.set_edgecolor(self.C_GRID)

    # ── Panel C : stress distribution ─────────────────────────────────────────
    def _panel_stress(self, ax) -> None:
        self._style_ax(ax, "Axial Stress Distribution  σ(x)",
                       "x  (position)", "σ  (Pa)")
        r   = self.r
        geo = self.geo

        # Step-function FEM stress
        for e in range(r.n):
            x0, x1 = r.x_nodes[e], r.x_nodes[e+1]
            sv      = float(r.sigma[e])
            # Normalised colour
            ratio   = abs(sv) / (max(abs(r.sigma)) + 1e-30)
            col     = cm.plasma(0.15 + 0.75 * ratio)
            ax.fill_between([x0, x1], [0, 0], [sv, sv],
                            color=col, alpha=0.65, linewidth=0, zorder=2)
            ax.hlines(sv, x0, x1, color=col, lw=2.5, zorder=3)
            ax.vlines([x0, x1], 0, sv, color=col, lw=0.8, ls=":", alpha=0.5, zorder=3)
            ax.text((x0+x1)/2, sv * 1.04, f"{sv:.2e}",
                    color=self.C_TEXT, fontsize=6, ha="center", va="bottom",
                    rotation=45 if r.n > 4 else 0)

        # Zero baseline
        ax.axhline(0, color=self.C_MUTED, lw=0.8, alpha=0.6)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cm.plasma,
                               norm=mcolors.Normalize(0, float(abs(r.sigma).max())))
        sm.set_array([])
        cb = ax.get_figure().colorbar(sm, ax=ax, shrink=0.8, pad=0.01, aspect=20)
        cb.set_label("|σ|  (Pa)", color=self.C_MUTED, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, colors=self.C_MUTED)
        cb.outline.set_edgecolor(self.C_GRID)

        ax.set_xlim(-0.02 * geo.L, geo.L * 1.02)

    # ── Panel D : stiffness horizontal bars ────────────────────────────────────
    def _panel_stiffness(self, ax) -> None:
        self._style_ax(ax, "Element Stiffness  kₑ",
                       "kₑ  (E·A/h)", "Element")
        r  = self.r
        ns = np.arange(1, r.n + 1)

        ke_norm = (r.ke - r.ke.min()) / (r.ke.max() - r.ke.min() + 1e-30)
        colors  = [matplotlib.colormaps["YlOrRd"](0.25 + 0.65 * v) for v in ke_norm]

        bars = ax.barh(ns, r.ke, color=colors, edgecolor=self.C_BG,
                       linewidth=0.6, height=0.7, zorder=3)
        for bar, kv in zip(bars, r.ke):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{kv:.3e}", va="center", ha="left",
                    color=self.C_TEXT, fontsize=6.5)

        ax.set_yticks(ns)
        ax.set_yticklabels([f"e{i}" for i in ns], fontsize=7.5)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=self.C_GRID, alpha=0.5, zorder=0)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cm.YlOrRd,
                               norm=mcolors.Normalize(float(r.ke.min()), float(r.ke.max())))
        sm.set_array([])
        cb = ax.get_figure().colorbar(sm, ax=ax, shrink=0.8, pad=0.02, aspect=20)
        cb.set_label("kₑ", color=self.C_MUTED, fontsize=7)
        cb.ax.tick_params(labelsize=6, colors=self.C_MUTED)
        cb.outline.set_edgecolor(self.C_GRID)

    # ── Panel E : convergence — dual y-axis ────────────────────────────────────
    def _panel_convergence(self, ax, solver: TaperedBarFEM) -> None:
        self._style_ax(ax, "Convergence Study",
                       "Number of elements  n", "Error  (%)")
        r   = self.r
        geo = self.geo

        ns, errors = solver.convergence_study(n_max=50)
        err_arr   = np.array(errors)
        ns_arr    = np.array(ns)

        # Right y-axis for δ_FEM
        ax2 = ax.twinx()
        ax2.set_facecolor(self.C_AX)
        ax2.tick_params(colors=self.C_MUTED, labelsize=7.5)
        ax2.set_ylabel("δ_FEM  (m)", color=self.C_LIME, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=self.C_LIME)
        for sp in ax2.spines.values():
            sp.set_edgecolor(self.C_GRID)

        # δ_FEM convergence
        delta_arr = np.array([solver.solve(ni).delta_fem for ni in ns])
        ax2.plot(ns_arr, delta_arr, color=self.C_LIME, lw=1.5, alpha=0.7,
                 ls="--", label="δ_FEM")
        ax2.axhline(geo.delta_exact(), color=self.C_LIME, lw=1.0, ls=":",
                    alpha=0.5, label=f"δ_exact = {geo.delta_exact():.4e}")

        # Error on log scale
        ax.set_yscale("log")
        ax.scatter(ns_arr, err_arr,
                   c=err_arr, cmap="RdYlGn_r",
                   norm=mcolors.LogNorm(vmin=max(err_arr.min(), 1e-5),
                                        vmax=err_arr.max()),
                   s=22, zorder=5, edgecolors=self.C_BG, linewidths=0.4)
        ax.plot(ns_arr, err_arr, color=self.C_CYAN, lw=1.6, alpha=0.85,
                zorder=4, label="Error (%)")

        # Mark current n
        ax.axvline(r.n, color=self.C_MAGENTA, lw=1.8, ls="--", zorder=6,
                   label=f"n = {r.n}")
        ax.scatter([r.n], [r.error_pct], color=self.C_MAGENTA,
                   s=80, zorder=7, edgecolors=self.C_TEXT, linewidths=0.8)
        ax.annotate(f" n={r.n}\n err={r.error_pct:.3f}%",
                    xy=(r.n, r.error_pct),
                    xytext=(r.n + 2, r.error_pct * 3),
                    arrowprops=dict(arrowstyle="->", color=self.C_MAGENTA, lw=1.0),
                    color=self.C_MAGENTA, fontsize=7.5)

        ax.set_xlim(0, 52)
        ax.set_ylim(max(err_arr.min() * 0.4, 1e-4), err_arr.max() * 3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  facecolor=self.C_BG, edgecolor=self.C_GRID,
                  labelcolor=self.C_TEXT, fontsize=7.5, loc="upper right")

    # ── main render ───────────────────────────────────────────────────────────
    def render(self, solver: TaperedBarFEM,
               save_path: Optional[str] = None) -> None:
        if not HAS_PLOT:
            print("  (matplotlib not available — skipping plots)")
            return

        plt.rcParams.update({
            "font.family":      "DejaVu Sans",
            "text.color":       self.C_TEXT,
            "axes.titlecolor":  self.C_TEXT,
            "axes.labelcolor":  self.C_MUTED,
            "figure.facecolor": self.C_BG,
            "axes.facecolor":   self.C_AX,
            "xtick.color":      self.C_MUTED,
            "ytick.color":      self.C_MUTED,
            "grid.color":       self.C_GRID,
            "legend.facecolor": self.C_BG,
            "legend.edgecolor": self.C_GRID,
        })

        fig = plt.figure(figsize=(19, 14))
        fig.patch.set_facecolor(self.C_BG)

        gs = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.42,
                      left=0.06, right=0.97, top=0.94, bottom=0.06)
        ax_def  = fig.add_subplot(gs[0, :3])   # deformed shape (wide)
        ax_cs   = fig.add_subplot(gs[0,  3])   # cross-sections
        ax_sig  = fig.add_subplot(gs[1, :3])   # stress distribution
        ax_ke   = fig.add_subplot(gs[1,  3])   # stiffness bars
        ax_conv = fig.add_subplot(gs[2, :])    # convergence (full width)

        self._panel_deformed(ax_def)
        self._panel_cross_sections(ax_cs)
        self._panel_stress(ax_sig)
        self._panel_stiffness(ax_ke)
        self._panel_convergence(ax_conv, solver)

        geo = self.geo
        r   = self.r
        fig.suptitle(
            f"FEM — Circular Tapered Bar   "
            f"d1={geo.d1*1000:.0f}mm → d2={geo.d2*1000:.0f}mm   "
            f"L={geo.L}m   E={geo.E:.0e}Pa   P={geo.P:.0f}N   "
            f"n={r.n} elements   Error={r.error_pct:.3f}%",
            fontsize=11, fontweight="bold", color=self.C_TEXT, y=0.98
        )

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight",
                        facecolor=self.C_BG)
            print(f"\n  ✔  Plot saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-LEVEL RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def assignment_demo() -> None:
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        ASSIGNMENT DEMO — Default Problem Values  (v2)             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    geo    = BarGeometry(L=1.0, d1=0.05, d2=0.025, E=200e9, P=10_000.0)
    solver = TaperedBarFEM(geo)

    print(f"\n  L={geo.L}m  d1={geo.d1*1000:.0f}mm  d2={geo.d2*1000:.0f}mm  "
          f"E={geo.E:.2e}Pa  P={geo.P:.0f}N")
    print(f"  δ_exact = {geo.delta_exact():.8e} m")

    # Quick n=1,2,3 table
    print_convergence_table(solver, [1, 2, 3])

    # Full detail for n=3
    r3 = solver.solve(3)
    print_full_report(r3, geo)

    # Extended convergence table
    print_convergence_table(solver, [1, 2, 3, 5, 10, 25, 50, 100])

    # Plot
    if HAS_PLOT:
        vis = FEMVisualiser(geo, r3)
        vis.render(solver, save_path="tapered_bar_fem_v2.png")

    print(f"\n{'═'*70}\n  Done.  ✔\n{'═'*70}\n")


def interactive_mode() -> None:
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   FEM — Circular Tapered Bar  v2  (Interactive Mode)              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    def get_float(prompt):
        while True:
            try:
                v = float(input(f"  {prompt}"))
                if v <= 0:
                    print("    Must be > 0.")
                    continue
                return v
            except ValueError:
                print("    Invalid number.")

    def get_int(prompt):
        while True:
            try:
                v = int(input(f"  {prompt}"))
                if v < 1:
                    print("    Must be ≥ 1.")
                    continue
                return v
            except ValueError:
                print("    Invalid integer.")

    geo    = BarGeometry(
        L  = get_float("Bar length           L  : "),
        d1 = get_float("Diameter at fixed end d1 : "),
        d2 = get_float("Diameter at free end  d2 : "),
        E  = get_float("Young's modulus       E  : "),
        P  = get_float("Axial load at tip     P  : "),
    )
    n      = get_int("Number of elements    n  : ")
    solver = TaperedBarFEM(geo)
    r      = solver.solve(n)
    print_full_report(r, geo)
    print_convergence_table(solver, [1, 2, 3, 5, 10, 25, 50, 100])

    if HAS_PLOT and input("\n  Generate plot? (y/n): ").strip().lower() == "y":
        vis = FEMVisualiser(geo, r)
        vis.render(solver, save_path="tapered_bar_fem_v2.png")

    print(f"\n{'═'*70}\n  Done.  ✔\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--demo" in sys.argv:
        assignment_demo()
    else:
        print("\n  Run mode:")
        print("    [1] Assignment demo  (default values)")
        print("    [2] Interactive mode")
        choice = input("\n  Choose (1/2) [default=1]: ").strip()
        if choice == "2":
            interactive_mode()
        else:
            assignment_demo()
