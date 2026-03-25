"""
Parallel-plate waveguide with superconducting walls — test geometry for quasi-TEM solver.

Structure (cross-section in y):

        PEC (top wall)     TAG_BC_PEC_TOP
    +----------------------------+  y = d_gap + t_sc
    |       superconductor       |  TAG_metal
    +----------------------------+  y = d_gap     TAG_BC_INTERFACE_TOP
    |                            |
    |     vacuum gap (width d)   |  TAG_DOMAIN
    |                            |
    +----------------------------+  y = 0         TAG_BC_INTERFACE_BOT
    |       superconductor       |  TAG_metal
    +----------------------------+  y = -t_sc
        PEC (bottom wall)  TAG_BC_PEC_BOT

Left/right: PMC (natural BC) — infinite extent in x.

Analytic solutions exist for both the electrostatic and
London-magnetostatic problems, enabling direct solver verification.
"""

import numpy as np
from scipy.optimize import brentq
from mpi4py import MPI
import gmsh
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
import matplotlib as mpl
import matplotlib.pyplot as plt


class ParallelPlateSC:
    """
    Parallel-plate SC waveguide test geometry.

    All lengths are in consistent units (user chooses).
    """

    def __init__(self, d_gap=1.0, t_sc=0.5, width=0.5, lambda_L=0.05):
        self.d_gap = d_gap
        self.t_sc = t_sc
        self.width = width
        self.lambda_L = lambda_L

        # Cell tags
        self.TAG_DOMAIN = 1       # vacuum gap
        self.TAG_metal = 2        # superconductor
        self.TAG_dielectric = 3   # unused (but solver may check for it)

        # Facet tags
        self.TAG_BC_PEC_TOP = 10      # top PEC wall (y = d + t)
        self.TAG_BC_PEC_BOT = 11      # bottom PEC wall (y = -t)
        self.TAG_BC_INTERFACE_TOP = 12   # SC-vacuum interface at y = d
        self.TAG_BC_INTERFACE_BOT = 13   # SC-vacuum interface at y = 0
        self.TAG_BC_PMC_LEFT = 14     # left (x=0), natural BC
        self.TAG_BC_PMC_RIGHT = 15    # right (x=W), natural BC

        # Aliases for the quasi-TEM solver:
        # In PEC mode: center = top interface, ground = bottom interface
        self.TAG_BC_CENTER = self.TAG_BC_INTERFACE_TOP
        self.TAG_BC_GND = self.TAG_BC_INTERFACE_BOT
        # In SC mode: outer = PEC walls
        self.TAG_BC_OUTER = self.TAG_BC_PEC_BOT  # A_z = 0 here

    def build_geometry(self):
        d, t, W = self.d_gap, self.t_sc, self.width
        eps = 1e-6

        # Safety cleanup if gmsh was left open by a previous crash
        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        gmsh.model.add("ParallelPlateSC")
        occ = gmsh.model.occ

        lower_sc = occ.addRectangle(0, -t, 0, W, t)
        gap = occ.addRectangle(0, 0, 0, W, d)
        upper_sc = occ.addRectangle(0, d, 0, W, t)

        objects = [lower_sc, gap, upper_sc]
        frag, mapping = occ.fragment([(2, obj) for obj in objects], [])
        occ.synchronize()

        lower_sc_tags = set(tag for dim, tag in mapping[0] if dim == 2)
        gap_tags = set(tag for dim, tag in mapping[1] if dim == 2)
        upper_sc_tags = set(tag for dim, tag in mapping[2] if dim == 2)
        sc_tags = lower_sc_tags | upper_sc_tags

        gmsh.model.addPhysicalGroup(2, list(gap_tags), self.TAG_DOMAIN)
        gmsh.model.addPhysicalGroup(2, list(sc_tags), self.TAG_metal)

        # --- Classify boundary edges ---
        # IMPORTANT: getBoundary(frag) only returns EXTERNAL boundary of the union.
        # Internal interfaces (y=0, y=d) are shared edges and won't appear there.
        # Instead, get boundary of the GAP surface to find interface edges,
        # and boundary of the full union for PEC/PMC edges.

        # External boundary (PEC + PMC edges)
        outer_edges = gmsh.model.getBoundary(frag, oriented=False)

        # Interface edges: boundary of the gap surface that aren't on the outer boundary
        gap_boundary = gmsh.model.getBoundary(
            [(2, t) for t in gap_tags], oriented=False)

        outer_edge_tags = set(abs(tag) for dim, tag in outer_edges if dim == 1)
        gap_edge_tags = set(abs(tag) for dim, tag in gap_boundary if dim == 1)
        # Interface edges = gap boundary edges that are NOT on the outer boundary
        interface_edge_tags = gap_edge_tags - outer_edge_tags

        pec_top, pec_bot = [], []
        iface_top, iface_bot = [], []
        pmc_left, pmc_right = [], []

        # Classify outer boundary edges
        for dim_e, tag_e in outer_edges:
            if dim_e != 1:
                continue
            tag_e = abs(tag_e)
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag_e)
            ymid = 0.5 * (ymin + ymax)
            xmid = 0.5 * (xmin + xmax)
            is_horizontal = abs(ymax - ymin) < eps

            if is_horizontal:
                if abs(ymid - (d + t)) < eps:
                    pec_top.append(tag_e)
                elif abs(ymid - (-t)) < eps:
                    pec_bot.append(tag_e)
            else:
                if abs(xmid) < eps:
                    pmc_left.append(tag_e)
                elif abs(xmid - W) < eps:
                    pmc_right.append(tag_e)

        # Classify interface edges
        for tag_e in interface_edge_tags:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag_e)
            ymid = 0.5 * (ymin + ymax)
            if abs(ymid - d) < eps:
                iface_top.append(tag_e)
            elif abs(ymid - 0) < eps:
                iface_bot.append(tag_e)

        gmsh.model.addPhysicalGroup(1, pec_top, self.TAG_BC_PEC_TOP)
        gmsh.model.addPhysicalGroup(1, pec_bot, self.TAG_BC_PEC_BOT)
        if iface_top:
            gmsh.model.addPhysicalGroup(1, iface_top, self.TAG_BC_INTERFACE_TOP)
        if iface_bot:
            gmsh.model.addPhysicalGroup(1, iface_bot, self.TAG_BC_INTERFACE_BOT)
        # PMC: natural BC, no Dirichlet needed, but tag for reference
        if pmc_left:
            gmsh.model.addPhysicalGroup(1, pmc_left, self.TAG_BC_PMC_LEFT)
        if pmc_right:
            gmsh.model.addPhysicalGroup(1, pmc_right, self.TAG_BC_PMC_RIGHT)

        print(f"  Facets: {len(pec_top)} PEC-top, {len(pec_bot)} PEC-bot, "
              f"{len(iface_top)} iface-top, {len(iface_bot)} iface-bot")

        return self

    def generate_mesh(self, refinement_factor=3, bulk_factor=10):
        d, t, lam = self.d_gap, self.t_sc, self.lambda_L
        h_min = lam / refinement_factor
        h_max = d / bulk_factor

        field = gmsh.model.mesh.field
        field.add("MathEval", 1)
        transition = 5 * lam
        # Refine near ALL boundaries with exponential field structure:
        # SC-vacuum interfaces at y=0 and y=d, AND PEC walls at y=-t and y=d+t
        field.setString(
            1, "F",
            f"{h_min} + ({h_max} - {h_min}) * "
            f"Min(1.0, Min(Min(Abs(y), Abs(y - {d})), "
            f"Min(Abs(y + {t}), Abs(y - {d} - {t}))) / {transition})"
        )
        field.setAsBackgroundMesh(1)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.model.mesh.generate(2)

        _, elemTags2, _ = gmsh.model.mesh.getElements(dim=2)
        n_elem = sum(len(tags) for tags in elemTags2)
        print(f"  Mesh: {n_elem} elements, h_min={h_min:.4g}, h_max={h_max:.4g}")

        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags

        return self

    # ------------------------------------------------------------------
    # Analytic solutions for the London magnetostatic problem
    # ------------------------------------------------------------------
    @staticmethod
    def analytic_magnetostatic(d_gap, t_sc, lambda_L, width):
        """
        Analytic solution for the parallel-plate London magnetostatic problem.

        Setup: A_z = 0 at y = -t_sc,  A_z = 1 at y = d_gap + t_sc.

        Returns
        -------
        dict with:
            Sigma_geo, Sigma_kin, Sigma_total : energy integrals × width
            L, L_k, L_geo : inductance per unit length (H/m, needs μ₀ factor)
            C_coeff : coefficient C in the sinh solution
            alpha, beta : vacuum solution A_z = alpha + beta * y
        """
        d, t, lam, W = d_gap, t_sc, lambda_L, width
        st = np.sinh(t / lam)
        ct = np.cosh(t / lam)

        # Matching coefficient
        C_coeff = 1.0 / (2 * st + d * ct / lam)

        alpha = C_coeff * st
        beta = C_coeff * ct / lam

        # --- Compute energy integrals (per unit width W) ---
        # Lower SC: A_z = C * sinh((y+t)/lam), y in [-t, 0]
        #   |dA/dy|² = (C/lam)² cosh²((y+t)/lam)
        #   |A_z|²   = C² sinh²((y+t)/lam)
        # Integral of cosh²(u) from 0 to t/lam = (sinh(2t/lam)/(4/lam) + t/(2lam))... 
        # Use: ∫₀ᵃ cosh²(u) du = sinh(2a)/4 + a/2
        #      ∫₀ᵃ sinh²(u) du = sinh(2a)/4 - a/2

        a = t / lam

        int_cosh2 = np.sinh(2 * a) / 4 + a / 2   # ∫₀ᵃ cosh²(u) du
        int_sinh2 = np.sinh(2 * a) / 4 - a / 2   # ∫₀ᵃ sinh²(u) du

        # Gradient energy in lower SC: W * ∫_{-t}^{0} (C/lam)² cosh²((y+t)/lam) dy
        #  = W * (C²/lam²) * lam * int_cosh2 = W * C² * int_cosh2 / lam
        Sigma_grad_sc_one = C_coeff**2 * int_cosh2 / lam

        # London energy in lower SC: W * (1/lam²) ∫_{-t}^{0} C² sinh²((y+t)/lam) dy
        #  = W * (C²/lam²) * lam * int_sinh2 = W * C² * int_sinh2 / lam
        Sigma_london_sc_one = C_coeff**2 * int_sinh2 / lam

        # By symmetry, upper SC contributes the same
        Sigma_grad_sc = 2 * W * Sigma_grad_sc_one
        Sigma_london = 2 * W * Sigma_london_sc_one

        # Vacuum: A_z = alpha + beta*y,  dA/dy = beta
        # ∫₀ᵈ beta² dy = beta² * d
        Sigma_grad_vac = W * beta**2 * d

        Sigma_geo = Sigma_grad_vac + Sigma_grad_sc
        Sigma_kin = Sigma_london / lam**2 * lam**2  # already has 1/lam² factor
        # Wait, let me recompute. The solver computes:
        #   Sigma_kin = (1/lam²) ∫_SC |A_z|² dA
        # So: Sigma_kin = (1/lam²) * 2 * W * C² * lam * int_sinh2
        #             = 2 * W * C² * int_sinh2 / lam
        Sigma_kin = 2 * W * C_coeff**2 * int_sinh2 / lam

        Sigma_total = Sigma_geo + Sigma_kin

        # Cross-check: Sigma_grad_sc + Sigma_london_sc should relate to Sigma_kin
        # In the London eq: ∇²A - A/lam² = 0 → ∫|∇A|² + ∫(1/lam²)|A|² = boundary terms
        # For one SC slab tested with A: ∫|∇A|² = (boundary) - (1/lam²)∫|A|²
        # So Sigma_grad_sc = (boundary from SC) - Sigma_kin ... but this involves 
        # internal boundary terms. Let's just trust the direct integrals.

        return {
            'C_coeff': C_coeff,
            'alpha': alpha,
            'beta': beta,
            'Sigma_geo': Sigma_geo,
            'Sigma_kin': Sigma_kin,
            'Sigma_total': Sigma_total,
            'L_norm': 1.0 / Sigma_total,  # L / μ₀
        }

    @staticmethod
    def analytic_electrostatic(d_gap, width, eps_r=1.0):
        """
        Analytic capacitance for parallel plate (per unit length in z).
        C = ε₀ ε_r W / d.  Returns C/ε₀ = ε_r W / d.
        """
        return eps_r * width / d_gap

    def plot(self, ax=None):
        points = self.domain.geometry.x[:, :2]
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        triangles = self.domain.topology.connectivity(tdim, 0).array.reshape((-1, 3))

        styles = {
            self.TAG_DOMAIN: ("white", "Vacuum gap"),
            self.TAG_metal: ("steelblue", "Superconductor"),
        }

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        else:
            fig = ax.figure

        for tag, (color, label) in styles.items():
            mask = self.cell_tags.values == tag
            if mask.sum() == 0:
                continue
            ax.tripcolor(
                points[:, 0], points[:, 1], triangles[mask],
                facecolors=np.full(mask.sum(), 1.0),
                cmap=mpl.colors.ListedColormap([color]),
                edgecolors="black", lw=0.2, label=label
            )

        ax.axhline(0, color="red", ls="--", lw=1, label="SC-vacuum interface")
        ax.axhline(self.d_gap, color="red", ls="--", lw=1)
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Parallel-plate SC (test geometry)")
        plt.tight_layout()
        return fig, ax
