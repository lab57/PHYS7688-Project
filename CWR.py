"""
Coplanar Waveguide Resonator — geometry for quasi-TEM solver.

Modified from original CWR.py to separate boundary tags:
  TAG_BC_CENTER  — center conductor boundary (SC-vacuum interface)
  TAG_BC_GND     — ground conductor boundaries (SC-vacuum interface)
  TAG_BC_OUTER   — outer domain boundary (computational box)

This separation is essential for the SC magnetostatic solve, where
the ground conductors participate in the London equation and must NOT
have Dirichlet BCs applied to them.
"""

import matplotlib as mpl
import numpy as np
from mpi4py import MPI
import gmsh
from dolfinx import fem, mesh
from dolfinx.io import gmsh as gmshio
import matplotlib.pyplot as plt


class CoplanarWaveguideResonator:
    """
    CPW cross-section geometry.

    Parameters
    ----------
    w : float
        Center conductor width.
    s : float
        Gap width (each side).
    t : float
        Metal thickness.
    """

    def __init__(self, w, s, t):
        self.w = w
        self.s = s
        self.t = t
        self.h = 3 * (w + 2 * s)             # substrate height

        self.Dx = 7 * (w + 2 * s)            # domain width
        self.Dy = 2 * self.h + t              # domain height

        # --- Cell tags ---
        self.TAG_DOMAIN = 1       # vacuum / air
        self.TAG_metal = 3        # conductor (PEC or SC)
        self.TAG_dielectric = 4   # substrate

        # --- Facet tags ---
        self.TAG_BC_GND = 50      # ground conductor boundaries
        self.TAG_BC_CENTER = 51   # center conductor boundary
        self.TAG_BC_OUTER = 52    # outer domain boundary (computational box)

    def build_geometry(self):
        w, s, t, h = self.w, self.s, self.t, self.h
        Dx, Dy = self.Dx, self.Dy

        # Safety cleanup if gmsh was left open by a previous crash
        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        gmsh.model.add("CoplanarWaveguide")
        occ = gmsh.model.occ

        # Rectangular subdomains
        region = occ.addRectangle(0, 0, 0, Dx, Dy)
        substrate = occ.addRectangle(0, 0, 0, Dx, h)
        return_conductor_l = occ.addRectangle(0, h, 0, Dx/2 - w/2 - s, t)
        return_conductor_r = occ.addRectangle(Dx/2 + w/2 + s, h, 0,
                                               Dx/2 - w/2 - s, t)
        middle_conductor = occ.addRectangle(Dx/2 - w/2, h, 0, w, t)

        # Fragment for conformal meshing at interfaces
        objects = [region, substrate, return_conductor_l,
                   return_conductor_r, middle_conductor]
        frag, mapping = occ.fragment([(2, obj) for obj in objects], [])
        occ.synchronize()

        # Recover surface tags after fragmentation
        (region_tags, substrate_tags, rcl_tags, rcr_tags, mc_tags) = [
            set(tag for dim, tag in mapping[i] if dim == 2)
            for i in range(len(objects))
        ]

        metal = mc_tags | rcl_tags | rcr_tags
        sub = substrate_tags - metal
        air = region_tags - sub - metal

        # Physical groups for cells
        gmsh.model.addPhysicalGroup(2, list(sub), self.TAG_dielectric)
        gmsh.model.addPhysicalGroup(2, list(metal), self.TAG_metal)
        gmsh.model.addPhysicalGroup(2, list(air), self.TAG_DOMAIN)

        # ---- Identify boundary facets ----
        eps = 1e-6  # tolerance for coordinate comparison after OCC fragmentation

        # Outer boundary of the computational domain
        outer_boundary = set()
        all_boundary = gmsh.model.getBoundary(frag, oriented=False)
        for dim, tag in all_boundary:
            if dim != 1:
                continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
            # Edge is on outer boundary if it lies on x=0, x=Dx, y=0, or y=Dy
            on_left = abs(xmin) < eps and abs(xmax) < eps
            on_right = abs(xmin - Dx) < eps and abs(xmax - Dx) < eps
            on_bottom = abs(ymin) < eps and abs(ymax) < eps
            on_top = abs(ymin - Dy) < eps and abs(ymax - Dy) < eps
            if on_left or on_right or on_bottom or on_top:
                outer_boundary.add(tag)

        # Ground conductor boundaries (edges of left/right return conductors)
        gnd_edges_l = set(
            abs(edge) for _, edge in
            gmsh.model.getBoundary([(2, r) for r in rcl_tags])
            if _ == 1
        )
        gnd_edges_r = set(
            abs(edge) for _, edge in
            gmsh.model.getBoundary([(2, r) for r in rcr_tags])
            if _ == 1
        )
        gnd_boundary = (gnd_edges_l | gnd_edges_r) - outer_boundary

        # Center conductor boundary
        center_edges = set(
            abs(edge) for _, edge in
            gmsh.model.getBoundary([(2, r) for r in mc_tags])
            if _ == 1
        )
        center_boundary = center_edges - outer_boundary

        # Assign physical groups
        gmsh.model.addPhysicalGroup(1, list(outer_boundary), self.TAG_BC_OUTER)
        gmsh.model.addPhysicalGroup(1, list(gnd_boundary), self.TAG_BC_GND)
        gmsh.model.addPhysicalGroup(1, list(center_boundary), self.TAG_BC_CENTER)

        print(f"  Facets: {len(outer_boundary)} outer, "
              f"{len(gnd_boundary)} ground, {len(center_boundary)} center")

        return self

    def generate_mesh(self, mesh_factor=1/10):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h * mesh_factor)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h * mesh_factor)
        gmsh.model.mesh.generate(2)

        _, elemTags2, _ = gmsh.model.mesh.getElements(dim=2)
        n2 = sum(len(tags) for tags in elemTags2)
        print(f"  Mesh: {n2} triangular elements, h ≈ {self.h * mesh_factor:.4g}")

        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags

        return self

    def plot(self, ax=None):
        points = self.domain.geometry.x[:, :2]
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        topology = self.domain.topology.connectivity(tdim, 0).array.reshape((-1, 3))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        else:
            fig = ax.figure

        styles = {
            self.TAG_dielectric: ("steelblue", "Substrate"),
            self.TAG_DOMAIN: ("white", "Air"),
            self.TAG_metal: ("grey", "Metal"),
        }

        for tag, (color, label) in styles.items():
            mask = self.cell_tags.values == tag
            if mask.sum() == 0:
                continue
            ax.tripcolor(
                points[:, 0], points[:, 1], topology[mask],
                facecolors=np.full(mask.sum(), 1.0),
                cmap=mpl.colors.ListedColormap([color]),
                edgecolors='black', label=label, lw=0.3
            )

        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("CPW Cross-Section")
        plt.tight_layout()
        return fig, ax
