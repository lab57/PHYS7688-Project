import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

from CWR import CoplanarWaveguideResonator
from quasi_tem_solver import QuasiTEMSolver

MU_0 = 4 * np.pi * 1e-7
H_PLANCK = 6.62607015e-34
E_CHARGE = 1.602176634e-19
R_Q = H_PLANCK / (E_CHARGE**2)
F_RES = 5e9
OMEGA = 2 * np.pi * F_RES

class CharpentierCWR(CoplanarWaveguideResonator):
    def __init__(self, w, s, t):
        super().__init__(w, s, t)
        self.TAG_METAL_CENTER = 31
        self.TAG_METAL_GND = 32
        self.TAG_metals = [self.TAG_METAL_CENTER, self.TAG_METAL_GND]

    def build_geometry(self):
        gmsh.initialize()
        gmsh.model.add("CoplanarWaveguide")
        occ = gmsh.model.occ

        Dx, Dy, h, w, s, t = self.Dx, self.Dy, self.h, self.w, self.s, self.t
        region = occ.addRectangle(0, 0, 0, Dx, Dy)
        substrate = occ.addRectangle(0, 0, 0, Dx, h)
        return_conductor_l = occ.addRectangle(0, h, 0, Dx/2 - w/2 - s, t)
        return_conductor_r = occ.addRectangle(Dx/2 + w/2 + s, h, 0, Dx/2 - w/2 - s, t)
        middle_conductor = occ.addRectangle(Dx/2 - w/2, h, 0, w, t)

        objects = [region, substrate, return_conductor_l, return_conductor_r, middle_conductor]
        frag, mapping = occ.fragment([(2, obj) for obj in objects], [])
        occ.synchronize()

        (region_tags, substrate_tags, rcl_tags, rcr_tags, mc_tags) = [
            set(tag for dim, tag in mapping[i] if dim == 2) for i in range(len(objects))
        ]

        metal = mc_tags | rcl_tags | rcr_tags
        sub = substrate_tags - metal
        air = region_tags - sub - metal

        gmsh.model.addPhysicalGroup(2, list(sub), self.TAG_dielectric)
        gmsh.model.addPhysicalGroup(2, list(air), self.TAG_DOMAIN)
        gmsh.model.addPhysicalGroup(2, list(mc_tags), self.TAG_METAL_CENTER)
        gmsh.model.addPhysicalGroup(2, list(rcl_tags | rcr_tags), self.TAG_METAL_GND)

        eps = 1e-6
        outer_boundary = set()
        for dim, tag in gmsh.model.getBoundary(frag, oriented=False):
            if dim == 1:
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
                if (abs(xmin) < eps and abs(xmax) < eps) or (abs(xmin - Dx) < eps and abs(xmax - Dx) < eps) or \
                   (abs(ymin) < eps and abs(ymax) < eps) or (abs(ymin - Dy) < eps and abs(ymax - Dy) < eps):
                    outer_boundary.add(tag)

        gnd_edges_l = set(abs(edge) for _, edge in gmsh.model.getBoundary([(2, r) for r in rcl_tags]) if _ == 1)
        gnd_edges_r = set(abs(edge) for _, edge in gmsh.model.getBoundary([(2, r) for r in rcr_tags]) if _ == 1)
        center_edges = set(abs(edge) for _, edge in gmsh.model.getBoundary([(2, r) for r in mc_tags]) if _ == 1)

        gmsh.model.addPhysicalGroup(1, list(outer_boundary), self.TAG_BC_OUTER)
        gmsh.model.addPhysicalGroup(1, list((gnd_edges_l | gnd_edges_r) - outer_boundary), self.TAG_BC_GND)
        gmsh.model.addPhysicalGroup(1, list(center_edges - outer_boundary), self.TAG_BC_CENTER)
        return self

    def generate_mesh(self, metal_mesh_size=0.005, bulk_mesh_size=2.0, filename="cached_cpw.msh"):
        metal_curves = list(gmsh.model.getEntitiesForPhysicalGroup(1, self.TAG_BC_GND)) + \
                       list(gmsh.model.getEntitiesForPhysicalGroup(1, self.TAG_BC_CENTER))
        
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", metal_curves)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", metal_mesh_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", bulk_mesh_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.5)

        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)
        
        if filename:
            gmsh.write(filename)
            
        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags
        return self
    
    def plot_mesh(self, ax=None, zoom_to_metal=False):
        """
        Renders the cross-section showing material regions and the underlying finite element mesh.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        # Extract vertex coordinates and topological connectivity from FEniCSx
        points = self.domain.geometry.x[:, :2]
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        topology = self.domain.topology.connectivity(tdim, 0).array.reshape((-1, 3))

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        else:
            fig = ax.figure

        # Define colors for specific regions
        styles = {
            getattr(self, 'TAG_DOMAIN', 1): ("#f8f9fa", "Vacuum"),
            getattr(self, 'TAG_dielectric', 4): ("#a8dadc", "Substrate"),
            getattr(self, 'TAG_METAL_CENTER', 31): ("#e63946", "Center Conductor"),
            getattr(self, 'TAG_METAL_GND', 32): ("#457b9d", "Ground Plane"),
            getattr(self, 'TAG_metal', 3): ("#1d3557", "Metal") # Fallback for normal CWR
        }

        legend_patches = [] # Store manual patches for the legend

        # Iterate through physical groups and plot triangles
        for tag, (color, label) in styles.items():
            mask = self.cell_tags.values == tag
            if mask.sum() == 0:
                continue
            
            # edgecolors and linewidth explicitly render the mesh lattice
            ax.tripcolor(
                points[:, 0], points[:, 1], topology[mask],
                facecolors=np.full(mask.sum(), 1.0),
                cmap=mpl.colors.ListedColormap([color]),
                edgecolors='black', lw=0.15, alpha=0.9
            )
            
            # Create a proxy artist for the legend
            legend_patches.append(mpatches.Patch(color=color, label=label))

        ax.set_aspect('equal')
        ax.set_xlabel('x [μm]')
        ax.set_ylabel('y [μm]')
        ax.set_title("Finite Element Discretization")

        # Use the explicit proxy artists for the legend
        ax.legend(handles=legend_patches, loc='upper right')

        # Optional zooming to highlight the adaptive boundary layer
        if zoom_to_metal:
            center_x = self.Dx / 2
            
            # Define how much of the horizontal layout to show 
            x_span = 2 * (self.w + self.s * 1.5) 
            
            # Force a pleasant aspect ratio for the bounding box (Width / Height)
            aspect_ratio = 3.0 
            y_span = x_span / aspect_ratio
            
            ax.set_xlim(center_x - x_span/2, center_x + x_span/2)
            # Center the vertical view on the metal layer (y = self.h)
            ax.set_ylim(self.h - y_span/2, self.h + y_span/2)

        plt.tight_layout()
        return fig, ax
def run_sweep():
    w, s, t = 10.0, 5.0, 0.1
    lambda_L_um_array = np.geomspace(0.010, 0.600, 15)
    
    alpha_vals, sigma2_vals, Q_qp_vals = [], [], []

    mesh_filename = "cached_cpw.msh"
    geo = CharpentierCWR(w=w, s=s, t=t)

    if False: #unused, caching meshing to save time
        print(f"Loading cached mesh from {mesh_filename}...")
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, 0, gdim=2)
        geo.domain = domain
        geo.cell_tags = cell_tags
        geo.facet_tags = facet_tags
    else:
        print("Building and meshing localized high-resolution geometry...")
        geo.build_geometry()
        geo.generate_mesh(metal_mesh_size=0.005, bulk_mesh_size=2.0, filename=mesh_filename)

    for lam_um in lambda_L_um_array:
        lam_m = lam_um * 1e-6
        sigma2 = 1.0 / (MU_0 * OMEGA * (lam_m**2))
        
        solver = QuasiTEMSolver(geometry=geo, eps_substrate=11.7, lambda_L=lam_um, degree=2)
        
        # Volumetric Transport Current solve
        solver.solve_magnetostatic(
            bc_low_tag=geo.TAG_BC_OUTER,
            sc_mode=True,
            transport_cells={
                geo.TAG_METAL_CENTER: 1.0, 
                geo.TAG_METAL_GND: 0.0
            },
            verbose=False
        )
        
        alpha = solver.results['L_k_fraction_approx']
        Q_qp = (1.0 / alpha) * R_Q * sigma2
        
        alpha_vals.append(alpha)
        sigma2_vals.append(sigma2)
        Q_qp_vals.append(Q_qp)
        print(f"λ_L = {lam_um*1000:6.1f} nm  |  σ_2 = {sigma2:.2e} S/m  |  α = {alpha:.5f}  |  Q_qp = {Q_qp:.2e}")

    sigma2_vals, Q_qp_vals = np.array(sigma2_vals), np.array(Q_qp_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    ax1.plot(lambda_L_um_array * 1000, alpha_vals, 'o-', color='steelblue', lw=2)
    ax1.set_xlabel(r"London Penetration Depth $\lambda_L$ (nm)")
    ax1.set_ylabel(r"Kinetic Inductance Fraction $\alpha$")
    ax1.set_title(r"$\alpha$ vs. Disorder (Fixed Geometry)")
    ax1.grid(True, alpha=0.3)

    ax2.loglog(sigma2_vals, Q_qp_vals, 'ko-', label="FEM Computed Limit")
    ax2.loglog(sigma2_vals[:6], Q_qp_vals[0] * (sigma2_vals[:6] / sigma2_vals[0])**1.5, 'b--', lw=1.5, label=r"Thick Regime $\propto \sigma_2^{3/2}$")
    ax2.loglog(sigma2_vals[-6:], Q_qp_vals[-1] * (sigma2_vals[-6:] / sigma2_vals[-1])**2.0, 'r--', lw=1.5, label=r"Thin Film $\propto \sigma_2^2$")
    ax2.set_xlabel(r"Imaginary Conductivity $\sigma_2$ (S/m)")
    ax2.set_ylabel(r"Max Quasiparticle Quality Factor $Q_{qp}$")
    ax2.set_title("Universal Scaling of Microwave Dissipation")
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sweep()