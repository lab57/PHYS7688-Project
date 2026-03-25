import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import fem, io

from charpentier_sweep import CharpentierCWR
from quasi_tem_solver import QuasiTEMSolver

def generate_field_exports():
    # 1. Setup Geometry and Mesh
    w, s, t = 10.0, 5.0, 0.1
    lam_um = 0.050  # 50 nm penetration depth
    mesh_filename = "export_cpw.msh"

    print("Building geometry and mesh for field export...")
    geo = CharpentierCWR(w=w, s=s, t=t)
    geo.build_geometry()
    
    # Using a refined mesh near the metal to capture the London layer accurately
    geo.generate_mesh(metal_mesh_size=0.005, bulk_mesh_size=2.0, filename=mesh_filename)

    # 2. Initialize Solver and Compute Base Fields
    solver = QuasiTEMSolver(geometry=geo, eps_substrate=11.7, lambda_L=lam_um, degree=2)
    
    print("Solving Electrostatic problem (phi)...")
    solver.solve_electrostatic(
        bc_high_tag=geo.TAG_BC_CENTER,
        bc_low_tag=[geo.TAG_BC_GND, geo.TAG_BC_OUTER],
        verbose=False
    )

    print("Solving Magnetostatic problem (A_z)...")
    solver.solve_magnetostatic(
        bc_high_tag=geo.TAG_BC_OUTER,
        sc_mode=True,
        transport_cells={
            geo.TAG_METAL_CENTER: 1.0, 
            geo.TAG_METAL_GND: 0.0
        },
        verbose=False
    )

    # 3. Compute Derived Fields (|E| and J_z)
    print("Computing derived fields (|E| and J_z)...")
    domain = solver.domain
    W_DG = fem.functionspace(domain, ("DG", 0))

    # Electric Field Magnitude: |E| = sqrt(grad(phi) dot grad(phi))
    E_mag = fem.Function(W_DG, name="E_mag")
    grad_phi = ufl.grad(solver.phi)
    expr_E = fem.Expression(ufl.sqrt(ufl.inner(grad_phi, grad_phi)), W_DG.element.interpolation_points)
    E_mag.interpolate(expr_E)

    # Supercurrent Density: J_z = chi_london * A_z
    # chi_london is already 1/lambda_L^2 in the SC and 0 elsewhere
    J_z = fem.Function(W_DG, name="J_z")
    # Revised J_z expression in export_fields.py
    c_transport = fem.Function(W_DG)
    # Re-apply the same logic used in the solver for the center conductor
    cells_center = geo.cell_tags.find(geo.TAG_METAL_CENTER)
    c_transport.x.array[cells_center] = 1.0 

    # The physical current density is proportional to (C - A_z)
    expr_J = fem.Expression(solver.chi_london * (c_transport - solver.A_z), 
                            W_DG.element.interpolation_points)
    J_z.interpolate(expr_J)

    # 4. Export to Paraview
    print("Exporting fields to Paraview (cpw_fields.xdmf)...")
    
    # Create a degree=1 continuous space for visualization compatibility
    V_vis = fem.functionspace(domain, ("Lagrange", 1))
    
    phi_vis = fem.Function(V_vis, name="phi")
    phi_vis.interpolate(solver.phi)
    
    Az_vis = fem.Function(V_vis, name="A_z")
    Az_vis.interpolate(solver.A_z)

    with io.XDMFFile(domain.comm, "cpw_fields.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi_vis)
        xdmf.write_function(Az_vis)
        # DG0 (cell-centered) fields are natively supported by XDMFFile 
        xdmf.write_function(E_mag) 
        xdmf.write_function(J_z)

# 5. Generate Matplotlib PNGs
    print("Generating pyplot PNGs...")
    
    from dolfinx import plot
    
    # Extract mesh topology and geometry correctly mapped to the CG1 function space
    vtk_topology, vtk_cell_types, vtk_geometry = plot.vtk_mesh(V_vis)
    
    # The VTK topology array is 1D, formatted as [3, v0, v1, v2, 3, v0, v1, v2, ...] for triangles
    triangles = vtk_topology.reshape((-1, 4))[:, 1:4]
    points = vtk_geometry[:, :2]

    def save_plot(filename, data, title, cmap, is_cell_data=False, zoom_metal=False):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        
        if is_cell_data:
            # DG0 data maps perfectly to the cell/triangle array
            tc = ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=data, cmap=cmap, shading='flat')
        else:
            # CG1 data maps perfectly to the geometry/vertex array
            tc = ax.tripcolor(points[:, 0], points[:, 1], triangles, data, cmap=cmap, shading='gouraud')
            
        fig.colorbar(tc, ax=ax, label=title)
        ax.set_aspect('equal')
        ax.set_xlabel('x [μm]')
        ax.set_ylabel('y [μm]')
        ax.set_title(title)

        if zoom_metal:
            center_x = geo.Dx / 2
            x_span = 2 * (geo.w + geo.s * 1.5) 
            y_span = x_span / 3.0
            ax.set_xlim(center_x - x_span/2, center_x + x_span/2)
            ax.set_ylim(geo.h - y_span/2, geo.h + y_span/2)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    # Use the VISUALIZATION (degree 1) functions for the continuous plots!
    phi_array = np.real(phi_vis.x.array)
    Az_array = np.real(Az_vis.x.array)
    Emag_array = np.real(E_mag.x.array)
    Jz_array = np.real(J_z.x.array)

    save_plot("plot_phi.png", phi_array, "Electric Potential (V)", "RdBu_r", zoom_metal=True)
    save_plot("plot_A_z.png", Az_array, "Magnetic Vector Potential (A_z)", "viridis", zoom_metal=True)
    
    # |E| and J_z are cell data (DG0)
    save_plot("plot_E_mag.png", Emag_array, "Electric Field Magnitude |E|", "magma", is_cell_data=True, zoom_metal=True)
    save_plot("plot_J_z.png", Jz_array, "Supercurrent Density (J_z)", "inferno", is_cell_data=True, zoom_metal=True)

    print("Done. Check your directory for the .png and .xdmf files.")

if __name__ == "__main__":
    generate_field_exports()