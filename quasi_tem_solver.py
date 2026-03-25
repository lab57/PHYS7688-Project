import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh as dmesh, geometry as dgeo
from dolfinx.fem.petsc import LinearProblem
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

EPS_0 = 8.854187817e-12
MU_0 = 4 * np.pi * 1e-7
C_LIGHT = 1.0 / np.sqrt(EPS_0 * MU_0)

class QuasiTEMSolver:
    def __init__(self, geometry, eps_substrate=1.0, lambda_L=None, degree=1):
        self.geo = geometry
        self.eps_sub = eps_substrate
        self.lambda_L = lambda_L
        self.degree = degree

        self.domain = geometry.domain
        self.cell_tags = geometry.cell_tags
        self.facet_tags = geometry.facet_tags

        self.V = fem.functionspace(self.domain, ("Lagrange", degree))
        self.DG0 = fem.functionspace(self.domain, ("DG", 0))

        self._setup_materials()
        self.phi = None
        self.A_z = None
        self.results = {}

    def _setup_materials(self):
        self.eps_r = fem.Function(self.DG0, name="eps_r")
        self.eps_r.x.array[:] = 1.0

        if hasattr(self.geo, 'TAG_dielectric'):
            cells_diel = self.cell_tags.find(self.geo.TAG_dielectric)
            if len(cells_diel) > 0:
                self.eps_r.x.array[cells_diel] = self.eps_sub

        self.chi_london = fem.Function(self.DG0, name="chi_london")
        self.chi_london.x.array[:] = 0.0

        if self.lambda_L is not None and self.lambda_L > 0:
            metal_tags = getattr(self.geo, 'TAG_metals', [getattr(self.geo, 'TAG_metal', None)])
            for tag in metal_tags:
                if tag is not None:
                    cells = self.cell_tags.find(tag)
                    if len(cells) > 0:
                        self.chi_london.x.array[cells] = 1.0 / self.lambda_L**2

    def solve_electrostatic(self, bc_high_tag=None, bc_low_tag=None, verbose=True):
        '''
        Electrostatic problem only
        
        '''
        if bc_high_tag is None: bc_high_tag = self.geo.TAG_BC_CENTER
        if bc_low_tag is None: bc_low_tag = self.geo.TAG_BC_GND

        V = self.V
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = ufl.inner(self.eps_r * ufl.grad(u), ufl.grad(v)) * ufl.dx
        f = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        L = ufl.inner(f, v) * ufl.dx

        bcs = self._make_dirichlet_bcs(V, bc_high_tag, 1.0, bc_low_tag, 0.0)

        problem = LinearProblem(
            a, L, bcs=bcs, 
            petsc_options_prefix="electrostatic_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )

        self.phi = problem.solve()
        self.phi.name = "phi"

        grad_phi = ufl.grad(self.phi)
        energy_form = fem.form(self.eps_r * ufl.inner(grad_phi, grad_phi) * ufl.dx)
        C_norm = self.domain.comm.allreduce(fem.assemble_scalar(energy_form), op=MPI.SUM)

        self.results['C_norm'] = C_norm
        self.results['C'] = EPS_0 * C_norm
        return C_norm

    def solve_magnetostatic(self, bc_high_tag=None, bc_low_tag=None, sc_mode=False, transport_cells=None, verbose=True):
        """
        Magnetostatic problem only
        
        """
        if bc_high_tag is None: bc_high_tag = self.geo.TAG_BC_CENTER
        if bc_low_tag is None: bc_low_tag = self.geo.TAG_BC_OUTER if sc_mode else self.geo.TAG_BC_GND

        V = self.V
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        if sc_mode and self.lambda_L is not None:
            a += self.chi_london * ufl.inner(u, v) * ufl.dx

        if transport_cells is not None and sc_mode:
            c_transport = fem.Function(self.DG0, name="c_transport")
            c_transport.x.array[:] = 0.0
            for tag, val in transport_cells.items():
                cells = self.cell_tags.find(tag)
                if len(cells) > 0:
                    c_transport.x.array[cells] = val

            f_expr = self.chi_london * c_transport
            L_form = ufl.inner(f_expr, v) * ufl.dx
            bcs = self._make_dirichlet_bcs(V, None, None, bc_low_tag, 0.0)
        else:
            f = fem.Constant(self.domain, PETSc.ScalarType(0.0))
            L_form = ufl.inner(f, v) * ufl.dx
            bcs = self._make_dirichlet_bcs(V, bc_high_tag, 1.0, bc_low_tag, 0.0)


        problem = LinearProblem(
            a, L_form, bcs=bcs, 
            petsc_options_prefix="magnetostatic_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        self.A_z = problem.solve()
        self.A_z.name = "A_z"

        Sigma_geo_form = fem.form(ufl.inner(ufl.grad(self.A_z), ufl.grad(self.A_z)) * ufl.dx)
        Sigma_geo = self.domain.comm.allreduce(fem.assemble_scalar(Sigma_geo_form), op=MPI.SUM)

        if transport_cells is not None and sc_mode:
            Sigma_kin_form = fem.form(self.chi_london * ufl.inner(c_transport - self.A_z, c_transport - self.A_z) * ufl.dx)
            Sigma_kin = self.domain.comm.allreduce(fem.assemble_scalar(Sigma_kin_form), op=MPI.SUM)

            I_form = fem.form(c_transport * self.chi_london * (c_transport - self.A_z) * ufl.dx)
            I_scaled = self.domain.comm.allreduce(fem.assemble_scalar(I_form), op=MPI.SUM)

            Sigma_total = Sigma_geo + Sigma_kin
            L_norm = Sigma_total / (I_scaled**2) if I_scaled > 0 else 0
            alpha = Sigma_kin / Sigma_total if Sigma_total > 0 else 0
        else:
            Sigma_kin = 0.0
            if sc_mode and self.lambda_L is not None:
                Sigma_kin_form = fem.form(self.chi_london * ufl.inner(self.A_z, self.A_z) * ufl.dx)
                Sigma_kin = self.domain.comm.allreduce(fem.assemble_scalar(Sigma_kin_form), op=MPI.SUM)

            Sigma_total = Sigma_geo + Sigma_kin
            L_norm = 1.0 / Sigma_total if Sigma_total > 0 else 0
            alpha = Sigma_kin / Sigma_total if Sigma_total > 0 else 0

        self.results['Sigma_geo'] = Sigma_geo
        self.results['Sigma_kin'] = Sigma_kin
        self.results['Sigma_total'] = Sigma_total
        self.results['L_norm'] = L_norm
        self.results['L'] = MU_0 * L_norm
        self.results['L_k_fraction_approx'] = alpha

        return L_norm

    def compute_line_parameters(self, verbose=True):
        C = self.results['C']
        L = self.results['L']
        self.results['Z_0'] = np.sqrt(L / C)
        self.results['v_ph'] = 1.0 / np.sqrt(L * C)
        self.results['eps_eff'] = (C_LIGHT / self.results['v_ph']) ** 2
        return self.results

    def _make_dirichlet_bcs(self, V, tag_high, val_high, tag_low, val_low):
        bcs = []
        if tag_high is not None:
            facets_high = self.facet_tags.find(tag_high)
            if len(facets_high) > 0:
                dofs_high = fem.locate_dofs_topological(V, 1, facets_high)
                bcs.append(fem.dirichletbc(PETSc.ScalarType(val_high), dofs_high, V))

        if tag_low is not None:
            tags_low = tag_low if isinstance(tag_low, (list, tuple)) else [tag_low]
            for tl in tags_low:
                facets_low = self.facet_tags.find(tl)
                if len(facets_low) > 0:
                    dofs_low = fem.locate_dofs_topological(V, 1, facets_low)
                    bcs.append(fem.dirichletbc(PETSc.ScalarType(val_low), dofs_low, V))
        return bcs
