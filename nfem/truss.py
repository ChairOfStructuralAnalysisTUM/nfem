"""This module only contains the truss element.

Authors: Thomas Oberbichler, Klaus Sautter
"""

from nfem.node import Node

import numpy as np
import numpy.linalg as la

from typing import Optional


class Truss:
    """FIXME"""

    node_a: Node
    node_b: Node
    youngs_modulus: float
    area: float
    prestress: float
    tensile_strength: Optional[float]
    compressive_strength: Optional[float]

    def __init__(self, id: str, node_a: Node, node_b: Node, youngs_modulus: float, area: float, prestress: float = 0.0, tensile_strength: Optional[float] = None, compressive_strength: Optional[float] = None):
        """FIXME"""

        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.youngs_modulus = youngs_modulus
        self.area = area
        self.prestress = prestress
        self.tensile_strength = tensile_strength
        self.compressive_strength = compressive_strength

    @property
    def dofs(self):
        """FIXME"""
        node_a = self.node_a
        node_b = self.node_b

        return [node_a._dof_x, node_a._dof_y, node_a._dof_z, node_b._dof_x, node_b._dof_y, node_b._dof_z]

    @property
    def ref_length(self) -> float:
        """Gets the length of the undeformed truss"""
        a = self.node_a.ref_location
        b = self.node_b.ref_location

        return la.norm(b - a)

    @property
    def length(self) -> float:
        """Gets the length of the deformed truss"""
        a = self.node_a.location
        b = self.node_b.location

        return la.norm(b - a)

    def calculate_elastic_stiffness_matrix(self):
        """FIXME"""
        E = self.youngs_modulus
        A = self.area
        L = self.ref_length

        Dx, Dy, Dz = self.node_b.ref_location - self.node_a.ref_location

        D = np.array([-Dx, -Dy, -Dz, Dx, Dy, Dz])

        return np.outer(E * A / L**3 * D, D)

    def calculate_material_stiffness_matrix(self):
        """FIXME"""
        E = self.youngs_modulus
        A = self.area
        L = self.ref_length

        dx, dy, dz = self.node_b.location - self.node_a.location

        d = np.array([-dx, -dy, -dz, dx, dy, dz])

        return np.outer(E * A / L**3 * d, d)

    def calculate_initial_displacement_stiffness_matrix(self):
        """FIXME"""
        K_m = self.calculate_material_stiffness_matrix()
        K_e = self.calculate_elastic_stiffness_matrix()

        return K_m - K_e

    def calculate_geometric_stiffness_matrix(self, linear: bool = False):
        E = self.youngs_modulus
        A = self.area
        L = self.ref_length
        prestress = self.prestress

        if linear:
            # FIXME this is not a clean solution to solve the LPB issue
            epsilon = self.calculate_linear_strain()
        else:
            epsilon = self.calculate_green_lagrange_strain()

        sigma = E * epsilon + prestress

        q = sigma * A / L

        return np.array([[ q,  0,  0, -q,  0,  0],
                         [ 0,  q,  0,  0, -q,  0],
                         [ 0,  0,  q,  0,  0, -q],
                         [-q,  0,  0,  q,  0,  0],
                         [ 0, -q,  0,  0,  q,  0],
                         [ 0,  0, -q,  0,  0,  q]])

    def calculate_stiffness_matrix(self):
        """FIXME"""

        K_m = self.calculate_material_stiffness_matrix()
        K_g = self.calculate_geometric_stiffness_matrix()

        return K_m + K_g

    def calculate_green_lagrange_strain(self):
        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        # green-lagrange strain
        epsilon_GL = (a1 @ a1 - A1 @ A1) / (2 * A1 @ A1)

        return epsilon_GL

    def calculate_linear_strain(self):
        """FIXME"""

        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        L = self.ref_length

        # project actual on reference
        projected_l = a1 @ A1 / np.sqrt(A1 @ A1)

        e_lin = (projected_l - L) / L

        return e_lin

    def calculate_internal_forces(self):
        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        # green-lagrange strain
        eps = self.calculate_green_lagrange_strain()

        E = self.youngs_modulus
        A = self.area
        L = np.sqrt(A1 @ A1)

        D_pi = (eps * E + self.prestress) * A * L

        D_eps = a1 / (A1 @ A1)

        D_a1 = np.array([[-1,  0,  0,  1,  0,  0],
                         [ 0, -1,  0,  0,  1,  0],
                         [ 0,  0, -1,  0,  0,  1]])

        F = D_pi * D_eps @ D_a1

        return F

    def calculate_stress(self):
        eps = self.calculate_green_lagrange_strain()
        E = self.youngs_modulus

        # stress:
        sigma = eps * E + self.prestress

        return sigma

    @property
    def normal_force(self) -> float:
        biot_stress = self.calculate_stress() * self.length / self.ref_length
        A = self.area
        F = biot_stress * A

        return F

    def draw(self, item):
        sigma = self.calculate_stress()
        color = 'black'
        eta = None

        if self.calculate_stress() > 1e-3:
            color = 'blue'
            if self.tensile_strength is not None:
                sigma_max = self.tensile_strength
                eta = sigma / sigma_max
        elif self.calculate_stress() < -1e-3:
            color = 'red'
            if self.compressive_strength is not None:
                sigma_max = -self.compressive_strength
                eta = sigma / sigma_max
        elif self.tensile_strength is not None and self.compressive_strength is not None:
            eta = 0.0

        item.set_label_location(
            ref=0.5 * (self.node_a.ref_location + self.node_b.ref_location),
            act=0.5 * (self.node_a.location + self.node_b.location),
        )

        item.add_line(
            points=[
                self.node_a.ref_location,
                self.node_b.ref_location,
            ],
            layer=10,
            color='gray',
        )

        item.add_line(
            points=[
                self.node_a.location,
                self.node_b.location,
            ],
            layer=20,
            color=color,
        )

        item.add_result('Length undeformed', self.ref_length)
        item.add_result('Length', self.length)
        item.add_result('Engineering Strain', self.calculate_linear_strain())
        item.add_result('Green-Lagrange Strain', self.calculate_green_lagrange_strain())
        item.add_result('PK2 Stress', sigma)
        item.add_result('Normal Force', self.normal_force)

        if eta is not None:
            item.add_result('Degree of Utilization', eta)
