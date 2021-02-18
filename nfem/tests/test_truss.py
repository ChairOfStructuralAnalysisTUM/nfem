"""
Tests for the Truss element
"""

import nfem
import pytest
from numpy.testing import assert_equal, assert_almost_equal


@pytest.fixture
def undeformed_truss():
    node_a = nfem.Node('A', 1, 2, 3)
    node_b = nfem.Node('B', 4, 6, 3)
    return nfem.Truss('1', node_a, node_b, 2, 1)


@pytest.fixture
def truss():
    node_a = nfem.Node('A', 1, 2, 3)
    node_b = nfem.Node('B', 4, 6, 3)
    node_b.u = 3
    node_b.v = 4
    node_b.w = 0
    return nfem.Truss('1', node_a, node_b, 2, 1)


@pytest.fixture
def truss_xls():
    node_a = nfem.Node('A', 0, 0, 0)
    node_b = nfem.Node('B', 1, 2, 3)
    node_b.u += 0.1
    return nfem.Truss('1', node_a, node_b, 1, 1)


def test_truss_ref_length(truss):
    assert_equal(truss.ref_length, 5)


def test_truss_act_length(truss):
    assert_equal(truss.length, 10)


def test_truss_linear_strain_is_zero(undeformed_truss):
    assert_equal(undeformed_truss.engineering_strain, 0)


def test_truss_green_lagrange_strain_is_zero(undeformed_truss):
    assert_equal(undeformed_truss.green_lagrange_strain, 0)


def test_truss_engineering_strain(truss_xls):
    assert_equal(truss_xls.engineering_strain, 0.0071428571428572)


def test_truss_green_lagrange_strain(truss_xls):
    assert_equal(truss_xls.green_lagrange_strain, 0.00750000000000003)


def test_truss_normal_force(truss_xls):
    assert_equal(truss_xls.normal_force, 0.007556040629853737)


def test_truss_stiffness(truss_xls):
    k_actual = truss_xls.calculate_stiffness_matrix()

    # values from nfem.Truss.xls
    k_expected = [
        [ 0.025103466943026,  0.041998194741606,  0.062997292612409, -0.025103466943026, -0.041998194741606, -0.062997292612409],
        [ 0.041998194741606,  0.07836481431434 ,  0.114540532      , -0.041998194741606, -0.07836481431434 , -0.114540532      ],
        [ 0.062997292612409,  0.114540532      ,  0.17381525731434 , -0.062997292612409, -0.114540532      , -0.17381525731434 ],
        [-0.025103466943026, -0.041998194741606, -0.062997292612409,  0.025103466943026,  0.041998194741606,  0.062997292612409],
        [-0.041998194741606, -0.07836481431434 , -0.114540532      ,  0.041998194741606,  0.07836481431434 ,  0.114540532      ],
        [-0.062997292612409, -0.114540532      , -0.17381525731434 ,  0.062997292612409,  0.114540532      ,  0.17381525731434 ],
    ]

    assert_almost_equal(k_actual, k_expected)


def test_truss_material_stiffness(truss_xls):
    k_actual = truss_xls.calculate_material_stiffness_matrix()

    import numpy
    numpy.set_printoptions(15)
    print(k_actual)

    k_expected = [
        [ 0.023099007336717,  0.041998195157667,  0.0629972927365  , -0.023099007336717, -0.041998195157667, -0.0629972927365  ],
        [ 0.041998195157667,  0.076360354832121,  0.114540532248182, -0.041998195157667, -0.076360354832121, -0.114540532248182],
        [ 0.0629972927365  ,  0.114540532248182,  0.171810798372273, -0.0629972927365  , -0.114540532248182, -0.171810798372273],
        [-0.023099007336717, -0.041998195157667, -0.0629972927365  ,  0.023099007336717,  0.041998195157667,  0.0629972927365  ],
        [-0.041998195157667, -0.076360354832121, -0.114540532248182,  0.041998195157667,  0.076360354832121,  0.114540532248182],
        [-0.0629972927365  , -0.114540532248182, -0.171810798372273,  0.0629972927365  ,  0.114540532248182,  0.171810798372273],
    ]

    assert_almost_equal(k_actual, k_expected)


def test_truss_elastic_stiffness(truss_xls):
    k_e_actual = truss_xls.calculate_elastic_stiffness_matrix()

    # values from nfem.Truss.xls
    k_e_expected = [
        [ 0.019090089,  0.038180177,  0.057270266, -0.019090089, -0.038180177, -0.057270266],
        [ 0.038180177,  0.076360355,  0.114540532, -0.038180177, -0.076360355, -0.114540532],
        [ 0.057270266,  0.114540532,  0.171810798, -0.057270266, -0.114540532, -0.171810798],
        [-0.019090089, -0.038180177, -0.057270266,  0.019090089,  0.038180177,  0.057270266],
        [-0.038180177, -0.076360355, -0.114540532,  0.038180177,  0.076360355,  0.114540532],
        [-0.057270266, -0.114540532, -0.171810798,  0.057270266,  0.114540532,  0.171810798]
    ]

    assert_almost_equal(k_e_actual, k_e_expected)


def test_truss_initial_displacement_stiffness(truss_xls):
    k_u_actual = truss_xls.calculate_initial_displacement_stiffness_matrix()

    # values from nfem.Truss.xls
    k_u_expected = [
        [ 0.004008918628686,  0.003818017741606,  0.005727026612409, -0.004008918628686, -0.003818017741606, -0.005727026612409],
        [ 0.003818017741606,  0.000000000000000,  0.000000000000000, -0.003818017741606,  0.000000000000000,  0.000000000000000],
        [ 0.005727026612409,  0.000000000000000,  0.000000000000000, -0.005727026612409,  0.000000000000000,  0.000000000000000],
        [-0.004008918628686, -0.003818017741606, -0.005727026612409,  0.004008918628686,  0.003818017741606,  0.005727026612409],
        [-0.003818017741606,  0.000000000000000,  0.000000000000000,  0.003818017741606,  0.000000000000000,  0.000000000000000],
        [-0.005727026612409,  0.000000000000000,  0.000000000000000,  0.005727026612409,  0.000000000000000,  0.000000000000000]
    ]

    assert_almost_equal(k_u_actual, k_u_expected)


def test_truss_geometric_stiffness(truss_xls):
    k_g_actual = truss_xls.calculate_geometric_stiffness_matrix()

    # values from nfem.Truss.xls
    k_g_expected = [
        [ 2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00, -2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00],
        [ 0.00000000000E+00,  2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00, -2.00445931434E-03,  0.00000000000E+00],
        [ 0.00000000000E+00,  0.00000000000E+00,  2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00, -2.00445931434E-03],
        [-2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00,  2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00],
        [ 0.00000000000E+00, -2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00,  2.00445931434E-03,  0.00000000000E+00],
        [ 0.00000000000E+00,  0.00000000000E+00, -2.00445931434E-03,  0.00000000000E+00,  0.00000000000E+00,  2.00445931434E-03]
    ]

    assert_almost_equal(k_g_actual, k_g_expected)


def test_truss_internal_forces(truss_xls):
    f_actual = truss_xls.calculate_internal_forces()

    f_expected = [-0.0022049, -0.0040089, -0.0060134,  0.0022049,  0.0040089, 0.0060134]

    assert_almost_equal(f_actual, f_expected)
