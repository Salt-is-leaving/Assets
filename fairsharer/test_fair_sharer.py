
import numpy as np
from distribution_among_neighbours import fair_sharer 

def test_fair_sharer_basic():
    values = np.array([100, 200, 300, 400])
    num_iterations = 1
    share = 0.1
    expected = np.array([130, 170, 270, 330])
    result = fair_sharer(values, num_iterations, share)
    np.testing.assert_array_equal(result, expected)

def test_fair_sharer_no_change():
    values = np.array([100, 100, 100, 100])
    num_iterations = 1
    share = 0.1
    expected = values.copy()
    result = fair_sharer(values, num_iterations, share)
    np.testing.assert_array_equal(result, expected)

def test_fair_sharer_behavior():
    values = np.array([400, 100, 200, 300])
    num_iterations = 1
    share = 0.1
    expected = np.array([400, 140, 170, 290])
    result = fair_sharer(values, num_iterations, share)
    np.testing.assert_array_equal(result, expected)
