import random
import pytest

from distribution import NormalDistribution, UniformDistribution

NUMBER_OF_SAMPLES = 100_000

arguments = [
    (1, 2, 0.2),
    (1, 2, 0.8),
    (1, 2, 1.0),
    (1, 2, 1.2),
    (1, 2, 1.5),
    (1, 2, 1.8),
    (1, 2, 2.0),
    (1, 2, 2.2),
    (1, 2, 3),
]

@pytest.mark.parametrize("a, b, upper", arguments)
def test_uniform_expected_value_max(a: float, b: float, upper: float):
    """
    Testing if expected_value_max returns a close number
    to the statistically measured average.
    """
    random.seed(42)
    distribution = UniformDistribution(a, b)
    samples = [min(distribution.sample(), upper) for _ in range(NUMBER_OF_SAMPLES)]
    measured_average = sum(samples) / len(samples)
    expected_val = distribution.expected_value_max(upper)
    assert abs(expected_val - measured_average) < 0.001
    

@pytest.mark.parametrize("mean, scale, upper", arguments)
def test_normal_expected_value_max(mean: float, scale: float, upper: float):
    """
    Testing if expected_value_max returns a close number
    to the statistically measured average.
    """
    random.seed(42)
    distribution = NormalDistribution(mean, scale)
    samples = [min(distribution.sample(), upper) for _ in range(NUMBER_OF_SAMPLES)]
    measured_average = sum(samples) / len(samples)
    expected_val = distribution.expected_value_max(upper)
    assert abs(expected_val - measured_average) < 0.005