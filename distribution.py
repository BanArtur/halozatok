from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Self

import numpy as np

SAMPLE_SIZE = 10_000


class IDistribution(ABC):
    @abstractmethod
    def sample(self) -> float:
        """
        Samples an element from the distribution randomly
        """

    @abstractmethod
    def expected_value(self) -> float: ...

    @abstractmethod
    def expected_value_max(self, upper: float) -> float:
        """
        Returns E(min(distribution, upper))
        See page 6 '3 Bounding the Optimal Adaptive Policy'
        in Lemma 1 "Let μB (v) = E[min{B, C(v)}]"
        """

    @abstractmethod
    def multiply(self, number: float) -> Self:
        """
        During bounded depth trees the upper bound might be too big, so we can
        adjust the sum of expected_value_max(...) by multiplying the sample by number
        """


class UniformDistribution(IDistribution):
    def __init__(self, lower: float, upper: float) -> None:
        assert 0 <= lower < upper
        self.lower = lower
        self.upper = upper
    
    def __repr__(self) -> str:
        return f"Uniform({self.lower, self.upper})"

    def sample(self) -> float:
        return random.uniform(self.lower, self.upper)

    def expected_value(self) -> float:
        return (self.lower + self.upper) / 2

    def expected_value_max(self, upper: float) -> float:
        """
        Don't worry about this, I tested it in tests/test_uniform.py
        """
        if self.upper <= upper:
            return self.expected_value()  # sample is not larger than upper bound
        elif self.lower <= upper:
            return (
                (upper - self.lower) * (self.lower + upper) / 2
                + (self.upper - upper) * upper
            ) / (self.upper - self.lower)
        else:
            return upper

    def multiply(self, number: float) -> Self:
        assert 0 < number
        return UniformDistribution(number * self.lower, number * self.upper)


class NormalDistribution(IDistribution):
    """
    Almost normal distribution, we ensure it is not negative
    """

    def __init__(
        self, mean: float, scale: float, sample_size: int = SAMPLE_SIZE
    ) -> None:
        assert 0 <= mean
        assert 0 < scale
        assert 100 < sample_size
        self.mean = mean
        self.scale = scale
        self.sample_size = sample_size
        self.cache: dict[float, float] = {}

    def __repr__(self) -> str:
        return f"Normal({self.mean, self.scale})"

    def sample(self) -> float:
        return max(0.0, np.random.normal(self.mean, self.scale))

    def expected_value(self) -> float:
        return float(
            np.average(
                np.maximum(
                    0.0, np.random.normal(self.mean, self.scale, size=self.sample_size)
                )
            )
        )

    def expected_value_max(self, upper: float) -> float:
        if upper in self.cache:
            return self.cache[upper]
        
        value = float(
            np.average(
                np.minimum(
                    np.maximum(
                        0.0,
                        np.random.normal(self.mean, self.scale, size=self.sample_size),
                    ),
                    upper,
                )
            )
        )
        self.cache[upper] = value
        return value
    

    def multiply(self, number: float) -> Self:
        """
        This is not exactly x * number, but I don't care :)
        """
        assert 0 < number
        return NormalDistribution(number * self.mean, number * self.scale)
