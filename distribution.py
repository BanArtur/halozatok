from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Self


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


@dataclass
class UniformDistribution(IDistribution):
    def __init__(self, lower: float, upper: float) -> None:
        assert 0 <= lower < upper
        self.lower = lower
        self.upper = upper

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