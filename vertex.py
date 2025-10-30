from dataclasses import dataclass

from distribution import IDistribution


@dataclass
class Vertex:
    reward: float
    out_edges: list["Edge"]

@dataclass
class Edge:
    origin: Vertex
    end: Vertex
    distribution: IDistribution
    secret_sample: float | None = None
