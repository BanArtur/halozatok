from dataclasses import dataclass, field

from distribution import IDistribution


@dataclass
class Vertex:
    reward: float
    ancestor: "Vertex | None" = None # Assuming its a directed tree, it has a 0 or 1 ancestor
    out_edges: list["Edge"] = field(default_factory=list)

@dataclass
class Edge:
    origin: Vertex
    end: Vertex
    cost_distribution: IDistribution
    secret_cost: float | None = None
    probed: bool = False
    
    def probe(self) -> None:
        assert self.secret_cost is None, "Vertex already probed"
        self.secret_cost = self.cost_distribution.sample()
        self.probed = True


@dataclass
class Graph:
    start: Vertex
    
    vertices: list[Vertex]
    edges: list[Edge]
