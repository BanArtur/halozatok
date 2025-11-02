from dataclasses import dataclass, field

from distribution import IDistribution


@dataclass
class Vertex:
    id: int
    reward: float
    ancestor: "Vertex | None" = (
        None  # Assuming its a directed tree, it has a 0 or 1 ancestor
    )
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

    def mu_cost(self, B: float) -> float:
        """
        Important for bounded trees. Returns the maximal cost for the leaves bounded by B.
        For bounded trees we expect all of our trees in the dataset to be smaller than beta
        """
        todos: list[tuple[Vertex, float]] = [(self.start, 0)]
        index = 0
        while index < len(todos):
            vertex, cost = todos[index]
            for edge in vertex.out_edges:
                mu = edge.cost_distribution.expected_value_max(B)
                todos.append((edge.end, cost + mu))

            index += 1

        return max([cost for _, cost in todos])

    def lower_mu_cost(self, B: float, beta: float) -> None:
        """
        Returns a graph where the cost is smaller than beta
        """
        cost = self.mu_cost(B)
        if cost <= beta:
            return None

        ratio = beta / cost
        for edge in self.edges:
            edge.cost_distribution = edge.cost_distribution.multiply(ratio)

        cost = self.mu_cost(B)
        while cost > beta:
            for edge in self.edges:
                edge.cost_distribution = edge.cost_distribution.multiply(0.9)
            cost = self.mu_cost(B)
