import copy
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

    vertices: dict[int, Vertex]
    edges: dict[tuple[int, int], Edge]

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
        for edge in self.edges.values():
            edge.cost_distribution = edge.cost_distribution.multiply(ratio)

        cost = self.mu_cost(B)
        while cost > beta:
            for edge in self.edges.values():
                edge.cost_distribution = edge.cost_distribution.multiply(0.9)
            cost = self.mu_cost(B)

    def copy(self) -> "Graph":
        root = copy.deepcopy(self.start)
        vertices: dict[int, Vertex] = {}
        edges: dict[tuple[int, int], Edge] = {}

        todos: list[Vertex] = [root]
        index = 0
        while index < len(todos):
            vertex = todos[index]
            vertices[vertex.id] = vertex
            for edge in vertex.out_edges:
                edges[(edge.origin.id, edge.end.id)] = edge
                todos.append(edge.end)
            index += 1

        return Graph(start=root, vertices=vertices, edges=edges)

    def removeVertex(self, vertex: Vertex) -> None:
        # self.check_reachable()
        todos: list[Vertex] = [vertex]
        index = 0
        while index < len(todos):
            curr = todos[index]
            del self.vertices[curr.id]
            for edge in curr.out_edges:
                del self.edges[(edge.origin.id, edge.end.id)]
                todos.append(edge.end)

            index += 1

        parent = vertex.ancestor
        if parent is not None:
            parent.out_edges = [
                edge for edge in parent.out_edges if edge.end.id != vertex.id
            ]
        # self.check_reachable()

    def addSubtree(self, T: "Graph", vertex: Vertex) -> None:
        # self.check_reachable()
        if vertex.id not in self.vertices:
            ancestor = vertex.ancestor
            assert ancestor is not None
            assert ancestor.id in self.vertices
            true_ancestor = self.vertices[ancestor.id]
            true_vertex = Vertex(
                id=vertex.id, reward=vertex.reward, ancestor=true_ancestor
            )
            true_edge = Edge(
                true_ancestor,
                true_vertex,
                T.edges[(ancestor.id, vertex.id)].cost_distribution,
            )
            self.edges[(ancestor.id, vertex.id)] = true_edge
            self.vertices[vertex.id] = true_vertex
            true_ancestor.out_edges.append(true_edge)

        for edge in vertex.out_edges:
            self.addSubtree(T, edge.end)

    def check_reachable(self) -> None:
        all_vert = set(self.vertices)
        reachable = [self.start]
        idx = 0
        while idx < len(reachable):
            vertex = reachable[idx]
            assert vertex is self.vertices[vertex.id], f"{vertex.id=}"
            for edge in vertex.out_edges:
                reachable.append(edge.end)
            idx += 1
        reachable = [v.id for v in reachable]
        assert set(reachable) == set(all_vert), f"{set(reachable)} == {set(all_vert)}"
