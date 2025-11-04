import json
from typing import Any

from distribution import (
    SAMPLE_SIZE,
    IDistribution,
    NormalDistribution,
    UniformDistribution,
)
from graph import Edge, Graph, Vertex


class GraphJSONEncoder(json.JSONEncoder):
    """
    Graph -> JSON
    Dumps only the start vertex
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Graph):
            return {"start": self._encode_vertex(obj.start)}

        elif isinstance(obj, UniformDistribution):
            return {"type": "uniform", "lower": obj.lower, "upper": obj.upper}

        elif isinstance(obj, NormalDistribution):
            return {
                "type": "normal",
                "mean": obj.mean,
                "scale": obj.scale,
                "sample_size": obj.sample_size,
            }

        return super().default(obj)

    def _encode_vertex(self, v: Vertex) -> dict:
        return {
            "reward": v.reward,
            "out_edges": [self._encode_edge(e) for e in v.out_edges],
        }

    def _encode_edge(self, e: Edge) -> dict:
        return {
            "cost_distribution": self.default(e.cost_distribution),
            "secret_cost": e.secret_cost,
            "probed": e.probed,
            "end": self._encode_vertex(e.end),
        }


class GraphJSONDecoder(json.JSONDecoder):
    """
    Decodes JSON -> Graph.
    Rebuilds the graph from the start vertex recursively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, obj: dict):
        return obj

    def object_hook_graph(self, obj):
        # Custom hook: return Graph object if 'start' key detected
        if isinstance(obj, dict) and "start" in obj:
            return self._decode_graph_from_obj(obj)
        return obj

    def _decode_graph_from_obj(self, data):
        start = self._decode_vertex(data["start"], ancestor=None)
        vertices, edges = self._collect(start)
        vertex_dict: dict[int, Vertex] = {}
        for i, vertex in enumerate(vertices):
            vertex.id = i
            vertex_dict[i] = vertex
        edges_dict = {}
        for edge in edges:
            edges_dict[(edge.origin.id, edge.end.id)] = edge
        return Graph(start=start, vertices=vertex_dict, edges=edges_dict)

    def decode_graph(self, s: str):
        return self._decode_graph_from_obj(json.loads(s))

    def _decode_vertex(self, data: dict, ancestor: Vertex | None) -> Vertex:
        v = Vertex(id=None, reward=data["reward"], ancestor=ancestor)
        for e_data in data.get("out_edges", []):
            end_vertex = self._decode_vertex(e_data["end"], ancestor=v)
            dist = self._decode_distribution(e_data["cost_distribution"])
            e = Edge(
                origin=v,
                end=end_vertex,
                cost_distribution=dist,
                secret_cost=e_data.get("secret_cost"),
                probed=e_data.get("probed", False),
            )
            v.out_edges.append(e)
        return v

    def _decode_distribution(self, data: dict) -> IDistribution:
        t = data["type"]
        if t == "uniform":
            return UniformDistribution(data["lower"], data["upper"])
        elif t == "normal":
            return NormalDistribution(
                data["mean"], data["scale"], data.get("sample_size", SAMPLE_SIZE)
            )
        else:
            raise ValueError(f"Unknown distribution type: {t}")

    def _collect(self, start: Vertex) -> tuple[list[Vertex], list[Edge]]:
        vertices, edges = [], []

        def dfs(v: Vertex):
            vertices.append(v)
            for e in v.out_edges:
                edges.append(e)
                dfs(e.end)

        dfs(start)
        return vertices, edges
