import json
import os
import tempfile
import pytest
from pathlib import Path

from distribution import NormalDistribution, UniformDistribution
from serialize import GraphJSONDecoder, GraphJSONEncoder
from graph import Edge, Graph, Vertex


def graphs_equal(g1: Graph, g2: Graph) -> bool:
    """Recursive equality check for small graphs."""
    def equal_vertex(v1: Vertex, v2: Vertex):
        if v1.reward != v2.reward or len(v1.out_edges) != len(v2.out_edges):
            return False
        for e1, e2 in zip(v1.out_edges, v2.out_edges):
            if not equal_edge(e1, e2):
                return False
        return True

    def equal_edge(e1: Edge, e2: Edge):
        if e1.probed != e2.probed or e1.secret_cost != e2.secret_cost:
            return False
        d1, d2 = e1.cost_distribution, e2.cost_distribution
        if type(d1) != type(d2):
            return False
        if isinstance(d1, UniformDistribution):
            if (d1.lower, d1.upper) != (d2.lower, d2.upper):
                return False
        elif isinstance(d1, NormalDistribution):
            if (d1.mean, d1.scale) != (d2.mean, d2.scale):
                return False
        return equal_vertex(e1.end, e2.end)

    return equal_vertex(g1.start, g2.start)


def make_sample_graphs():
    graphs = []

    # Graph 1
    v1 = Vertex(10)
    v2 = Vertex(5, ancestor=v1)
    v3 = Vertex(2, ancestor=v1)
    e1 = Edge(v1, v2, UniformDistribution(1, 3))
    e2 = Edge(v1, v3, NormalDistribution(4, 1))
    v1.out_edges = [e1, e2]
    g1 = Graph(start=v1, vertices=[v1, v2, v3], edges=[e1, e2])

    # Graph 2
    a1 = Vertex(3)
    a2 = Vertex(7, ancestor=a1)
    e3 = Edge(a1, a2, UniformDistribution(2, 6), secret_cost=4.0, probed=True)
    a1.out_edges = [e3]
    g2 = Graph(start=a1, vertices=[a1, a2], edges=[e3])

    # Graph 3
    b1 = Vertex(8)
    b2 = Vertex(1, ancestor=b1)
    e4 = Edge(b1, b2, NormalDistribution(5, 0.5))
    b1.out_edges = [e4]
    g3 = Graph(start=b1, vertices=[b1, b2], edges=[e4])

    return [g1, g2, g3]


def test_graph_json_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        decoder = GraphJSONDecoder()
        graphs = make_sample_graphs()

        
        # Save each separately
        for i, g in enumerate(graphs):
            path = os.path.join(tmp, f"graph_{i}.json")
            with open(path, "w") as f:
                json.dump(g, f, cls=GraphJSONEncoder)
            with open(path, "r") as f:
                loaded = json.load(f, object_hook=decoder.object_hook_graph)
            assert graphs_equal(g, loaded)

        
        # Dump as list of graphs
        list_file = os.path.join(tmp, "graphs_list.json")
        with open(list_file, "w") as f:
            json.dump(graphs, f, cls=GraphJSONEncoder, indent=2)
        with open(list_file, "r") as f:
            reloaded = json.load(f, object_hook=decoder.object_hook_graph)
        assert len(reloaded) == len(graphs)
        assert all(graphs_equal(g, r) for g, r in zip(graphs, reloaded))
        
        
        # Dump as a dict
        dict_file = os.path.join(tmp, "graphs_dict.json")
        dict_graphs = {
            "a": graphs[0],
            "b": graphs[1],
            "c": graphs[2],
        }
        with open(dict_file, "w") as f:
            json.dump(dict_graphs, f, cls=GraphJSONEncoder, indent=2)
        with open(dict_file, "r") as f:
            reloaded = json.load(f, object_hook=decoder.object_hook_graph)
        assert len(reloaded) == len(graphs)
        assert all(graphs_equal(v, reloaded[k]) for k, v in dict_graphs.items())

        