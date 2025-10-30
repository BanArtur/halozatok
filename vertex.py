from dataclasses import dataclass


@dataclass
class Vertex:
    secret_cost: float
    reward: float


@dataclass
class Edge:
    origin: Vertex
    end: Vertex


def non_adaptive_sp_graph(graph: list[Vertex], source: int, budget: float) -> None:
    print("TODO")


if __name__ == "__main__":
    spider_graph: list[Vertex]
    bw_depth_tree: list[Vertex]
