import argparse
from algorithm import (
    prepare_decompose,
    run_list_model,
    spiderNonAdaptive,
    transform_to_list,
    treeDecompose,
)
from dataset import BoundedTreeDataset, GraphDataset, SpiderDataset
from distribution import NormalDistribution

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx

from graph import Graph


def get_dataset(g_type: str) -> GraphDataset:
    if g_type == "bounded":
        return BoundedTreeDataset(
            reward_distribution=NormalDistribution(1.5, 0.8),
            max_depth=6,
            halt_prob_min=0.25,
            halt_prob_max=0.75,
            B=1,
            beta=0.9,
        )
    elif g_type == "spider":
        return SpiderDataset(
            reward_distribution=NormalDistribution(1.5, 0.6),
            min_legs=2,
            max_legs=4,
            min_lenght=1,
            max_length=5,
        )
    else:
        raise ValueError(f"No known type named: {g_type}")


def solve_bounded_trees(graph: Graph) -> list[tuple[Graph, dict[int, str]]]:
    T = prepare_decompose(graph)
    decomposed = treeDecompose(T, 0.9, 1)
    list_strategy = transform_to_list(decomposed)
    history = run_list_model(T, list_strategy)
    return history[0]


def solve_spider_graphs(graph: Graph) -> list[tuple[Graph, dict[int, str]]]:
    list_strategy = spiderNonAdaptive(graph)
    if not list_strategy or list_strategy[0] != 0:
        list_strategy = [0] + list_strategy
    history = run_list_model(graph, list_strategy)
    return history[0]


def show(list_model: list[tuple[Graph, dict[int, str]]]):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    current = 0

    G = nx.DiGraph()
    pos: dict = {}

    def buildGraph(graph):
        for v in graph.vertices.keys():
            G.add_node(v)
        for e in graph.edges.keys():
            G.add_edge(
                graph.edges[e].origin.id,
                graph.edges[e].end.id,
                lenght=graph.edges[e].cost_distribution.expected_value(),
            )
        return nx.spring_layout(G)

    def drawGraph(index, pos):
        colorMap = []
        labelMap = {}
        edgeLabelMap = {}
        for node in G:
            colorMap.append(list_model[index][1][node])
            labelMap[node] = f"{list_model[index][0].vertices[node].reward:.2f}"
        for u, v in G.edges():
            if list_model[index][0].edges[(u, v)].secret_cost is not None:
                edgeLabelMap[(u, v)] = (
                    f"{list_model[index][0].edges[(u,v)].secret_cost:.2f}"
                )
            else:
                edgeLabelMap[(u, v)] = str(
                    list_model[index][0].edges[(u, v)].cost_distribution
                )
        ax.clear()
        nx.draw_networkx(G, pos, node_color=colorMap, with_labels=False, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labelMap, ax=ax)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edgeLabelMap, rotate=False, ax=ax
        )
        plt.draw()

    def nextGraph(event):
        nonlocal current
        if current < len(list_model) - 1:
            current += 1
        drawGraph(current, pos)

    def prevGraph(event):
        nonlocal current
        if current > 0:
            current -= 1
        drawGraph(current, pos)

    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Következő")
    bprev = Button(axprev, "Előző")
    bnext.on_clicked(nextGraph)
    bprev.on_clicked(prevGraph)

    pos = buildGraph(list_model[0][0])
    drawGraph(current, pos)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Assignment 2")

    parser.add_argument(
        "mode",
        choices=["spider", "bounded"],
        help="Select graph type: 'spider' or 'bounded'.",
    )

    parser.add_argument(
        "--example",
        type=int,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    spider_best = [0, 9, 10]
    bounded_best = [15, 28, 29]

    numbers = spider_best if args.mode == "spider" else bounded_best
    dataset = get_dataset(args.mode)

    dataset.load_dataset("best")
    graph = dataset.graphs[numbers[args.example]]
    history = (
        solve_spider_graphs(graph)
        if args.mode == "spider"
        else solve_bounded_trees(graph)
    )
    show(history)


if __name__ == "__main__":
    main()
