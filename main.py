import argparse
from collections import defaultdict
from typing import Any

import pulp
from tqdm import tqdm
from algorithm import (
    LP,
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

# import matplotlib as plt

import json


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


def solve_bounded_trees(graph: Graph, beta: float) -> tuple[tuple[list[tuple[Graph, dict[int, str]]], float, float], float]:
    T = prepare_decompose(graph)
    decomposed = treeDecompose(T, beta, 1)
    list_strategy = transform_to_list(decomposed)
    assert len(T.edges) > 0
    assert len(T.vertices) > 1
    history = run_list_model(T, list_strategy)
    return history, len(decomposed)


def solve_spider_graphs(graph: Graph, B: float = 1.0) -> tuple[tuple[list[tuple[Graph, dict[int, str]]], float, float], float]:
    list_strategy, optimal_upper = spiderNonAdaptive(graph)
    if not list_strategy or list_strategy[0] != 0:
        list_strategy = [0] + list_strategy
    history = run_list_model(graph, list_strategy, B=B)
    return history, optimal_upper


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


def main_phase2():
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

def main_phase3():
    dataset_spider = SpiderDataset(
        reward_distribution=NormalDistribution(0.4, 1.3),
        min_legs=2,
        max_legs=10,
        min_lenght=1,
        max_length=5,
    )
    dataset_spider.load_dataset("real")

    spider_data: list[dict[str, Any]] = []
    for graph_idx, data_graph in enumerate(dataset_spider.graphs[:100]):
        depth = data_graph.depth()
        for epsilon in [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]:
            runs = []
            best_upper = None
            
            for _ in tqdm(range(25)):
                graph = data_graph.copy()
                history, best_upper = solve_spider_graphs(graph, B=1+epsilon)
                runs.append(history[-1])
            assert best_upper is not None
            spider_data.append({
                "epsilon": epsilon,
                "nonadaptive_runs": runs,
                "adaptive_upper_bound": best_upper,
                "alpha": epsilon / 24,
                "vertices": len(data_graph.vertices),
                "depth": depth,
                "legs": len(data_graph.start.out_edges),
                "graph_index": graph_idx,
            })
            avg = sum(runs) / len(runs)
            assert avg >= spider_data[-1]["adaptive_upper_bound"] * spider_data[-1]["alpha"]
            #spider_dict[graph_index][epsilon] = avg / best_upper * 24 / epsilon
            
    with open('spider_data.json', 'w') as fs:
        json.dump(spider_data, fs, indent=2)
    
    # plt.scatter(
    #     [x for val in spider_dict.values() for x, y in val.items()],
    #     [y for val in spider_dict.values() for x, y in val.items()]
    # )
    # plt.plot([i / 1000 for i in range(1000)], [1 for _ in range(1000)])
    # plt.show()
    
    
    bounded_data: list[dict[str, Any]] = []
    Bs = [0.3, 0.5, 0.7, 0.9, 0.95]
    for beta in Bs:
        dataset_bounded = BoundedTreeDataset(
            reward_distribution=NormalDistribution(0.4, 1.3),
            max_depth=7,
            halt_prob_min=0.25,
            halt_prob_max=0.75,
            B=1,
            beta=beta,
        )
        dataset_bounded.load_dataset(f"data_{beta=}")
        
        epsilon = 1 - beta
        for graph_idx, data_graph in enumerate(dataset_bounded.graphs[:100]):
            depth = data_graph.depth()
            
            x, y = LP(data_graph, B=1, t=2)
            optimal_upper = 0.0
            for id, vertex in data_graph.vertices.items():
                optimal_upper += pulp.value(x[id] * vertex.reward)
            runs: list[float] = []
            lens_and_max: list[tuple[float, float]] = []
            
            mu = data_graph.expected_value_max(upper=1)
            
            for _ in tqdm(range(25)):
                graph = data_graph.copy()
                assert len(graph.vertices) > 1
                history, len_of_decomp = solve_bounded_trees(graph, beta=beta)
                assert len_of_decomp < 2 * mu / (epsilon / 2)
                runs.append(history[-1])
                lens_and_max.append((len_of_decomp, 2 * mu / (epsilon / 2)))
                    
            bounded_data.append({
                "epsilon": epsilon,
                "nonadaptive_runs": runs,
                "adaptive_upper_bound": optimal_upper,
                "alpha": (epsilon * epsilon) / (16 * (1 + 1)), # Theorem 13
                "lens_and_max": lens_and_max,
                "vertices": len(data_graph.vertices),
                "depth": depth,
                "graph_index": graph_idx,
                
            })
            avg = sum(runs) / len(runs)
            assert avg >= bounded_data[-1]["adaptive_upper_bound"] * bounded_data[-1]["alpha"]
            # bounded_dict[epsilon].append(avg / optimal_upper * 16 * (1 + 1) / (epsilon * epsilon))
    
    with open('bounded_data.json','w') as fb:
        json.dump(bounded_data, fb, indent=2)
    
    # plt.scatter(
    #     [eps for eps, entry in bounded_dict.items() for y in entry],
    #     [y for eps, entry in bounded_dict.items() for y in entry]
    # )
    # plt.plot([i / 1000 for i in range(1000)], [1 for _ in range(1000)])
    # plt.show()


if __name__ == "__main__":
    main_phase3()
