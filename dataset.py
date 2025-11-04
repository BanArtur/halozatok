from abc import ABC, abstractmethod
import json
import os
import random
from typing import Callable

from tqdm import tqdm
import yaml

from distribution import IDistribution, NormalDistribution, UniformDistribution
from graph import Edge, Graph, Vertex
from serialize import GraphJSONDecoder, GraphJSONEncoder
from visualize import GraphVisualization
import matplotlib as plt
from matplotlib.widgets import Button

import algorithm


# Default configuration parameters for edge distribution generation
config_path = "dist_config.yaml"

with open(config_path, "r") as f:
    config: dict[str, float] = yaml.safe_load(f)

normal_chance = config["normal_chance"]
mean_min = config["mean_min"]
mean_max = config["mean_max"]
scale_min = config["scale_min"]
scale_max = config["scale_max"]
bound_min = config["bound_min"]
bound_max = config["bound_max"]


def edge_distribution() -> IDistribution:
    if random.uniform(0, 1) < normal_chance:
        mean = random.uniform(mean_min, mean_max)
        scale = random.uniform(scale_min, scale_max)
        return NormalDistribution(mean, scale)
    else:
        a = random.uniform(bound_min, bound_max)
        b = random.uniform(bound_min, bound_max)
        while a == b:
            a = random.uniform(bound_min, bound_max)
            b = random.uniform(bound_min, bound_max)
        return UniformDistribution(min(a, b), max(a, b))

class GraphDataset(ABC):
    def __init__(
        self,
        base_folder: str,
        reward_distribution: IDistribution,
        edge_distribution_factory: Callable[[], IDistribution] = edge_distribution # we can query this function to get distribution of costs for the edges
    ):
        self.base_folder = base_folder

        self.graphs: list[Graph] = None
        self.decoder = GraphJSONDecoder()

        self.reward_distribution = reward_distribution  # We will sample the vertex rewards during data generation
        self.edge_distribution_factory = edge_distribution_factory

    def load_dataset(self, dataset_name: str) -> None:
        path = os.path.join(self.base_folder, f"{dataset_name}.json")
        assert os.path.exists(path), f"No dataset under {path}"

        with open(path, "r") as f:
            self.graphs = json.load(f, object_hook=self.decoder.object_hook_graph)

    def save_dataset(self, dataset_name: str):
        assert (
            self.graphs is not None
        ), "Empty dataset should not be saved. Generate or load one first"
        os.makedirs(self.base_folder, exist_ok=True)
        path = os.path.join(self.base_folder, f"{dataset_name}.json")
        assert not os.path.exists(path), f"Already a dataset on {path=}"
        with open(path, "w") as f:
            json.dump(self.graphs, f, cls=GraphJSONEncoder)

    def generate_dataset(self, samples: int) -> None:
        """
        Append new entries to self.graphs
        """
        assert samples > 0
        if self.graphs is None:
            self.graphs = []

        for _ in tqdm(range(samples)):
            self.graphs.append(self.generate_graph())

    @abstractmethod
    def generate_graph(self) -> Graph: ...


class SpiderDataset(GraphDataset):
    def __init__(
        self,
        reward_distribution: IDistribution,
        min_legs: int,
        max_legs: int,
        min_lenght: int,
        max_length: int,
    ):
        base = os.path.join("datasets", "spider")
        super().__init__(base, reward_distribution)
        assert min_legs <= max_legs
        assert min_lenght <= max_length

        self.min_legs = min_legs
        self.max_legs = max_legs
        self.min_lenght = min_lenght
        self.max_length = max_length

    def generate_graph(self) -> Graph:
        legs = random.randint(self.min_legs, self.max_legs)
        id = 0
        start = Vertex(id = id, reward=self.reward_distribution.sample())
        result = Graph(start, {0: start}, {})
        for _ in range(legs):
            length = random.randint(self.min_lenght, self.max_length)
            curr = start
            for _ in range(length):
                id += 1
                new = Vertex(
                    id=id,
                    reward=self.reward_distribution.sample(),
                    ancestor=curr,
                )
                edge = Edge(
                    origin=curr,
                    end=new,
                    cost_distribution=self.edge_distribution_factory(),
                )
                curr.out_edges.append(
                    edge
                )
                result.edges[(edge.origin.id, edge.end.id)] = edge
                result.vertices[new.id] = new
                curr = new
        
        return result

class BoundedTreeDataset(GraphDataset):
    def __init__(
        self,
        reward_distribution: IDistribution,
        max_depth: int,
        halt_prob_min: float,
        halt_prob_max: float,
        B: float,
        beta: float,
    ):
        base = os.path.join("datasets", "bounded_trees")
        super().__init__(base, reward_distribution)
        
        self.max_depth = max_depth
        assert halt_prob_min < halt_prob_max
        self.halt_prob_min = halt_prob_min
        self.halt_prob_max = halt_prob_max
        self.B = B
        self.beta = beta
    
    def generate_graph(self) -> Graph:
        id = 0
        start = Vertex(reward=self.reward_distribution.sample(), id=id)
        id += 1
        out1 = Vertex(reward=self.reward_distribution.sample(), ancestor=start, id=id)
        edge = Edge(start, out1, self.edge_distribution_factory())
        start.out_edges.append(edge)
        
        result = Graph(start, {0: start, 1: out1}, {(0, 1): edge})
        
        todos: list[tuple[Vertex, int]] = [(start, 0), (out1, 1)] # list of (vertex, depth)
        index = 0
        while index < len(todos):
            halt_prob = random.uniform(self.halt_prob_min, self.halt_prob_max)
            vertex, depth = todos[index]
            if depth >= self.max_depth:
                index += 1
                continue
            while random.uniform(0, 1) >= halt_prob:
                id += 1
                new = Vertex(
                    reward=self.reward_distribution.sample(),
                    ancestor=vertex,
                    id = id,
                )
                edge = Edge(
                    origin=vertex,
                    end=new,
                    cost_distribution=self.edge_distribution_factory(),
                )
                result.edges[(edge.origin.id, edge.end.id)] = edge
                result.vertices[new.id] = new
                vertex.out_edges.append(edge)
                todos.append((new, depth + 1))
            index += 1
        
        result.lower_mu_cost(self.B, self.beta)
        return result
                
        

if __name__ == "__main__":
    dataset = SpiderDataset(
        reward_distribution=NormalDistribution(0.4, 1.3),
        min_legs=2,
        max_legs=10,
        min_lenght=1,
        max_length=5,
    )
    """
    
    dataset.generate_dataset(1000)
    dataset.save_dataset("base")
    """
    """
    dataset = BoundedTreeDataset(
        reward_distribution=NormalDistribution(0.4, 1.3),
        max_depth=6,
        halt_prob_min=0.25,
        halt_prob_max=0.75,
        B=6,
        beta=8,
    )
    dataset.generate_dataset(1000)
    dataset.save_dataset("base")
    """

    dataset.load_dataset("base")
    print(len(dataset.graphs[5].vertices))

    """
    graphs : list[Graph]

    current = 0
    
    def drawGraph(index):
        G = GraphVisualization()
        G.buildFromGraph(graphs[current])
        G.visualize()
    
    def nextGraph(event):
        global current
        if current<len(graphs)-1:
            current+=1
        drawGraph(current)

    def prevGraph(event):
        global current
        if current >0:
            current-=1
        drawGraph(current)

    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Következő')
    bprev = Button(axprev, 'Előző')
    bnext.on_clicked(nextGraph)
    bprev.on_clicked(prevGraph)
    """
    """
    G = GraphVisualization()
    G.buildFromGraph(dataset.graphs[0])
    G.visualize()
    """
    print(algorithm.nonRisky(dataset.graphs[0],1,2))