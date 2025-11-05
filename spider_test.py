from algorithm import spiderNonAdaptive
from dataset import SpiderDataset
from distribution import NormalDistribution


dataset = SpiderDataset(
    reward_distribution=NormalDistribution(1.5, 1.0),
    min_legs=2,
    max_legs=10,
    min_lenght=1,
    max_length=5,
)

#dataset.generate_dataset(200)
#dataset.save_dataset("real")

dataset.load_dataset("real")
for graph in dataset.graphs:
    x = spiderNonAdaptive(graph)
    print(f"{len(graph.vertices)=}")
    print(f"{len(x)=}")