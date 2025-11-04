from algorithm import run_list_model, transform_to_list, treeDecompose
from dataset import BoundedTreeDataset
from distribution import NormalDistribution


dataset = BoundedTreeDataset(
    reward_distribution=NormalDistribution(0.6, 1.3),
    max_depth=6,
    halt_prob_min=0.25,
    halt_prob_max=0.75,
    B=1,
    beta=0.9,
)

#dataset.generate_dataset(200)
#dataset.save_dataset("normal")
#exit()
dataset.load_dataset("normal")
graph = max(dataset.graphs, key=lambda x: len(x.vertices))

print(f"{len(graph.vertices)}=")
x = treeDecompose(graph, 0.9, 1)
list_strategy = transform_to_list(x)
print(f"{list_strategy=}")
print(f"{len(list_strategy)=}")
print(len(run_list_model(graph, list_strategy)))

