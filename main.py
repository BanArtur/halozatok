from algorithm import run_list_model, transform_to_list, treeDecompose
from dataset import BoundedTreeDataset
from distribution import NormalDistribution

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx


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
graph = min(dataset.graphs, key=lambda x: len(x.vertices))
graph = dataset.graphs[6]

print(f"{len(graph.vertices)}=")
x = treeDecompose(graph, 0.9, 1)
list_strategy = transform_to_list(x)
print(f"{list_strategy=}")
print(f"{len(list_strategy)=}")
list_model = run_list_model(graph, list_strategy)
print(len(list_model))

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
current = 0

G = nx.DiGraph()
pos:dict = {}

def buildGraph(graph):
    for v in graph.vertices.keys():
        G.add_node(v)
    for e in graph.edges.keys():
        G.add_edge(graph.edges[e].origin.id,graph.edges[e].end.id, lenght=graph.edges[e].cost_distribution.expected_value())
    return nx.spring_layout(G)

def drawGraph(index,pos):
    colorMap = []
    labelMap = {}
    edgeLabelMap = {}
    for node in G:
        colorMap.append(list_model[0][index][1][node])
        labelMap[node] = f"{list_model[0][index][0].vertices[node].reward:.2f}"
    for u,v in G.edges():
        if list_model[0][index][0].edges[(u,v)].secret_cost is not None:
            edgeLabelMap[(u,v)] = f"{list_model[0][index][0].edges[(u,v)].secret_cost:.2f}"
        else:
            edgeLabelMap[(u,v)] = str(list_model[0][index][0].edges[(u,v)].cost_distribution)
    ax.clear()
    nx.draw_networkx(G,pos,node_color=colorMap,with_labels=False, ax=ax)
    nx.draw_networkx_labels(G,pos,labels=labelMap, ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabelMap,rotate=False, ax=ax)
    plt.draw()

def nextGraph(event):
    global current
    if current<len(list_model[0])-1:
        current+=1
    drawGraph(current,pos)

def prevGraph(event):
    global current
    if current >0:
        current-=1
    drawGraph(current,pos)

axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Következő')
bprev = Button(axprev, 'Előző')
bnext.on_clicked(nextGraph)
bprev.on_clicked(prevGraph)

pos = buildGraph(list_model[0][0][0])
drawGraph(current,pos)

plt.show() 