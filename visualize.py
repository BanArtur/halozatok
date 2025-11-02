import networkx as nx
import matplotlib.pyplot as plt
from graph import Graph, Edge, Vertex

class GraphVisualization:
    def __init__(self):
        self.visual=[]
    
    def addEdge(self,a:int,b:int,c:float):
        temp = [a,b,c]
        self.visual.append(temp)

    def buildFromGraph(self,g:Graph):
        for e in g.edges:
            self.addEdge(e.origin.id,e.end.id,e.cost_distribution.expected_value())

    def visualize(self):
        G = nx.DiGraph()
        for e in self.visual:
            G.add_edge(e[0],e[1], lenght=e[2])
        pos = nx.spring_layout(G)
        nx.draw_networkx(G,pos)
        nx.draw_networkx_edge_labels(G,pos,rotate=False)
        plt.show()