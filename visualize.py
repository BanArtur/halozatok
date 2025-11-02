import networkx as nx
import matplotlib.pyplot as plt
from graph import Graph, Edge, Vertex

class GraphVisualization:
    def __init__(self):
        self.visual=[]
    
    def addEdge(self,a:int,b:int):
        temp = [a,b]
        self.visual.append(temp)

    def buildFromGraph(self,g:Graph):
        graphIdDict = {}
        i = 0
        for v in g.vertices:
            graphIdDict[v] = i
            i+=1
        for e in g.edges:
            temp = [graphIdDict[e.origin],graphIdDict[e.end]]
            self.visual.append(temp)

    def visualize(self):
        G = nx.DiGraph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()