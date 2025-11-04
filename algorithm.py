from graph import Graph, Vertex, Edge
import copy

def expectedCostSubTree(v: Vertex, B: float) -> float:
    sum = 0
    for vc in v.out_edges:
        sum += vc.cost_distribution.expected_value_max(B) + expectedCostSubTree(vc.end,B)
    return sum

def maxCostChild(v:Vertex, B:float) -> tuple[Vertex,float]:
    return max([(edge.end, expectedCostSubTree(edge.end, B)) for edge in v.out_edges], key=lambda x: x[1])

def treeDecompose(To:Graph,beta:float,B:float)->list[Graph]:
    T = To.copy()
    epsilon = 1-beta
    alpha = 1 - (epsilon/2)
    S = []
    mu_T=expectedCostSubTree(T.start,B)
    
    while(mu_T>alpha):
        r = Vertex(id=T.start.id, reward=T.start.reward, ancestor=None)
        Si = Graph(r,{r.id: r}, {})
        v = T.start
        mu_Si = 0
        anc = r
        while mu_Si+expectedCostSubTree(v,B)>alpha:
            v, _ = maxCostChild(v,B)
            v_copy = Vertex(v.id, reward=v.reward, ancestor=anc)
            edge = Edge(anc, v_copy, cost_distribution=T.edges[(anc.id, v.id)].cost_distribution)
            anc.out_edges.append(edge)
            Si.vertices[v.id] = v
            Si.edges[(anc.id, v.id)] = edge
            mu_Si+=T.edges[(anc.id, v.id)].cost_distribution.expected_value_max(B)
            anc = v_copy
        w = v.ancestor
        assert w is not None
        mu_Si+=expectedCostSubTree(v,B)
        Si.addSubtree(T,v)
        T.removeVertex(v)
        if w.out_edges:
            nextChild = maxCostChild(w, B)
            while nextChild is not None and mu_Si + nextChild[1] <= alpha:
                mu_Si += nextChild[1]
                Si.addSubtree(T, nextChild[0])
                T.removeVertex(nextChild[0])
                nextChild = maxCostChild(w,B) if w.out_edges else None
        S.append(Si)
        mu_T = expectedCostSubTree(T.start,B)
        
    S.append(T)
    return S

def nonRisky(T:Graph) -> list[Vertex]:
    print("todo")

def risky(T:Graph) -> list[Vertex]:
    print("todo")

def isRisky(T:Graph, v:Vertex) -> bool:
    print("todo")

def spiderNonAdaptive(T:Graph) -> list[Vertex]:
    print("todo")