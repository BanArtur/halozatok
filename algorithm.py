from graph import Graph, Vertex, Edge
import copy

def expectedCostSubTree(T:Graph, v:Vertex, B:float)->int:
    sum = 0
    for vc in v.out_edges:
        sum += vc.cost_distribution.expected_value_max(B)+expectedCostSubTree(T, vc.end,B)
    return sum

def maxCostChild(T:Graph, v:Vertex, B:float) -> list[Vertex,int]:
    vMax = v.out_edges[0].end
    mu_vMax = expectedCostSubTree(T, vMax, B)
    for vc in v.out_edges:
        if mu_vMax<expectedCostSubTree(T,vc.end, B):
            vMax = vc.end
            mu_vMax = expectedCostSubTree(T,vc.end, B)
    return [vMax,mu_vMax]

def treeDecompose(To:Graph,beta:float,B:float)->list[Graph]:
    T = copy.deepcopy(To)
    epsilon = 1-beta
    alpha = 1 - (epsilon/2)
    S = []
    mu_T=expectedCostSubTree(T,T.start,B)
    Si = Graph(T.start,[T.start])
    while(mu_T>alpha):
        v = T.start
        mu_Si = 0
        while(mu_Si+expectedCostSubTree(T,v,B)>alpha):
            vNext = maxCostChild(T,v,B)
            v = vNext[0]
            Si.vertices.append(v)
            edges = v.ancestor.out_edges
            for e in edges:
                if e.end.id==v.id:
                    Si.edges.append(e)
            mu_Si+=vNext[1]
        w=v.ancestor
        mu_Si+=expectedCostSubTree(T,v,B)
        Si.addSubtree(T,v)
        T.remove(v)
        nextChild = maxCostChild(T,w,B)
        while(mu_Si + nextChild[1] <= alpha):
            mu_Si += nextChild[1]
            Si.addSubtree(T, nextChild[0])
            T.remove(nextChild[0])
            nextChild = maxCostChild(T,w,B)
        S.append(copy.deepcopy(Si))
        mu_T = mu_T - mu_Si+cost(T.start,w)
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