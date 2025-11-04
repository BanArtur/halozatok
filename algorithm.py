from graph import Graph, Vertex, Edge
import copy
import pulp
import random

def expectedCostSubTree(T:Graph, v:Vertex, B:float)->float:
    sum = 0
    for vc in v.out_edges:
        sum += vc.cost_distribution.expected_value_max(B)+expectedCostSubTree(T, vc.end,B)
    return sum

def maxCostChild(T:Graph, v:Vertex, B:float) -> tuple[Vertex,float]:
    vMax = v.out_edges[0].end
    mu_vMax = expectedCostSubTree(T, vMax, B)
    for vc in v.out_edges:
        if mu_vMax<expectedCostSubTree(T,vc.end, B):
            vMax = vc.end
            mu_vMax = expectedCostSubTree(T,vc.end, B)
    return (vMax,mu_vMax)

def treeDecompose(To:Graph,beta:float,B:float)->list[Graph]:
    T = To.copy()
    epsilon = 1-beta
    alpha = 1 - (epsilon/2)
    S = []
    mu_T=expectedCostSubTree(T,T.start,B)
    Si = Graph(T.start,{T.start.id: T.start}, {})
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
        mu_T = mu_T - mu_Si+cost(T.start,w,B)
    S.append(T)
    return S

def nonRisky(T:Graph, B:float) -> list[Vertex]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    x:list[float]
    xi=0
    T_1:list[Vertex] = []
    T_2:list[Vertex] = []
    R_1 = 0
    R_2 = 0
    for v in T.vertices:
        if x[v] == 1:
            T_1.append(v)
            R_1+=v.reward
        elif x[v] == xi:
            T_2.append(v)
            R_2+=v.reward
    if R_1>R_2:
        return T_1
    else:
        return T_2


def risky(T:Graph, B:float) -> list[Vertex]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    leg:list[Vertex] = []
    for v in T.start.out_edges:
        leg = []
        sum = 0
        v_r = v.end
        cond = True
        while cond:
            leg.append(v_r)
            sum = x[v] * cost(v_r.ancestor,v_r,B)
            if len(v_r.out_edges)>0:
                v_r = v_r.out_edges[0].end
            else:
                cond = False
        if random.uniform(0,1) < sum/2:
            return leg
    return leg

def isRisky(T:Graph, v:Vertex, B:float) -> bool:
    sumCost=0.0
    if v.id!=T.start.id:
        sumCost = cost(v.ancestor, v, B)
    return sumCost>0.5

def spiderNonAdaptive(T:Graph, B:float) -> list[Vertex]:
    T_risky:Graph  = T.copy()
    T_non:Graph = T.copy()
    for v in T.vertices:
        if isRisky(v):
            T_non.vertives[v.id].reward = 0
        else:
            T_risky.vertices[v.id].reward = 0
    L_r = risky(T_risky)
    R_r = 0.0
    for v in L_r:
        R_r += v.reward
    L_n = nonRisky(T_non)
    R_n = 0.0
    for v in L_n:
        R_n += v.reward
    if R_r > R_n:
        return L_r
    else:
        return L_n
    
def cost(o:Vertex, e:Vertex, B:float) -> float:
    if o.id == e.id:
        return 0.0
    sum:float = 0.0
    for edge in e.ancestor.out_edges:
        if edge.end.id == e.id:
            sum+= edge.cost_distribution.expected_value_max(B)
    return sum+cost(o,e.ancestor,B)