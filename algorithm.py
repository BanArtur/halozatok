from graph import Graph, Vertex, Edge
import copy
import pulp
import random

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

def nonRisky(T:Graph, B:float,t:float) -> list[Vertex]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    x={}
    graphKeys = []
    for key in T.vertices.keys():
        if key!=T.start.id:
            graphKeys.append(key)
    x = pulp.LpVariable.dicts("VertexesX",T.vertices, lowBound = 0, upBound=1)
    model+=pulp.lpSum(x[id]*T.vertices[id].reward for id in T.vertices.keys())
    model+=pulp.lpSum(x[id]*T.edges[(T.vertices[id].ancestor.id,id)].cost_distribution.expected_value_max(B) for id in graphKeys ) <= 1
    for key in graphKeys:
        model+=x[key]<=x[T.vertices[key].ancestor.id]
    model.solve()
    T_1:list[Vertex] = []
    T_2:list[Vertex] = []
    R_1 = 0
    R_2 = 0
    xi = 0
    for v in T.vertices.keys():
        if pulp.value(x[v]) == 1.0:
            T_1.append(T.vertices[v])
            R_1+=T.vertices[v].reward
        elif pulp.value(x[v]) == xi:
            T_2.append(T.vertices[v])
            R_2+=T.vertices[v].reward
    if R_1>R_2:
        return T_1
    else:
        return T_2


def risky(T:Graph, B:float, t:float) -> list[Vertex]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    x={}
    y={}
    x = pulp.LpVariable.dicts("VertexesX",T.vertices, lowBound = 0, upBound=1)
    y = pulp.LpVariable.dicts("VertexesY",T.vertices, lowBound = 0, upBound=1)
    graphKeys = []
    for key in T.vertices.keys():
        if key!=T.start.id:
            graphKeys.append(key)
    model+=pulp.lpSum([y[id]*T.vertices[id].reward for id in T.vertices.keys()])
    model+=pulp.lpSum(x[id]*T.edges[(T.vertices[id].ancestor.id,id)].cost_distribution.expected_value_max(1) for id in graphKeys )<=t
    for id in graphKeys:
        v_o = T.vertices[id]
        v_r = v_o
        cond = True
        while cond:
            model+=y[id]<=costSampling(T,v_r,v_o,B)*x[v_r.id]
            if v_r.ancestor is not None:
                v_r = v_r.ancestor
            else:
                cond = False
    model.solve()
    leg:list[Vertex] = []
    for v in T.start.out_edges:
        leg = []
        sum = 0
        v_r = v.end
        cond = True
        while cond:
            leg.append(v_r)
            sum = pulp.value(x[v.end.id]) * cost(v_r.ancestor,v_r,B)
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

def costSampling(g:Graph, v1:Vertex, v2:Vertex, B:float) -> float:
    sum = 0
    for i in range(100):
        if cost(v1,v2,B)<=1:
            sum+=1
    return sum/100