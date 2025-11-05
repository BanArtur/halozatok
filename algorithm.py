from graph import Graph, Vertex, Edge
import copy
import pulp
import random

COST_SAMPLE_SIZE = 100


def expectedCostSubTree(v: Vertex, B: float) -> float:
    sum = 0
    for vc in v.out_edges:
        sum += vc.cost_distribution.expected_value_max(B) + expectedCostSubTree(
            vc.end, B
        )
    return sum


def maxCostChild(v: Vertex, B: float) -> tuple[Vertex, float]:
    return max(
        [(edge.end, expectedCostSubTree(edge.end, B)) for edge in v.out_edges],
        key=lambda x: x[1],
    )


def prepare_decompose(T: Graph) -> Graph:
    x = LPHat(T, B=1.0, t=0.5)
    root = Vertex(T.start.id, T.start.reward, ancestor=None)
    todos: list[int] = [root.id]
    index = 0
    new_graph = Graph(start=root, vertices={root.id: root}, edges={})
    while index < len(todos):
        id = todos[index]
        current = new_graph.vertices[id]
        big_vertex = T.vertices[id]
        for edge in big_vertex.out_edges:
            candidate = edge.end
            if pulp.value(x[candidate.id]) > 0:
                todos.append(candidate.id)
                new = Vertex(candidate.id, reward=candidate.reward, ancestor=current)
                new_graph.vertices[new.id] = new
                new_edge = Edge(current, new, edge.cost_distribution)
                current.out_edges.append(new_edge)
                new_graph.edges[(current.id, new.id)] = new_edge

        index += 1

    new_graph.check_reachable()
    return new_graph


def treeDecompose(To: Graph, beta: float, B: float) -> list[Graph]:
    T = To.copy()
    epsilon = 1 - beta
    alpha = 1 - (epsilon / 2)
    S = []
    mu_T = expectedCostSubTree(T.start, B)

    while mu_T > alpha:
        r = Vertex(id=T.start.id, reward=T.start.reward, ancestor=None)
        Si = Graph(r, {r.id: r}, {})
        v = T.start
        mu_Si = 0
        anc = r
        while mu_Si + expectedCostSubTree(v, B) > alpha:
            v, _ = maxCostChild(v, B)
            v_copy = Vertex(v.id, reward=v.reward, ancestor=anc)
            edge = Edge(
                anc, v_copy, cost_distribution=T.edges[(anc.id, v.id)].cost_distribution
            )
            anc.out_edges.append(edge)
            Si.vertices[v.id] = v_copy
            Si.edges[(anc.id, v.id)] = edge
            mu_Si += T.edges[(anc.id, v.id)].cost_distribution.expected_value_max(B)
            anc = v_copy
        w = v.ancestor
        assert w is not None
        mu_Si += expectedCostSubTree(v, B)
        Si.addSubtree(T, v)
        T.removeVertex(v)
        if w.out_edges:
            nextChild = maxCostChild(w, B)
            while nextChild is not None and mu_Si + nextChild[1] <= alpha:
                mu_Si += nextChild[1]
                Si.addSubtree(T, nextChild[0])
                T.removeVertex(nextChild[0])
                nextChild = maxCostChild(w, B) if w.out_edges else None
        S.append(Si)
        mu_T = expectedCostSubTree(T.start, B)

    S.append(T)
    return S


def transform_to_list(sub_graphs: list[Graph]) -> list[int]:
    graph = random.choice(sub_graphs)
    result: list[int] = []
    root = graph.start
    todos: list[Vertex] = [root]
    index = 0
    while index < len(todos):
        vertex = todos[index]
        if vertex.id not in result:
            result.append(vertex.id)
        for edge in vertex.out_edges:
            todos.append(edge.end)
        index += 1

    return result


def run_list_model(
    T: Graph, strategy: list[int], B: float = 1.0
) -> tuple[list[tuple[Graph, dict[int, str]]], float, float]:
    T = T.copy()
    R: float = 0.0
    budget: float = 0.0
    colors: dict[int, str] = {v: "lightgray" for v in T.vertices}
    history: list[tuple[int, dict[int, str]]] = []
    for id in strategy:
        if id == T.start.id:
            R += T.vertices[id].reward
            history.append((T, colors))
            colors = dict(colors)
            T = T.copy()
            continue
        edge = T.edges[(T.vertices[id].ancestor.id, id)]

        edge.probe()
        if budget + edge.secret_cost > B:
            colors[id] = "red"
            history.append((T, colors))
            break
        else:
            budget += edge.secret_cost
            R += T.vertices[id].reward
            colors[id] = "green"
            history.append((T, colors))
            T = T.copy()
            colors = dict(colors)

    return history, budget, R


def LPHat(T: Graph, B: float, t: float) -> dict[int, float]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    x = {}
    graphKeys = []
    for key in T.vertices.keys():
        if key != T.start.id:
            graphKeys.append(key)
    x = pulp.LpVariable.dicts("VertexesX", T.vertices, lowBound=0, upBound=1)
    model += pulp.lpSum(x[id] * T.vertices[id].reward for id in T.vertices.keys())
    model += (
        pulp.lpSum(
            x[id]
            * T.edges[
                (T.vertices[id].ancestor.id, id)
            ].cost_distribution.expected_value_max(B)
            for id in graphKeys
        )
        <= t
    )
    for key in graphKeys:
        model += x[key] <= x[T.vertices[key].ancestor.id]
    model.solve()
    return x


def nonRisky(T: Graph, B: float, t: float) -> list[Vertex]:
    x = LPHat(T, B, t)
    T_1: list[Vertex] = []
    T_2: list[Vertex] = []
    R_1 = 0
    R_2 = 0
    for v in T.vertices.keys():
        if pulp.value(x[v]) == 1.0:
            T_1.append(T.vertices[v])
            R_1 += T.vertices[v].reward
        elif pulp.value(x[v]) != 0.0:
            T_2.append(T.vertices[v])
            R_2 += T.vertices[v].reward
    if R_1 > R_2:
        return T_1
    else:
        return T_2


def risky(T: Graph, B: float, t: float) -> list[Vertex]:
    model = pulp.LpProblem("Maximalization", pulp.LpMaximize)
    x = {}
    y = {}
    x = pulp.LpVariable.dicts("VertexesX", T.vertices, lowBound=0, upBound=1)
    y = pulp.LpVariable.dicts("VertexesY", T.vertices, lowBound=0, upBound=1)
    graphKeys = []
    for key in T.vertices.keys():
        if key != T.start.id:
            graphKeys.append(key)
    model += pulp.lpSum([y[id] * T.vertices[id].reward for id in T.vertices.keys()])
    model += (
        pulp.lpSum(
            x[id]
            * T.edges[
                (T.vertices[id].ancestor.id, id)
            ].cost_distribution.expected_value_max(B)
            for id in graphKeys
        )
        <= t
    )
    for id, v_o in T.vertices.items():
        v_r = v_o
        cond = True
        edges: list[Edge] = []
        while cond:
            model += y[id] <= costSampling(edges) * x[v_r.id]
            if v_r.ancestor is not None:
                edges.append(T.edges[(v_r.ancestor.id, v_r.id)])
                v_r = v_r.ancestor
            else:
                cond = False
    model.solve()
    legs: list[tuple[list[Vertex], float]] = []
    for v in T.start.out_edges:
        leg: list[Vertex] = []
        s = 0
        v_r = v.end
        cond = True
        while cond:
            leg.append(v_r)

            s = pulp.value(x[v.end.id]) * cost(v_r.ancestor, v_r, B)
            assert len(v_r.out_edges) <= 1

            if v_r.out_edges:
                v_r = v_r.out_edges[0].end
            else:
                cond = False
        legs.append((leg, s))
    probabilities = [entry[1] for entry in legs]
    s_prob = sum(probabilities)
    assert s_prob >= 0
    if s_prob == 0:
        return random.choice(legs)[0]
    else:
        probabilities = [p / s_prob for p in probabilities]
        choice = random.choices(legs, weights=probabilities, k=1)[0]
        return choice[0]


def isRisky(T: Graph, v: Vertex, B: float) -> bool:
    return cost(T.start, v, B) > 0.5


def spiderNonAdaptive(T: Graph) -> list[int]:
    T_risky: Graph = T.copy()
    T_non_risky: Graph = T.copy()
    for id, v in T.vertices.items():
        if isRisky(T, v, B=1.0):
            T_non_risky.vertices[id].reward = 0
        else:
            T_risky.vertices[id].reward = 0
    L_r = risky(T_risky, B=1.0, t=2)
    R_r = 0.0
    for v in L_r:
        R_r += v.reward
    L_n = nonRisky(T_non_risky, B=1.0, t=0.5)
    R_n = 0.0
    for v in L_n:
        R_n += v.reward
    if R_r > R_n:
        better = L_r
    else:
        better = L_n

    return [vertex.id for vertex in better]


def cost(o: Vertex, e: Vertex, B: float) -> float:
    sum: float = 0.0
    while e.id != o.id:
        edge = [el for el in e.ancestor.out_edges if el.end.id == e.id][0]
        sum += edge.cost_distribution.expected_value_max(B)
        e = e.ancestor
    return sum


def costSampling(edges: list[Edge]) -> float:
    s = 0
    for _ in range(COST_SAMPLE_SIZE):
        if sum([edge.cost_distribution.sample() for edge in edges]) <= 1:
            s += 1
    return s / COST_SAMPLE_SIZE
