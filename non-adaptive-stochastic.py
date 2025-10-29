class Vertex:
    origin:int
    end:int
    secret_cost:float
    reward:float

spider_graph:list[Vertex]
bw_depth_tree:list[Vertex]

def non_adaptive_sp_graph(graph:list[Vertex], source:int, budget:float)->None:
    print("TODO")