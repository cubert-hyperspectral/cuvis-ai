import networkx as nx

from cuvis_ai.node import Node


class ShapeValidator:
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        self.input_shape = input_shape

    def verify(self, node: Node, inshape: tuple[int, int, int] = None):
        if inshape is None:
            inshape = self.input_shape


class GraphValidator:
    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    def verify(self) -> bool:
        if len(list(nx.simple_cycles(self.graph))) > 0:
            return False

        return True
