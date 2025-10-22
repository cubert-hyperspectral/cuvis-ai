import sklearn.decomposition

from cuvis_ai.node.wrap import make_node


@make_node
class PCA(sklearn.decomposition.PCA):
    pass


@make_node
class NMF(sklearn.decomposition.NMF):
    pass
