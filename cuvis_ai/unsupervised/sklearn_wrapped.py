import sklearn.cluster
import sklearn.mixture

from cuvis_ai.node.wrap import make_node


@make_node
class KMeans(sklearn.cluster.KMeans):
    pass


@make_node
class MeanShift(sklearn.cluster.MeanShift):
    pass


@make_node
class GMM(sklearn.mixture.GaussianMixture):
    pass
