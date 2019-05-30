from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np


class NT2VEC:

    PROB_KEY = "probabilities"
    ATTR_PROB_KEY = "attr"
    NETWORK_PROB_KEY = "network"

    def __init__(self, graph, attr, dim=10, knn=30):
        self.graph = graph  # networkX graph
        self.attr = attr  # node attributes
        self.dim = dim  # length of the output vectors
        self.knn = knn  # number of neighbors to use in node_attr
        self.p = 1  # node2vec return parameter
        self.q = 1  # node2vec inout parameter
        self.walks = 10  # number of walks to sample
        self.walk_length = 20  # length of samples
        self.d_graph = defaultdict(dict)

    def precompute_nearest_neighbors(self):

        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto', metric="cosine").fit(self.attr)
        distances, indices = nbrs.kneighbors()  # Gets distances and nearest neighbors
        similarities = -np.log(distances)  # (1-distances)

        # # normalize similarities into probability
        for i in range(len(similarities)):
            similarities[i] = similarities[i] / sum(similarities[i])

        return similarities, indices

    def precompute_probabilities(self):

        d_graph = self.d_graph
        nodes_gen = self.graph.nodes()

        similarities, nn_indices = self.precompute_nearest_neighbors()

        for s in nodes_gen:

            if self.PROB_KEY not in d_graph[s]:
                d_graph[s][self.PROB_KEY] = {self.NETWORK_PROB_KEY: {}, self.ATTR_PROB_KEY: {}}

            for node in nn_indices:
                d_graph[s][self.PROB_KEY][self.ATTR_PROB_KEY]




