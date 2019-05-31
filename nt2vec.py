from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np


class NT2VEC:

    PROB_KEY = "probabilities"
    ATTR_PROB_KEY = "attr"
    NETWORK_PROB_KEY = "network"

    def __init__(self, graph, attr, dim=10, knn=10):
        self.graph = graph  # networkX graph
        self.attr = attr  # node attributes
        self.dim = dim  # length of the output vectors
        self.knn = knn  # number of neighbors to use in node_attr
        self.p = 1  # node2vec return parameter
        self.q = 1  # node2vec inout parameter
        self.t = 0.5  # parameter that controls where to sample walks (0 is fully network, 1 is no network)
        self.num_walks = 10  # number of walks to sample
        self.walk_length = 30  # length of samples
        self.d_graph = defaultdict(dict)
        self.d_attr = defaultdict(dict)

    def precompute_nearest_neighbors(self):

        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto', metric="cosine").fit(self.attr)
        distances, indices = nbrs.kneighbors()  # Gets distances and nearest neighbors
        similarities = -np.log(distances)  # (1-distances)

        # # normalize similarities into probability
        for i in range(len(similarities)):
            similarities[i] = similarities[i] / sum(similarities[i])

        return similarities, indices

    def precompute_attr_probabilities(self):

        d_attr = self.d_attr

        similarities, nn_indices = self.precompute_nearest_neighbors()

        for s in range(len(nn_indices)):

            if self.PROB_KEY not in d_attr[s]:
                d_attr[s][self.PROB_KEY] = dict()

            for i in range(len(nn_indices[int(s)])):
                node = nn_indices[int(s)][i]
                d_attr[s][self.PROB_KEY][node] = similarities[int(s)][i]

    def generate_attr_walk(self):
        d_attr = self.d_attr
        walks = list()

        for n_walk in range(self.num_walks):  # iterate num_walks per node
            for node in d_attr:  # do a walk starting from each node

                walk = [node]

                while len(walk) < self.walk_length:
                    destinations = list(d_attr[walk[-1]][self.PROB_KEY].keys())  # possible destinations
                    probabilities = list(d_attr[walk[-1]][self.PROB_KEY].values())  # list of probabilities
                    walk_to = np.random.choice(destinations, size=1, p=probabilities)[0]  # make a choice
                    walk.append(walk_to)
                walks.append(walk)

        return walks



