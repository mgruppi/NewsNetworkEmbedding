from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np
import gensim


class NT2VEC:

    PROB_KEY = "probabilities"
    ATTR_PROB_KEY = "attr"
    NETWORK_PROB_KEY = "network"

    def __init__(self, graph, attr, dim=10, knn=10, workers=12):
        self.graph = graph  # networkX graph
        self.attr = attr  # node attributes
        self.dim = dim  # length of the output vectors
        self.knn = knn  # number of neighbors to use in node_attr
        self.p = 1  # node2vec return parameter
        self.q = 1  # node2vec inout parameter
        self.t = 0.5  # parameter that controls where to sample walks (0 is fully network, 1 is no network)
        self.num_walks = 10  # number of walks to sample
        self.walk_length = 30  # length of samples
        self.walks = list()
        self.d_graph = defaultdict(dict)
        self.d_attr = defaultdict(dict)

        self.workers = workers

    def precompute_nearest_neighbors(self):

        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto', metric="cosine").fit(self.attr)
        distances, indices = nbrs.kneighbors()  # Gets distances and nearest neighbors
        similarities = -np.log(distances)  # (1-distances)

        # # normalize similarities into probability
        for i in range(len(similarities)):
            sum_ = sum(similarities[i])
            if sum_ != 0:
                similarities[i] = similarities[i] / (sum(similarities[i]))



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

    def generate_single_attr_walk(self, source):
        d_attr = self.d_attr
        walk = [source]

        while len(walk) < self.walk_length:
            destinations = list(d_attr[walk[-1]][self.PROB_KEY].keys())  # possible destinations
            probabilities = list(d_attr[walk[-1]][self.PROB_KEY].values())  # list of probabilities
            if not np.any(probabilities):
                break
            walk_to = np.random.choice(destinations, size=1, p=probabilities)[0]  # make a choice
            walk.append(walk_to)
        walk = list(map(str, walk))  # make sure walk contains only strings

        return walk

    def generate_attr_walks(self):
        d_attr = self.d_attr
        walks = list()

        for n_walk in range(self.num_walks):  # iterate num_walks per node
            for node in d_attr:  # do a walk starting from each node

                walk = [node]

                while len(walk) < self.walk_length:
                    destinations = list(d_attr[walk[-1]][self.PROB_KEY].keys())  # possible destinations
                    probabilities = list(d_attr[walk[-1]][self.PROB_KEY].values())  # list of probabilities
                    if not np.any(probabilities):
                        break
                    walk_to = np.random.choice(destinations, size=1, p=probabilities)[0]  # make a choice
                    walk.append(walk_to)
                    walk = list(map(str, walk))  # make sure walk contains only strings
                walks.append(walk)

        return walks

    def generate_walks(self):
        for n_walk in range(self.num_walks):
            for source in self.d_attr:
                self.walks.append(self.generate_single_attr_walk(source))

        return self.walks

    def fit(self, **skip_gram_params):

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params["size"] = self.dim

        if 'sg' not in skip_gram_params:
            skip_gram_params["sg"] = 1  # 1 - use skip-gram; otherwise, use CBOW

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)




