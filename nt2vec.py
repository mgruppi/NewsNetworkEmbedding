from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np
import gensim
import random


class NT2VEC:
    PROB_KEY = "probabilities"
    PROBABILITIES_KEY = PROB_KEY
    ATTR_PROB_KEY = "attr"
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, attr, labels=None, dim=200, knn=10, workers=12, num_walks=100, walk_length=50, sampling_strategy=None,
                 weight_key='weight', sg=1, p=0.4, q=0.3, t=0.3):
        self.graph = graph  # networkX graph
        self.attr = attr  # node attributes
        self.labels = labels  # labels for assignment if output must be in terms of names rather than int ids
        self.dim = dim  # length of the output vectors
        self.knn = knn  # number of neighbors to use in node_attr
        self.p = p  # node2vec return parameter
        self.q = q  # node2vec inout parameter
        self.t = t  # parameter that controls where to sample walks (0 is fully network, 1 is no network)
        self.num_walks = num_walks  # number of walks to sample
        self.walk_length = walk_length  # length of samples
        self.walks = list()
        self.weight_key = weight_key
        self.d_graph = defaultdict(dict)
        self.d_attr = defaultdict(dict)
        self.sg=sg

        self.workers = workers

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy


    def precompute_nearest_neighbors(self):

        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto', metric="cosine").fit(self.attr)
        distances, indices = nbrs.kneighbors()  # Gets distances and nearest neighbors
        # similarities = -np.log(distances)  # (1-distances)
        similarities = 1 - distances

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

    # Node2Vec NODES
    def precompute_network_probabilities(self):

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes()

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

    def generate_single_network_walk(self, source):

        walk = [source]
        d_graph = self.d_graph
        while len(walk) < self.walk_length:
            walk_options = d_graph[walk[-1]].get(self.NEIGHBORS_KEY, None)

            # Skip dead end nodes
            if not walk_options:
                break

            if len(walk) == 1:  # For the first step
                probabilities = d_graph[walk[-1]][self.FIRST_TRAVEL_KEY]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
            else:
                probabilities = d_graph[walk[-1]][self.PROBABILITIES_KEY][walk[-2]]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

            walk.append(walk_to)
        walk = list(map(str, walk))

        return walk

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

    def generate_walks(self):
        for n_walk in range(self.num_walks):
            nodes = list(self.d_attr.keys())
            random.shuffle(nodes)
            for source in nodes:
                choice = np.random.sample(size=1)[0]
                if choice > self.t:
                    current_walk = self.generate_single_network_walk(str(source))
                else:
                    current_walk = self.generate_single_attr_walk(source)
                self.walks.append(current_walk)

        return self.walks

    def fit(self, **skip_gram_params):

        print("Pre-computing probabilities...")
        self.precompute_attr_probabilities()
        self.precompute_network_probabilities()
        print("Generating walks...")
        self.generate_walks()

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params["size"] = self.dim

        if 'sg' not in skip_gram_params:
            skip_gram_params["sg"] = self.sg  # 1 - use skip-gram; otherwise, use CBOW

        model = gensim.models.Word2Vec(self.walks, **skip_gram_params)
        if self.labels is None:  # do not need to label output, just return
            return model.wv
        else:  # create dictionary with source labels before outputting
            output = dict()
            for node in model.wv.vocab:
                output[self.labels[int(node)].strip()] = model.wv[node]
            return output



