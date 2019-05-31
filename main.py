import networkx as nx
import nt2vec
import numpy as np

def main():
    # Read data in
    input_path = "data/news/edge_list.csv"
    input_attributes = "data/news/test.txt"

    g = nx.read_weighted_edgelist(input_path)
    attr = []
    with open(input_attributes) as fin:
        for line in fin:
            line = line.split(",")
            attr.append(np.array(line[1:], dtype=float))

    nt = nt2vec.NT2VEC(g, attr)

    nt.precompute_attr_probabilities()

    for key in nt.d_graph:
        print(key, nt.d_graph[key][nt.PROB_KEY][nt.ATTR_PROB_KEY])

    w = nt.generate_attr_walk()
    for walk in w:
        print(walk)


if __name__ == "__main__":
    main()