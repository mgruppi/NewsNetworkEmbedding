import networkx as nx
import nt2vec
import numpy as np


def main():
    # Read data in
    input_path = "data/news/edge_list.csv"
    input_attributes = "data/news/source_attributes.txt"

    g = nx.read_weighted_edgelist(input_path)
    attr = []
    with open(input_attributes) as fin:
        for line in fin:
            line = line.split(",")
            attr.append(np.array(line[1:], dtype=float))

    nt = nt2vec.NT2VEC(g, attr, dim=100, sg=1, p=1, q=1, t=0.75, knn=100)

    nt.precompute_attr_probabilities()
    nt.precompute_network_probabilities()

    nt.generate_walks()
    model = nt.fit(window=10, min_count=1, batch_words=4)

    r = model.wv.most_similar('0')
    for v in r:
        print(v)

    model.wv.save_word2vec_format("nt2v_sourcevecs_100_t075.txt")


if __name__ == "__main__":
    main()