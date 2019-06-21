import networkx as nx
import nt2vec
import numpy as np


def read_edgelist_file(path):
    g = nx.read_weighted_edgelist(path)

    return g


def read_attr_file(path):
    attr = []
    with open(path) as fin:
        for line in fin:
            line = line.split(",")
            attr.append(np.array(line[1:], dtype=float))

    return attr


def get_vectors(g, attr, labels=None, dim=40, t=1, p=1, q=1, knn=20):
    nt = nt2vec.NT2VEC(g, attr, labels=labels, sg=1, dim=dim, p=p, q=q, t=t, knn=knn)

    model = nt.fit(window=10, min_count=1, batch_words=4)

    return model


def main():
    # Read data in
    input_path = "data/news/edge_list.csv"
    input_attributes = "data/news/source_attributes.csv"
    label_path =  "data/news/source_labels.csv"

    g = read_edgelist_file(input_path)
    attr = read_attr_file(input_attributes)
    with open(label_path, "r") as fin:
        labels = fin.readlines()

    print(get_vectors(g, attr, labels)["Vox"])

    # N = 10  # 10 value per parameter in grid search
    # t_list = np.linspace(0.1, 1, N, endpoint=True)
    # p_list = np.linspace(0.1, 1, N, endpoint=True)
    # q_list = np.linspace(0.1, 1, N, endpoint=True)
    #
    # dim = 100
    #
    # for t in t_list:
    #     for p in p_list:
    #         for q in q_list:
    #             output_file = "vec/nt2v_sourcevecs_%s_t%.2f_p%.2f_q%.2f.txt" % (dim, t, p, q)
    #             nt = nt2vec.NT2VEC(g, attr, dim=100, sg=1, p=p, q=q, t=t, knn=100)
    #
    #             nt.precompute_attr_probabilities()
    #             nt.precompute_network_probabilities()
    #
    #             nt.generate_walks()
    #             model = nt.fit(window=10, min_count=1, batch_words=4)
    #
    #             r = model.wv.most_similar('0')
    #             for v in r:
    #                 print(v)
    #
    #             model.wv.save_word2vec_format(output_file)


if __name__ == "__main__":
    main()