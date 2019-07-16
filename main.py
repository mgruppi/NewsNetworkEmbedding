import networkx as nx
import nt2vec
import numpy as np


def read_edgelist_file(path, label_to_id):
    g = nx.DiGraph()

    with open(path, "r") as fin:
        for line in fin:
            line = line.split(",")
            s = line[0].lower()
            t = line[1].lower()
            g.add_edge(label_to_id[s], label_to_id[t], weight=float(line[2]))
    # g = nx.read_weighted_edgelist(path, delimiter=",")
    return g


def read_attr_file(path):
    attr = []
    with open(path) as fin:
        for line in fin:
            line = line.split(",")
            attr.append(np.array(line[1:], dtype=float))

    return attr


def get_vectors(g, attr, labels, dim=40, t=0.4, p=0.5, q=1, knn=10):
    nt = nt2vec.NT2VEC(g, attr, labels=labels, sg=1, dim=dim, p=p, q=q, t=t, knn=knn)

    model = nt.fit(window=10, min_count=1, batch_words=4)

    return model


def get_labels(input_path, input_attributes):
    label_to_id = dict()
    id_to_label = list()
    new_id = 0

    # NEED TO READ INPUT ATTRIBUTES FIRST SINCE DATA IS A NP MATRIX
    with open(input_attributes) as fin:
        for line in fin:
            line = line.split(",")
            source = line[0].lower()

            if source not in label_to_id:
                label_to_id[source] = new_id
                id_to_label.append(source)
                new_id += 1
            else:
                print(source)

    with open(input_path, "r") as fin:
        for line in fin:
            line = line.split(",")
            source = line[0].lower()

            if source not in label_to_id:
                label_to_id[source] = new_id
                id_to_label.append(source)
                new_id += 1

    return label_to_id, id_to_label


def main():
    # Read data in
    input_path = "data/news/edge_list_1month.csv"
    input_attributes = "data/news/source_attributes_1month.txt"
    label_path = "data/news/source_labels.txt"
    out_path = "nt2v_sourcevecs_1month.csv"

    label_to_id, id_to_label = get_labels(input_path, input_attributes)

    print(len(id_to_label), len(label_to_id))

    g = read_edgelist_file(input_path, label_to_id)
    attr = read_attr_file(input_attributes)

    labels = id_to_label
    # with open(label_path, "r") as fin:
    #     labels = fin.readlines()

    print(len(attr))

    out = get_vectors(g, attr, labels)
    print(out.keys())
    print(len(out.keys()))

    with open(out_path, "w") as fout:
        for k in out:
            fout.write(k)
            for v in out[k]:
                fout.write(",%f" % v)
            fout.write("\n")


if __name__ == "__main__":
    main()