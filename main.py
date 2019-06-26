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


def get_vectors(g, attr, labels, dim=20, t=1, p=1, q=1, knn=20):
    nt = nt2vec.NT2VEC(g, attr, labels=labels, sg=1, dim=dim, p=p, q=q, t=t, knn=knn)

    model = nt.fit(window=10, min_count=1, batch_words=4)

    return model


def main():
    # Read data in
    input_path = "data/news/edge_list.csv"
    input_attributes = "data/news/source_attributes.txt"
    label_path = "data/news/source_labels.txt"
    out_path = "vectors_label.txt"

    g = read_edgelist_file(input_path)
    attr = read_attr_file(input_attributes)
    with open(label_path, "r") as fin:
        labels = fin.readlines()

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