# Map source id to vector
import numpy as np
from collections import OrderedDict

not_in_network = []
not_in_network_labels = []
labels_id = dict()
i = 0
with open("data/news/source_labels.csv") as fin:
    for line in fin:
        line = line.strip().lower()
        labels_id[line] = i
        i += 1

id_vectors = np.zeros((len(labels_id), 200))
with open("data/news/keys_sourcevec_200.txt") as fin:
    for line in fin:
        line = line.strip().split(",")
        if line[0].lower() not in labels_id:
            not_in_network.append(np.array(line[1:]))
            not_in_network_labels.append(line[0].lower())
        else:
            id = labels_id[line[0].lower()]
            vector = np.array(line[1:])
            id_vectors[id] = vector

with open("data/news/source_attributes.txt", "w") as fout:
    for i in range(len(id_vectors)):
        fout.write(str(i))
        for x in id_vectors[i]:
            fout.write(","+str(x))
        fout.write("\n")

    for j in range(len(not_in_network)):
        fout.write(str(len(id_vectors)+j))
        for x in not_in_network[j]:
            fout.write(","+str(x))
        fout.write("\n")


with open("data/news/source_labels.txt", "w") as fout:
    for w in labels_id:
        fout.write(w)
        fout.write("\n")

    for l in not_in_network_labels:
        fout.write(l)
        fout.write("\n")