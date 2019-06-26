import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

path = "vectors_label.txt"
src_labels_path = "data/news/source_labels.txt"
gt_file = "data/news/GT_NEW.csv"

gt_dict = dict()
vecs = dict()

# read vectors
with open(path) as fin:
    fin.readline()
    for line in fin:
        row = line.strip().split(",")
        vecs[row[0].lower()] = row[1:]

with open(gt_file) as fin:
    for line in fin:
        row = line.strip().split(",")
        gt_dict[row[0].lower()] = int(row[1])

# vecs = np.array(vecs)
# X_train = vecs
# tsne = TSNE(random_state=41, n_components=2, n_iter=500).fit_transform(X_train)
#
# cmap = ["blue", "red"]
#
# for i in range(len(tsne)):
#     t = tsne[i]
#     plt.scatter(t[0], t[1], c=cmap[gt_dict[labels[i]]])

# cmap = ["blue", "red"]
# for source in vecs:
#     if source not in gt_dict:
#         continue
#     plt.scatter(vecs[source][0], vecs[source][1], c=cmap[gt_dict[source]])

plt.show()