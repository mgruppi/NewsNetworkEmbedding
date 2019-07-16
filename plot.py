import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

path = "nt2v_sourcevecs_1month.csv"
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

X_train = np.array(list(vecs.values()), dtype=float)
keys = list(vecs.keys())
tsne = TSNE(random_state=41, n_components=2, n_iter=500).fit_transform(X_train)

cmap = ["blue", "red"]

for i in range(len(tsne)):
    t = tsne[i]
    if keys[i] in gt_dict:
        plt.scatter(t[0], t[1], c=cmap[int(gt_dict[keys[i]])])
    else:
        plt.scatter(t[0], t[1], c="gray")

print(len(vecs))

# cmap = ["blue", "red"]
# for source in vecs:
#     if source not in gt_dict:
#         continue
#     plt.scatter(vecs[source][0], vecs[source][1], c=cmap[gt_dict[source]])

plt.show()