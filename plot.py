import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

path = "vectors.txt"
vecs = []
with open(path) as fin:
    fin.readline()
    for line in fin:
        vecs.append(line.strip().split()[1:])

vecs = np.array(vecs)
X_train = vecs
tsne = TSNE(random_state=41, n_components=2, n_iter=500).fit_transform(X_train)

for t in tsne:
    plt.scatter(t[0], t[1], c='blue')

plt.show()