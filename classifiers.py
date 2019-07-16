from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def read_gt(path):
    d = dict()
    with open(path) as fin:
        for line in fin:
            row = line.strip().split(",")
            d[row[0].lower()] = int(row[1])

    return d


data_path = "nt2v_sourcevecs_1month.csv"
gt_path = "data/news/GT_NEW.csv"

gt = read_gt(gt_path)
df = pd.read_csv(data_path, header=None)

sources = df[0]
unique_sources = list(sources.unique())
np.random.shuffle(unique_sources)
outsource = unique_sources[:int(0.2*len(unique_sources))]
insource = unique_sources[int(0.2*len(unique_sources)):]

y_train = np.zeros(len(insource))
y_test = np.zeros(len(outsource))


df_train = df[df[0].isin(insource)]
df_test = df[df[0].isin(outsource)]


for i in range(len(insource)):
    if insource[i] not in gt:
        continue
    y_train[i] = gt[insource[i]]

for i in range(len(outsource)):
    if outsource[i] not in gt:
        continue
    y_test[i] = gt[outsource[i]]

clf = SVC(kernel="rbf", gamma="auto")

# x = df[df.columns[1:]]

x_train = df_train[df_train.columns[1:]]
x_test = df_test[df_test.columns[1:]]

clf = RandomForestClassifier(n_estimators=300, max_features="log2", class_weight="balanced")

#  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf.fit(x_train, y_train)

print("Train", clf.score(x_train, y_train))
print("Test", clf.score(x_test, y_test))



