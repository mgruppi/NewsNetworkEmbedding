# Add source label to vectors
import os


path_labels = "data/news/source_labels.csv"

vecpath = "vec/"
outpath = "labeled_vec/"

labels = list()
with open(path_labels, "r") as fin:
    labels = fin.readlines()

print(labels)


for root, dirs, files in os.walk(vecpath):
    vecs = dict()

    for f in files:
        fpath = os.path.join(root, f)

        with open(fpath) as fin:
            fin.readline()  # read the header out
            for line in fin:
                line = line.strip().split(" ")
                vecs[labels[int(line[0])]] = line[1:]


        with open(os.path.join(outpath,f), "w") as fout:
            for key in vecs:
                fout.write(key.strip())
                for v in vecs[key]:
                    fout.write("," + str(v))

                fout.write("\n")
