# Given edge list and source->int id mapping, create edge list with strings rather than numbers


labels = []
with open("data/news/source_labels.csv") as fin:
    for line in fin:
        labels.append(line.strip().replace(" ",""))


outfile = "data/news/edge_list_labeled.csv"

with open(outfile, "w") as fout:
    with open("data/news/edge_list.csv") as fin:
        for line in fin:
            line = line.strip().split(" ")
            src = labels[int(line[0])]
            tar = labels[int(line[1])]
            w = float(line[2])

            fout.write("%s %s %s\n" % (src, tar, w))