# Reads results.csv with grid-search results for analysos
import pandas as pd

path = "results.csv"

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv(path, header=None)

df = df.sort_values(by=[3, 4], ascending=False)

print(df[:10])