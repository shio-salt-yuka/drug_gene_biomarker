import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


a_dataframe = pd.read_csv("Combined.ScreenA.gr50.scores.txt", sep="\t")
b_dataframe = pd.read_csv("Combined.ScreenB.gr50.scores.txt", sep="\t")
drugScreenDF = [a_dataframe, b_dataframe]
merge = []
# combining drug screen data A and B
for x in drugScreenDF:
    x = x[["HCIid", "drug", "GR_AOC"]]
    x = x[x.HCIid != "Control"]
    x = x[x.HCIid != "HCI-011.E2"]
    x = x[x.HCIid != "HCI-017.E2"]
    x = x
    x = x.pivot(index='HCIid', columns='drug', values='GR_AOC')
    merge.append(x)
drugDF = pd.concat(merge)

# determining responder vs. non-responder via Kmeans
def makeDrugDataframe():
    x = np.array(drugDF['birinapant'])
    # 2D array needed
    km = KMeans(n_clusters=2, random_state=0).fit(x.reshape(-1, 1))  # reshape() makes 1D to 2D array
    centers = km.cluster_centers_
    labels = km.labels_
    y = drugDF['birinapant']
    drugStatusDF = pd.DataFrame(y)
    # label:big cluster:responder, small cluster:non-responder
    drugStatusDF['status'] = labels
    drugStatusDF['status'] = drugStatusDF['status'].replace([0, 1], ['N', 'R'])
    # HCIid index to a column
    drugStatusDF = drugStatusDF.rename_axis('HCIid').reset_index()
    return drugStatusDF
