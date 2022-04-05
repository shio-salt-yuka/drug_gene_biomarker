from idlelib import editor

from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import geneExpression as geneFile
import drugStatus as drugFile

df = pd.read_csv('birinapant_fileNames.txt', sep=",", header=None)
df = df.rename(columns=df.iloc[0])  # df[0:] to header
df = df[1:]  # Take the data less the header row
nameFile = df
p_val = {}

# determining responder vs. non-responder via Kmeans
drugDF = drugFile.makeDrugDataframe()
# turn HCIid to row names
# drugDF = drugDF.set_index('HCIid')
print("drugDF made")

# determining high and low gene expression via Kmeans

df = pd.read_csv('geneDataframe.csv')
print(df.head(10))

# adjusting subject name in both dataframes
# for x in dataframes:
#     for i in range(len(drugDF['HCIid'])):
#         if x['HCIid'][i] in nameFile.values:
#             if x == drugDF:
#                 index = nameFile[nameFile['Drug'] == x['HCIid'][i]].index[0]
#             elif x == geneDF:
#                 index = nameFile[nameFile['Gene'] == x['HCIid'][i]].index[0]
#             y = nameFile['adjusted'][index]
#             x['HCIid'] = x['HCIid'].replace(x['HCIid'][i], y)
#     print(x)
#
# # combining drug and gene data
# df = drugDF.set_index('HCIid').combine_first(geneDF.set_index('HCIid'))
# df = df[['expression', 'responder']]
# df = df.dropna()
#
#
# # chi square table
# table = pd.crosstab(df['responder'], df['expression'])
# c, p, dof, expected = chi2_contingency(table)

# add gene name and p value to dictionary
# p_val[gene] = p
# pd.DataFrame.from_dict(data)

# top50, gene by pvalue
# compare to the pilot data
