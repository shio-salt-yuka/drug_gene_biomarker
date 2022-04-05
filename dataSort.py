import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import variation
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import re

#
#
# drug_dict = {};
# creating dataframe with calculated data......
# for x in range(len(df.columns)):
#     # Kmeans, output: 2 centroid points
#     drugName = df.columns[x]
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(df[[df.columns[x]]])
#     # print(sum(kmeans.labels_))
#     nDiff = abs((len(kmeans.labels_) - sum(kmeans.labels_)) - sum(kmeans.labels_))
#     # smaller the nDiff, closer the sizes are between the two groups
#     # print(nDiff)
#     clusters = kmeans.cluster_centers_
#     big = max(clusters)
#     big = big[0]
#     small = min(clusters)
#     small = small[0]
#     diff = big - small
#     mean = df[drugName].mean()
#     cv = variation(df[drugName])
#     drug_dict[df.columns[x]] = mean, cv, big, small, diff, nDiff
#
# data = pd.DataFrame.from_dict(drug_dict, orient='index')
# df = data.rename(columns={0: "Mean", 1: "CV", 2: "Big", 3: "Small", 4: "Difference", 5: "nDiff"})
# # equal num of centroids in each cluster
# df = df[df.nDiff == 0]
# CVmu = df["CV"].mean()
# CVstd = df["CV"].std()
# df = df[df.CV > (CVmu + CVstd)]

# eribulin!!!

# Mean = data["Mean"]
# CoeV = data["CV"]
# Big = data["Big"]
# Small = data["Small"]
# Difference = data["Difference"]
# nDiff = data["nDiff"]
# labels = ["4u","ab", "ad", "api","a2", "a3", "a1", "bi","bo", "cb", "cr",
#          "c8", "co", "cr", "db", "ds", "dm", "do", "et", "ez", "ep", "er", "ev", "fu", "ix", "la",
#          "m3","m6", "me", "mk", "na", "ne", "ni", "ol", "pa", "pe", "pi", "ro", "rm", "ru", "sa",
#           "se","so","ta","tl", "ts", "ti", "tr", "ve", "vi", "zw", "z4"]
# fig, ax = plt.subplots()
# ax.scatter(nDiff, Big)
# for i, drug in enumerate(labels):
#     ax.annotate(drug, (nDiff[i], Big[i]))

a_dataframe = pd.read_csv("Combined.ScreenA.gr50.scores.txt", sep="\t")
df = a_dataframe[(a_dataframe.drug == "eribulin")]
# print(df)
# HCI001,003,005,011,015,016,017,019,027

b_dataframe = pd.read_csv("RNAseq.batchnormalized.txt", sep="\t")
gene_dataframe = b_dataframe.filter(regex='HCI0[0-2][0-9]_pdx_.+[0-9]$')
gene_dataframe.drop(gene_dataframe.columns[len(gene_dataframe.columns) - 1], axis=1, inplace=True)  # removing HCI028
my_file = open("subject_list2.txt", "r")
content_list = my_file.readlines()
gene_dataframe = gene_dataframe[content_list]
gene_df = gene_dataframe.T  # flipping axis
# print(gene_df)


# HCI001,003,005,011,015,016,017,019,027

# print(gene_dataframe.values)
# print(gene_df)  # gene vs. subject
# print(gene_df.values)

x = gene_df.values

pca_gene = PCA()
principalComponents_gene = pca_gene.fit_transform(x)
principal_gene_Df = pd.DataFrame(data=principalComponents_gene)

# principal_gene_Df['Gene'] = gene_dataframe.axes[0].tolist()
# print(principal_gene_Df)

# cluster.pcaplot(x=principal_gene_Df[0], y=principal_gene_Df[1], labels=principal_gene_Df.columns.values,
#    var1=round(pca_gene.explained_variance_ratio_[0]*100, 2),
#    var2=round(pca_gene.explained_variance_ratio_[1]*100, 2))
principal_gene_Df = abs(principal_gene_Df)
indices = principal_gene_Df.idxmax(axis=0).tolist()

# print(gene_dataframe.index[indices])


# for x in range(len(gene_df.columns)):
#     # Kmeans, output: 2 centroid points
#     geneName = gene_df.columns[x]
#     #print(geneName)
#     kmeans = KMeans(n_clusters=2).fit(gene_df[[gene_df.columns[x]]])
#     # print(sum(kmeans.labels_))
#     clusters = kmeans.cluster_centers_

# print(principal_gene_Df.max())

# mav value over the entire dataframe
# print(df.max(numeric_only=True).max())

# eigenvalues = pca_gene.explained_variance_


# PC_values = np.arange(pca_gene.n_components_) + 1
# plt.plot(PC_values, pca_gene.explained_variance_ratio_, 'ro-', linewidth=2)
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Proportion of Variance Explained')
# plt.show()

# sns.scatterplot(data = principal_gene_Df, x = "PC1", y = "PC2")
# # for i, drug in enumerate(labels):
# #     ax.annotate(drug, (nDiff[i], Big[i]))
# plt.show()

# hi sq, pearson's'
