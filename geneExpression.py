import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing as mp
from multiprocessing import Pool


def km(dataFrame):
    result = pd.DataFrame()
    for col in dataFrame.columns:
        x2 = np.array(dataFrame[col])
        gene = col
        kMean = KMeans(n_clusters=2, random_state=0).fit(x2.reshape(-1, 1))
        labels2 = kMean.labels_
        df = pd.DataFrame(labels2, columns=[gene])
        result = pd.concat([result, df], axis=1)
    print('XXXXXXXXXXXXXX')
    return result


c_dataframe = pd.read_csv("RNAseq.batchnormalized.txt", sep="\t")
c_dataframe = c_dataframe.set_index('Gene')
c_dataframe = c_dataframe.filter(regex='HCI0[0-6][0-9]_pdx_.+[0-9]$')
c_dataframe = c_dataframe.T
c_dataframe.index.name = 'HCIid'
c_dataframe.reset_index(inplace=True)
x = c_dataframe['HCIid'].tolist()
data = {'HCIid': x}
df1 = pd.DataFrame(data)
del c_dataframe['HCIid']
geneDF = c_dataframe.loc[:, (c_dataframe != 0).any(axis=0)]  # removing the zero-columns
geneDF = geneDF.T
df_split = np.array_split(geneDF, 100)
split_data = []
for i in df_split:
    split_data.append(i.T)

cores = mp.cpu_count()


# def makeGeneDataframe():
#     finalDF = pd.DataFrame()
#     for x in subDF:
#         finalDF = pd.concat([finalDF, x], axis=1)
#
#     return finalDF


if __name__ == '__main__':
    with Pool(cores) as pool:
        smallDF = split_data
        # pool.map returns results as a list
        subDF = pool.map(km, smallDF)

        finalDF = pd.DataFrame()
        for x in subDF:
            finalDF = pd.concat([finalDF, x], axis=1)
            print("CONCAT-ing")

        print("WHAT")

    finalDF.to_csv('geneDataframe.csv', sep='\t')
