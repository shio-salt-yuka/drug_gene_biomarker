import pandas as pd

# drug database
a_dataframe = pd.read_csv("Combined.ScreenA.gr50.scores.txt", sep="\t")
df = a_dataframe[(a_dataframe.drug == "birinapant")]
HCIs = ['HCI001', 'HCI003', 'HCI005', 'HCI011', 'HCI015', 'HCI016', 'HCI017', 'HCI019', 'HCI027']
rslt_df = df[df['HCIid'].isin(HCIs)]
df = df["HCIid"]
# print(df)

a_dataframe = pd.read_csv("Combined.ScreenB.gr50.scores.txt", sep="\t")
df2 = a_dataframe[(a_dataframe.drug == "birinapant")]
df2 = df2["HCIid"]
# print(df2)

# subject database
b_dataframe = pd.read_csv("RNAseq.batchnormalized.txt", sep="\t")
list_col = b_dataframe.columns
# print(list_col)
gene_dataframe = b_dataframe.filter(regex='HCI0[0-2][0-9]_pdx_.+[0-9]$')
gene_df = gene_dataframe.T  # flipping axis
print(gene_dataframe)

#gene_list, top