import seaborn as sns

# carregar o conjunto de dados Iris usando seaborn
iris = sns.load_dataset("iris")

path = 'C:\\Users\\mylle\\OneDrive\\Documentos\\GitHub\\TestesTypescript\\Explications-ANNs\\datasets\\iris'

# salvar o conjunto de dados em formato TSV
iris.to_csv(path+'\\'+ 'iris.tsv', sep="\t", index=False)