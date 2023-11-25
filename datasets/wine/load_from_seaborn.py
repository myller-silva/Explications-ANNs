import pandas as pd

# URL do conjunto de dados Wine da UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# Nomes das colunas
column_names = [
    "Class",
    "Alcohol",
    "Malic_acid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280/OD315_of_diluted_wines",
    "Proline",
]

# Carregar o conjunto de dados usando pandas
wine = pd.read_csv(url, names=column_names)

# Especificar o caminho para salvar o conjunto de dados

# Salvar o conjunto de dados em formato TSV
wine.to_csv('wine.tsv', sep="\t", index=False)
