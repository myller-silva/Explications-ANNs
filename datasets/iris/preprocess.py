import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, scale

data = pd.read_csv('datasets\\iris\\iris.tsv', sep='\t')

# normalizar target como numérico
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species']) 

all_cols = data.columns
target = 'species'
# print(data)

y = data[target]
# print(y)
data.drop(target, axis=1, inplace=True)

# Normalize numeric features
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scaler = MinMaxScaler()
scaler.fit(data[numeric_cols])
X = scaler.transform(data[numeric_cols])
X = pd.DataFrame(X)
# X = scale(X)
data[numeric_cols] = X

# print(data)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0,stratify=y)

data_train = np.append(X_train, np.expand_dims(y_train, 1), axis=1)
data_train = pd.DataFrame(data_train)

data_test = np.append(X_test, np.expand_dims(y_test, 1), axis=1)
data_test = pd.DataFrame(data_test)

data_train.to_csv('datasets\\iris\\train.csv', index=False, header=all_cols)
data_test.to_csv('datasets\\iris\\test.csv', index=False, header=all_cols)
data.to_csv('datasets\\iris\\iris_preprocessado.csv')

print("len(data): ", len(data))
print("len(data_train): ", len(data_train))
print("len(data_test): ", len(data_test))