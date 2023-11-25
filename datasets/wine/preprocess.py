import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, scale

data = pd.read_csv('wine.tsv', sep='\t')


all_cols = data.columns
target = 'Class'


y = data[target]

data.drop(target, axis=1, inplace=True)


scaler = MinMaxScaler()
scaler.fit(data)
X = scaler.transform(data)
X = pd.DataFrame(X)
# X = scale(X)
data = X



X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0,stratify=y)

data_train = np.append(X_train, np.expand_dims(y_train, 1), axis=1)
data_train = pd.DataFrame(data_train)

data_test = np.append(X_test, np.expand_dims(y_test, 1), axis=1)
data_test = pd.DataFrame(data_test)

data_train.to_csv('train.csv', index=False, header=all_cols)
data_test.to_csv('test.csv', index=False, header=all_cols)


# data[target] = y
# data.to_csv('wine_preprocessado.csv', index=False, header=all_cols) 

print("len(data): ", len(data))
print("len(data_train): ", len(data_train))
print("len(data_test): ", len(data_test))
