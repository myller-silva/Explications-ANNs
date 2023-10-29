import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('datasets\\digits\\digits.csv', sep=',')

all_cols = data.columns
target = 'target'
y = data[target]
data.drop(target, axis=1, inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, columns=data.columns)


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0,stratify=y)


data_train = np.append(X_train, np.expand_dims(y_train, 1), axis=1)
data_train = pd.DataFrame(data_train)

data_test = np.append(X_test, np.expand_dims(y_test, 1), axis=1)
data_test = pd.DataFrame(data_test)

data_train.to_csv('datasets\\digits\\train.csv', index=False, header=all_cols)
data_test.to_csv('datasets\\digits\\test.csv', index=False, header=all_cols)
data.to_csv('datasets\\digits\\digits_preprocessado.csv')

print("len(data): ", len(data))
print("len(data_train): ", len(data_train))
print("len(data_test): ", len(data_test))