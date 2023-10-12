import pandas as pd

# gerar rede com dataset da iris


dir_path = 'datasets\\iris'
num_classes = 3
n_neurons = 20
n_hidden_layers = 1

data_train = pd.read_csv(dir_path+'\\'+'train.csv').to_numpy()
data_test = pd.read_csv(dir_path+'\\'+'test.csv').to_numpy()

# print(dataset_train)
# print(dataset_test)

x_train, y_train = data_train[:, :-1], data_train[:, -1]
x_test, y_test = data_test[:, :-1], data_test[:, -1]

