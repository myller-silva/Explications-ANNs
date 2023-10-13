import os
from time import time
import pandas as pd
import tensorflow as tf

from milp import codify_network
from teste import get_miminal_explanation


def gerar_rede(dir_path: str, num_classes: int, n_neurons: int, n_hidden_layers: int):
    data_train = pd.read_csv(dir_path + "\\" + "train.csv").to_numpy()
    data_test = pd.read_csv(dir_path + "\\" + "test.csv").to_numpy()

    x_train, y_train = data_train[:, :-1], data_train[:, -1]
    x_test, y_test = data_test[:, :-1], data_test[:, -1]

    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[x_train.shape[1]]),
        ]
    )

    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = os.path.join(
        dir_path, "models", f"model_{n_hidden_layers}layers_{n_neurons}neurons.h5"
    )

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    ck = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True
    )

    start = time()
    # treinamento
    model.fit(
        x_train,
        y_train_ohe,
        batch_size=4,
        epochs=100,
        validation_data=(x_test, y_test_ohe),
        verbose=2,
        callbacks=[ck, es],
    )
    print(f"Tempo de Treinamento: {time()-start}")

    # salvar modelo
    model = tf.keras.models.load_model(model_path)

    # avaliar modelo com os dados de treinamento
    print("Resultado Treinamento")
    model.evaluate(x_train, y_train_ohe, verbose=2)

    # avaliar modelo com os dados de teste
    print("Resultado Teste")
    model.evaluate(x_test, y_test_ohe, verbose=2)


# explicar instancia
def explicar_instancia( dataset: {}, configurations:[{}], index_instance:int):
    dir_path, n_classes, model = (
        dataset["dir_path"],
        dataset["n_classes"],
        dataset["model"],
    )
    for config in configurations:
        method = config["method"]
        relaxe_constraints = config["relaxe_constraints"]

        data_test = pd.read_csv(f"{dir_path}\\test.csv")
        data_train = pd.read_csv(f"{dir_path}\\train.csv")
        data = data_train._append(data_test)

        model = tf.keras.models.load_model(f"{dir_path}\\{model}")

        mdl, output_bounds = codify_network(model, data, method, relaxe_constraints)

        data = data_test.to_numpy()

        # for i in range(data.shape[0]):
          
        network_input = data[index_instance, :-1]
        print("network_input: ",network_input)

        network_input = tf.reshape(tf.constant(network_input), (1, -1))
        print("reshape: ", network_input)

        network_output = model.predict(tf.constant(network_input))[0]
        print("predict: ", network_output)

        network_output = tf.argmax(network_output)
        print("argmax: ", network_output)

        mdl_aux = mdl.clone()

        # todo verificar melhor como funciona o get minimal explanation
        explanation = get_miminal_explanation(
            mdl_aux,
            network_input,
            network_output,
            n_classes=n_classes,
            method=method,
            output_bounds=output_bounds,
        ) 
        for res in explanation:
            print(res)


# gerar rede com dataset da iris
dir_path = "datasets\\iris"
num_classes = 3
n_neurons = 20
# todo: por que funciona com n_hidden_layers == 0 ?
n_hidden_layers = 0
# gerar_rede(dir_path, num_classes, n_neurons, n_hidden_layers)


# explicar rede
print("explicar rede")
datasets = [
    {
        "dir_path": "datasets\\iris",
        "model": "models\\model_1layers_20neurons.h5",
        "n_classes": 3,
    }
]
configurations = [{"method": "fischetti", "relaxe_constraints": False}]

explicar_instancia(dataset=datasets[0], configurations=configurations, index_instance=3)