import os
from time import time
import pandas as pd
import tensorflow as tf
from milp import codify_network, codify_network_relaxed
from teste import get_explanation_relaxed, get_minimal_explanation
from typing import List
from docplex.mp.constr import LinearConstraint


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


def explain_instance(
    dataset: {}, configuration: {}, instance_index: int
) -> List[LinearConstraint]:
    dir_path, n_classes, model = (
        dataset["dir_path"],
        dataset["n_classes"],
        dataset["model"],
    )

    method = configuration["method"]
    relaxe_constraints = configuration["relaxe_constraints"]

    data_test = pd.read_csv(f"{dir_path}/test.csv")
    data_train = pd.read_csv(f"{dir_path}/train.csv")

    data = data_train._append(data_test)

    model = tf.keras.models.load_model(f"{dir_path}/{model}")

    (
        mdl_milp_with_binary_variable,
        output_bounds_binary_variables,
        bounds,
    ) = codify_network(model, data, method, relaxe_constraints)

    # usar bounds precisos do primeiro modelo
    model_milp_relaxed, output_bounds_relaxed = codify_network_relaxed(
        model,
        data,
        method,
        relaxe_constraints,
        output_bounds_binary_variables,
        bounds=bounds,
    )

    network_input = data.iloc[instance_index, :-1]
    print(network_input)  # network_input = instance

    network_input = tf.reshape(tf.constant(network_input), (1, -1))

    network_output = model.predict(tf.constant(network_input))[0]

    network_output = tf.argmax(network_output)

    mdl_aux = model_milp_relaxed.clone()

    # explanation = get_explanation_relaxed(
    #     mdl_aux,
    #     network_input,
    #     network_output,
    #     n_classes=n_classes,
    #     method=method,
    #     output_bounds=output_bounds_binary_variables,
    #     delta = 1
    # )

    explanation = get_minimal_explanation(
        mdl_aux,
        network_input,
        network_output,
        n_classes=n_classes,
        method=method,
        output_bounds=output_bounds_binary_variables,
    )
    return explanation


def gerar_rede_com_dataset_iris(n_neurons=20, n_hidden_layers=1):
    dir_path = "datasets\\iris"
    num_classes = 3
    gerar_rede(dir_path, num_classes, n_neurons, n_hidden_layers)


def gerar_rede_com_dataset_digits(n_neurons=20, n_hidden_layers=1):
    dir_path = "datasets\\digits"
    num_classes = 10
    gerar_rede(dir_path, num_classes, n_neurons, n_hidden_layers)


def gerar_rede_com_dataset_wine(n_neurons=20, n_hidden_layers=1):
    dir_path = "datasets\\wine"
    num_classes = 10
    gerar_rede(dir_path, num_classes, n_neurons, n_hidden_layers)

def ola_mundo():
    print("ola mundo")

def explicar_rede():
    datasets = [
        {
            "dir_path": "datasets/digits",
            "model": "models/model_1layers_20neurons.h5",
            "n_classes": 10,
        },
        {
            "dir_path": "datasets/iris",
            "model": "models/model_1layers_20neurons.h5",
            "n_classes": 3,
        },
        {
            "dir_path": "datasets/iris",
            "model": "models/model_6layers_20neurons.h5",
            "n_classes": 3,
        },
    ]
    configurations = [{"method": "fischetti", "relaxe_constraints": True}]

    for i in range(0, 10):
        explanation = explain_instance(
            dataset=datasets[0], configuration=configurations[0], instance_index=i
        )

        for x in explanation:
            print(x)
        print("len: ", len(explanation), "\n")
        # if(len(explanation)<4):
        #     break


explicar_rede()
