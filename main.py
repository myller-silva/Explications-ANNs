import os
from time import time
import pandas as pd
import tensorflow as tf
from milp import codify_network
from teste import get_miminal_explanation
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
    dataset: {}, configuration: {}, instance_index: int, instance: []
) -> List[LinearConstraint]:
    dir_path, n_classes, model = (
        dataset["dir_path"],
        dataset["n_classes"],
        dataset["model"],
    )
    # todo: receber uma configuracao especifica por argumento da funcao
    method = configuration["method"]
    relaxe_constraints = configuration["relaxe_constraints"]

    data_test = pd.read_csv(f"{dir_path}\\test.csv")
    data_train = pd.read_csv(f"{dir_path}\\train.csv")

    data = data_train._append(data_test)

    model = tf.keras.models.load_model(f"{dir_path}\\{model}")

    mdl, output_bounds = codify_network(model, data, method, relaxe_constraints)

    #
    # data = data_test.to_numpy()

    # network_input = data[instance_index, :-1]
    network_input = instance

    # todo: perguntar como essa linha funciona exatamente
    network_input = tf.reshape(tf.constant(network_input), (1, -1))
    # print("reshape: ", network_input)

    network_output = model.predict(tf.constant(network_input))[0]
    # print("predict: ", network_output)

    network_output = tf.argmax(network_output)
    # print("argmax: ", network_output)

    mdl_aux = mdl.clone()

    # todo verificar melhor como funciona o get minimal explanation
    explanation = get_miminal_explanation(
        mdl_aux,
        network_input,
        network_output,
        n_classes=n_classes,
        method=method,
        output_bounds=output_bounds,  #
    )
    return explanation


def gerar_rede_com_dataset_iris(n_neurons = 20, n_hidden_layers = 1):
    dir_path = "datasets\\iris"
    num_classes = 3 
    gerar_rede(dir_path, num_classes, n_neurons, n_hidden_layers)


def explicar_rede():
    datasets = [
        {
            "dir_path": "datasets\\iris",
            "model": "models\\model_7layers_20neurons.h5",
            "n_classes": 3,
        }
    ]
    configurations = [{"method": "fischetti", "relaxe_constraints": False}]

    instance = [
        -0.9006811702978081,
        1.019004351971608,
        -1.340226526622762,
        -1.3154442950077396,
    ]

    print("explicar rede")
    print("dataset: ", datasets[0])

    explanation = explain_instance(
        dataset=datasets[0],
        configuration=configurations[0],
        instance_index=3,
        instance=instance,
    )

    for x in explanation:
        print(x)


# gerar_rede_com_dataset_iris(n_neurons=20, n_hidden_layers=6)

explicar_rede()


# explicar rede
# dataset:  {'dir_path': 'datasets\\iris', 'model': 'models\\model_4layers_20neurons.h5', 'n_classes': 3}
# 2023-10-16 08:59:39.194132: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 1/1 [==============================] - 0s 184ms/step
# input1: x_0 == -0.90068119764328
# input2: x_1 == 1.0190043449401855
# input3: x_2 == -1.3402265310287476

# explicar rede
# dataset:  {'dir_path': 'datasets\\iris', 'model': 'models\\model_5layers_20neurons.h5', 'n_classes': 3}
# 2023-10-16 09:02:08.020179: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 1/1 [==============================] - 0s 246ms/step
# input2: x_1 == 1.0190043449401855
# input3: x_2 == -1.3402265310287476
# input4: x_3 == -1.3154443502426147
