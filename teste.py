from typing import List
import numpy as np
import tensorflow as tf
from milp import codify_network
from time import time
from statistics import mean, stdev
import pandas as pd
from docplex.mp.constr import LinearConstraint

# todo: ver se faz uma chamada para cada classe não predita
def insert_output_constraints_fischetti(
    mdl, output_variables, network_output, binary_variables
):
    variable_output = output_variables[network_output]
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            p = binary_variables[aux_var]
            aux_var += 1
            mdl.add_indicator(p, variable_output <= output, 1)

    return mdl


def insert_output_constraints_tjeng(
    mdl, output_variables, network_output, binary_variables, output_bounds
):
    variable_output = output_variables[network_output]
    upper_bounds_diffs = (
        output_bounds[network_output][1] - np.array(output_bounds)[:, 0]
    )  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            z = binary_variables[aux_var]
            mdl.add_constraint(variable_output - output - ub * (1 - z) <= 0)
            aux_var += 1

    return mdl


def get_minimal_explanation(
    mdl,
    network_input,
    network_output,
    n_classes,
    method,
    output_bounds=None,
    initial_explanation=None,
) -> List[LinearConstraint]:
    assert not (
        method == "tjeng" and output_bounds == None
    ), "If the method tjeng is chosen, output_bounds must be passed."

    output_variables = [mdl.get_var_by_name(f"o_{i}") for i in range(n_classes)]

    if initial_explanation is None:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == feature.numpy()
                for i, feature in enumerate(network_input[0])
            ],
            names="input",
        )
    else:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == network_input[0][i].numpy()
                for i in initial_explanation
            ],
            names="input",
        )

    binary_variables = mdl.binary_var_list(n_classes - 1, name="b") # todo: como isso é utilizado dentro do insert_output_constraints_fischetti?
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    if method == "tjeng":
        mdl = insert_output_constraints_tjeng(
            mdl, output_variables, network_output, binary_variables, output_bounds
        )
    else:
        mdl = insert_output_constraints_fischetti(
            mdl, output_variables, network_output, binary_variables
        )

    for constraint in input_constraints:
        mdl.remove_constraint(constraint)

        mdl.solve(log_output=False)
        if mdl.solution is not None: 
            mdl.add_constraint(constraint)

    return mdl.find_matching_linear_constraints("input")


def get_explanation_relaxed(
    mdl,
    network_input,
    network_output,
    n_classes,
    method,
    output_bounds=None,
    initial_explanation=None,
    delta=0.1,
) -> List[LinearConstraint]:
    # todo: output_bounds só é relevante se o metodo for tjeng
    assert not (
        method == "tjeng" and output_bounds == None
    ), "If the method tjeng is chosen, output_bounds must be passed."

    output_variables = [mdl.get_var_by_name(f"o_{i}") for i in range(n_classes)]

    if initial_explanation is None:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == feature.numpy()
                for i, feature in enumerate(network_input[0])
            ],
            names="input",
        )
    else:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == network_input[0][i].numpy()
                for i in initial_explanation
            ],
            names="input",
        )

    binary_variables = mdl.binary_var_list(n_classes - 1, name="b")
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    if method == "tjeng":
        mdl = insert_output_constraints_tjeng(
            mdl, output_variables, network_output, binary_variables, output_bounds
        )

    # todo: !(o1>o2 and o1>o3)
    # todo: modificar para o1<=o2 or o1<=o3
    else:
        mdl = insert_output_constraints_fischetti(
            mdl, output_variables, network_output, binary_variables
        )

    for constraint in input_constraints:
        mdl.remove_constraint(constraint)

        x = constraint.get_left_expr()
        v = constraint.get_right_expr()

        constraint_left = mdl.add_constraint(v - delta <= x)
        constraint_right = mdl.add_constraint(x <= v + delta)

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            mdl.remove_constraint(constraint_left)
            mdl.remove_constraint(constraint_right)

    return mdl.find_matching_linear_constraints("input")


def main():
    datasets = [  # {'dir_path': 'australian', 'n_classes': 2},
        # {'dir_path': 'auto', 'n_classes': 5},
        # {'dir_path': 'backache', 'n_classes': 2},
        # {'dir_path': 'breast-cancer', 'n_classes': 2},
        # {'dir_path': 'cleve', 'n_cla
        # sses': 2},
        # {'dir_path': 'cleveland', 'n_classes': 5},
        # {'dir_path': 'glass', 'n_classes': 5},
        {"dir_path": "glass2", "n_classes": 2},
        # {'dir_path': 'heart-statlog', 'n_classes': 2}, {'dir_path': 'hepatitis', 'n_classes': 2},
        # {'dir_path': 'spect', 'n_classes': 2},
        # {'dir_path': 'voting', 'n_classes': 2}
    ]

    configurations = [  # {'method': 'fischetti', 'relaxe_constraints': True},
        {"method": "fischetti", "relaxe_constraints": True},
        # {'method': 'tjeng', 'relaxe_constraints': True},
        {"method": "tjeng", "relaxe_constraints": False},
    ]

    df = {
        "fischetti": {
            True: {"size": [], "milp_time": [], "build_time": []},
            False: {"size": [], "milp_time": [], "build_time": []},
        },
        "tjeng": {
            True: {"size": [], "milp_time": [], "build_time": []},
            False: {"size": [], "milp_time": [], "build_time": []},
        },
    }

    for dataset in datasets:
        dir_path = dataset["dir_path"]
        n_classes = dataset["n_classes"]

        for config in configurations:
            print(dataset, config)

            method = config["method"]
            relaxe_constraints = config["relaxe_constraints"]

            data_test = pd.read_csv(f"datasets\\{dir_path}\\test.csv")
            data_train = pd.read_csv(f"datasets\\{dir_path}\\train.csv")
            data = data_train._append(data_test)

            model_path = f"datasets\\{dir_path}\\model_4layers_{dir_path}.h5"
            model = tf.keras.models.load_model(model_path)

            codify_network_time = []
            for _ in range(10):
                start = time()
                mdl, output_bounds = codify_network(
                    model, data, method, relaxe_constraints
                )
                codify_network_time.append(time() - start)
                print(codify_network_time[-1])

            time_list = []
            len_list = []
            # data = data.to_numpy()
            data = data_test.to_numpy()
            for i in range(data.shape[0]):
                # if i % 50 == 0:
                print(i)
                network_input = data[i, :-1]

                network_input = tf.reshape(tf.constant(network_input), (1, -1))
                network_output = model.predict(tf.constant(network_input))[0]
                network_output = tf.argmax(network_output)

                mdl_aux = mdl.clone()
                start = time()

                explanation = get_minimal_explanation(
                    mdl_aux,
                    network_input,
                    network_output,
                    n_classes=n_classes,
                    method=method,
                    output_bounds=output_bounds,
                )

                time_list.append(time() - start)

                len_list.append(len(explanation))

            df[method][relaxe_constraints]["size"].extend(
                [min(len_list), f"{mean(len_list)} +- {stdev(len_list)}", max(len_list)]
            )
            df[method][relaxe_constraints]["milp_time"].extend(
                [
                    min(time_list),
                    f"{mean(time_list)} +- {stdev(time_list)}",
                    max(time_list),
                ]
            )
            df[method][relaxe_constraints]["build_time"].extend(
                [
                    min(codify_network_time),
                    f"{mean(codify_network_time)} +- {stdev(codify_network_time)}",
                    max(codify_network_time),
                ]
            )

            print(
                f"Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)} +- {stdev(len_list)}\nM: {max(len_list)}"
            )
            print(
                f"Time:\nm: {min(time_list)}\na: {mean(time_list)} +- {stdev(time_list)}\nM: {max(time_list)}"
            )
            print(
                f"Build Time:\nm: {min(codify_network_time)}\na: {mean(codify_network_time)} +- {stdev(codify_network_time)}\nM: {max(codify_network_time)}"
            )
    "a" + 1
    df = {
        "fischetti_relaxe_size": df["fischetti"][True]["size"],
        "fischetti_relaxe_time": df["fischetti"][True]["milp_time"],
        "fischetti_relaxe_build_time": df["fischetti"][True]["build_time"],
        "fischetti_not_relaxe_size": df["fischetti"][False]["size"],
        "fischetti_not_relaxe_time": df["fischetti"][False]["milp_time"],
        "fischetti_not_relaxe_build_time": df["fischetti"][False]["build_time"],
        "tjeng_relaxe_size": df["tjeng"][True]["size"],
        "tjeng_relaxe_time": df["tjeng"][True]["milp_time"],
        "tjeng_relaxe_build_time": df["tjeng"][True]["build_time"],
        "tjeng_not_relaxe_size": df["tjeng"][False]["size"],
        "tjeng_not_relaxe_time": df["tjeng"][False]["milp_time"],
        "tjeng_not_relaxe_build_time": df["tjeng"][False]["build_time"],
    }

    index_label = []
    for dataset in datasets:
        index_label.extend(
            [
                f"{dataset['dir_path']}_m",
                f"{dataset['dir_path']}_a",
                f"{dataset['dir_path']}_M",
            ]
        )
    df = pd.DataFrame(data=df, index=index_label)
    df.to_csv("results.csv")
