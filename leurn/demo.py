import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from leurn.data import load_data
from leurn.utils import plot_explaination, read_partition_process_data, train_model


def helloworld():
    print("Hello world! Welcome to LEURN!")


def main():
    tf.debugging.set_log_device_placement(False)
    tf.autograph.set_verbosity(0)

    args = ArgumentParser()
    args.add_argument("-path", help="output path", type=str, default="./leurn_housing_demo")
    args.add_argument("-bs", help="batch size", type=int, default=512)
    args.add_argument("-lr", help="initial learning rate", type=float, default=5e-3)
    args.add_argument("-e", help="number of epoch", type=int, default=100)
    args.add_argument("-c", help="number of training cycle, in each cycle, lr is reduced", type=int, default=2)
    args.add_argument("-l", help="depth of the network", type=int, default=10)
    args.add_argument("-q", help="number of quantization regions", type=int, default=5)
    args.add_argument("-d", help="dropout rate", type=float, default=0.1)
    args = args.parse_args()

    output_path = os.path.realpath(args.path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    cycle_no = args.c  # how many cycles to train (in each cycle, learning rate is reduced)
    batch_size = int(args.bs)  # batch size
    learning_rate = float(args.lr)  # initial learning rate
    epoch_no = int(args.e)  # how many epochs to train per cycle
    n_layers = int(args.l)  # model depth
    quantization_regions = int(args.q)  # tanh quantization number
    dropout_rate = float(
        args.d
    )  # dropout rate : non-zero dropout is heavily recommended for more meaningful explanations

    print("Running the demo with the following hyperparameters:")
    print(" - output path: ", output_path)
    print(" - batch size: ", batch_size)
    print(" - initial learning rate: ", learning_rate)
    print(" - number of epochs: ", epoch_no)
    print(" - model depth: ", n_layers)
    print(" - tanh quantization number: ", quantization_regions)
    print(" - dropout rate: ", dropout_rate)

    # set memory growth for GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        warnings.warn("No GPU found. Running on CPU")
    else:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print("Running on GPU:", device)

    # HYPERPARAMETERS
    lr_reduction = 5  # reduction rate of learning rate per cycle
    # 0/bin: binary classification, 1/cls: multi-class classification, -1/reg: regression
    tasktype = "reg"

    # READ, PARTITION AND PROCESS THE DATA
    X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean, X_std = read_partition_process_data(
        load_data("housing"), target_name="median_house_value", task_type=tasktype
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_layers=n_layers,
        quantization_regions=quantization_regions,
        dropout_rate=dropout_rate,
        task_type=tasktype,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_train_cycles=cycle_no,
        epoch_no=epoch_no,
        lr_reduction=lr_reduction,
        output_path=output_path,
        verbose=2,
    )

    # #EVALUATE A SAMPLE FROM TEST DATA
    sample_indice = 1
    test_sample = X_test[sample_indice : sample_indice + 1]

    # #RETURN EXPLANATIONS
    Explanation = model.explain(test_sample, feat_names=X_names, y_max=y_max)
    print(Explanation)

    Explanation.to_csv(os.path.join(output_path, "explanation.csv"), index=True)

    # PLOT EXPLANATIONS
    plot_explaination(Explanation, os.path.join(output_path, "explanation.png"))

    print("Saved results to {output_path}".format(output_path=output_path))
    print("Run tensorbard with: tensorboard --logdir={output_path}".format(output_path=output_path))
    # EVALUATE A SAMPLE MANUALLY
    # test_sample_2 = np.array([[47.15, 2035.3, 512.2, 1132.09, 476.84, 2.05]])
    # Explanation = explainer(model, test_sample_2, quantization_regions, X_names, y_max, depth)
    # print(Explanation)


if __name__ == "__main__":
    main()
