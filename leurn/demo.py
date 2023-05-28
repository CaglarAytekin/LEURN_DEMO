import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from leurn.data import load
from leurn.utils import explainer, read_partition_process_data, train_model


def helloworld():
    print("Hello world!")


def main():
    tf.debugging.set_log_device_placement(False)
    tf.autograph.set_verbosity(0)

    # set memory growth for GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        warnings.warn("No GPU found. Running on CPU")
    else:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # HYPERPARAMETERS
    batch_size = 512  # batch size
    learning_rate = 5e-3  # initial learning rate
    cycle_no = 2  # how many cycles to train (in each cycle, learning rate is reduced)
    lr_reduction = 5  # reduction rate of learning rate per cycle
    epoch_no = 100  # how many epochs to train per cycle
    depth = 10  # model depth
    quantization_regions = 5  # tanh quantization number
    drop_rate = 0.1  # dropout rate : non-zero dropout is heavily recommended for more meaningful explanations
    tasktype = -1  # 0: binary classification, 1: multi-class classification, -1: regression
    dataset_name = "housing"  # name of the dataset - regression case

    # READ, PARTITION AND PROCESS THE DATA
    X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean = read_partition_process_data(
        load("housing"), tasktype
    )

    # TRAINS THE MODEL
    model, model_analyse = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        depth,
        quantization_regions,
        drop_rate,
        tasktype,
        dataset_name,
        batch_size,
        learning_rate,
        cycle_no,
        epoch_no,
        lr_reduction,
        verbose=2,
    )

    # #EVALUATE A SAMPLE FROM TEST DATA
    sample_indice = 1
    test_sample = X_test[sample_indice : sample_indice + 1]

    # #RETURN EXPLANATIONS
    Explanation = explainer(model_analyse, test_sample, quantization_regions, X_names, y_max, depth)
    print(Explanation)

    # EVALUATE A SAMPLE MANUALLY
    test_sample_2 = np.array([[47.15, 2035.3, 512.2, 1132.09, 476.84, 2.05]])
    Explanation = explainer(model_analyse, test_sample_2, quantization_regions, X_names, y_max, depth)
    print(Explanation)


if __name__ == "__main__":
    main()
