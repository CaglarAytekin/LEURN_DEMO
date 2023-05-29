"""
Attribution-NonCommercial-NoDerivatives 4.0 International

Copyright (c) 2023 Caglar Aytekin
"""
import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from leurn.models import LEURN


def read_partition_process_data(filename: Union[str, pd.DataFrame], target_name: str, task_type: str):
    """Reads the data, partitions it, and processes it for the given task type.

    Args:
        filename: str or pandas dataframe
        target_name: name of the target variable
        task_type: "reg" - regression, "cls" - multi-class classification, "bin" - binary classification
    """
    # READ THE DATA
    if isinstance(filename, str):
        ext = os.path.splitext(filename)[1]
        if ext in [".csv", ".tsv"]:
            data_frame = pd.read_csv(filename)
        elif ext in [".xls", ".xlsx"]:
            data_frame = pd.read_excel(filename)
        else:
            raise ValueError(f"Unknown file type: {filename}")
    elif isinstance(filename, pd.DataFrame):
        data_frame = filename
    else:
        raise ValueError("filename must be a string or a pandas dataframe")

    # "median_house_value"
    X_df = data_frame.drop([target_name], axis=1)  # FEATURES
    y_df = data_frame[target_name]  # TARGET
    X_names = X_df.columns  # feature names
    X = X_df.values  # features
    y = y_df.values  # target

    # PARTITION THE DATASET
    permute = np.random.permutation(np.arange(0, y.__len__(), 1))  # create a random permutation
    train_indices = permute[0 : int(permute.__len__() * 0.8)]  # First 80% is training data
    val_indices = permute[int(permute.__len__() * 0.8) : int(permute.__len__() * 0.9)]  # next 10% is validation data
    test_indices = permute[int(permute.__len__() * 0.9) :]  # rest is test data

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    # HANDLE MISSING VALUES
    X_mean = np.abs(np.nanmean(X_train, axis=0, keepdims=False))
    missing_sample, missing_feature = np.where(np.isnan(X_train))
    for m in range(missing_sample.__len__()):
        X_train[missing_sample[m], missing_feature[m]] = X_mean[missing_feature[m]]
    missing_sample, missing_feature = np.where(np.isnan(X_val))
    for m in range(missing_sample.__len__()):
        X_val[missing_sample[m], missing_feature[m]] = X_mean[missing_feature[m]]
    missing_sample, missing_feature = np.where(np.isnan(X_test))
    for m in range(missing_sample.__len__()):
        X_test[missing_sample[m], missing_feature[m]] = X_mean[missing_feature[m]]

    # PROCESS THE DATASET - Find max of features in training set, and normalize all partitions accordingly
    y_max = None
    if task_type == "reg":  # if the task is regression, normalize the target too
        y_max = np.max(np.abs(y), axis=0, keepdims=True) + (1e-10)
        y_train = y_train / y_max
        y_val = y_val / y_max
        y_test = y_test / y_max
        y_max = tf.cast(y_max, dtype=tf.float32)
        
    #Standardize the dataset:
    X_mean=np.mean(X_train,axis=0,keepdims=True)
    X_std=np.std(X_train,axis=0,keepdims=True)
    X_train=(X_train-X_mean)/(X_std+1e-6)
    X_val=(X_val-X_mean)/(X_std+1e-6)
    X_test=(X_test-X_mean)/(X_std+1e-6)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean,X_std


def prepare_dataset_for_tf(X_tr, Y_tr, X_val, Y_val, batch_no):
    """Prepare dataset pipe for training"""
    tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
    tr_dataset = tr_ds.cache().shuffle(X_tr.shape[0]).batch(batch_no).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    val_dataset = val_ds.cache().batch(batch_no).prefetch(tf.data.AUTOTUNE)

    return tr_dataset, val_dataset


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    output_path: str,
    n_layers: int = 10,
    quantization_regions: int = 5,
    dropout_rate: float = 0.1,
    batch_size: int = 512,
    learning_rate: float = 5e-3,
    n_train_cycles: int = 2,
    epoch_no: int = 10,
    lr_reduction: int = 5,
    verbose: int = 2,
) -> LEURN:
    """ """
    output_path = os.path.realpath(output_path)

    if task_type == "bin":
        class_no = 1
        print("Binary classification task")
    elif task_type == "cls":
        class_no = y_train.max() + 1
        print("Multi-class classification task")
    elif task_type == "reg":
        class_no = 0
        print("Regression task")
    else:
        raise ValueError(f"Unknown task type {task_type}, must be one of 'bin', 'cls', 'reg'")

    tr_dataset, val_dataset = prepare_dataset_for_tf(X_train, y_train, X_val, y_val, batch_no=batch_size)
    model = LEURN(
        n_layers=n_layers,
        input_dim=X_val.shape[1],
        n_classes=class_no,
        quantization_regions=quantization_regions,
        dropout_rate=dropout_rate,
    )
    model_config = dict(
        n_layers=n_layers,
        input_dim=X_val.shape[1],
        n_classes=class_no,
        quantization_regions=quantization_regions,
        dropout_rate=dropout_rate,
    )
    with open(os.path.join(output_path, "model_config.json"), "w") as f:
        json.dump(model_config, f)

    # model.summary()
    best_model_path = os.path.join(output_path, "best_model")

    if class_no == 0:  # Track minimum validation loss for regression problems
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=True,
            save_best_only=True,
        )
    else:  # Track maximum accuracy for classification problems
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=verbose,
            save_best_only=True,
        )
    # add tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=output_path, histogram_freq=1)

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if class_no == 0:
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt)

    elif class_no == 1:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=["accuracy"])
    else:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"])

    print(
        f"Training start, you can monitor the training process using Tensorboard: 'tensorboard --logdir={output_path} --bind_all'"
    )
    # Train for set number of epochs per learning rate. Reduce LR, continue training
    lr = learning_rate
    for _ in range(n_train_cycles):
        model.fit(
            tr_dataset,
            epochs=epoch_no,
            verbose=1,
            steps_per_epoch=len(tr_dataset),
            validation_data=val_dataset,
            callbacks=[model_checkpoint_callback, tensorboard_callback],
        )
        model.optimizer.lr.assign(lr / lr_reduction)

    print(f"Training finished. Loading best model from {best_model_path}")
    model.load_weights(best_model_path)
    return model


def plot_explaination(Explanation: pd.DataFrame, output_path: str):
    # PLOT EXPLANATIONS
    import seaborn as sns
    from matplotlib import pyplot as plt

    output_path = os.path.realpath(os.path.expanduser(output_path))

    sns.set()
    feat_name = Explanation["Feature Name"].values
    importance = Explanation["Global_Importance"].values[:-1]
    importance = np.array(importance)
    importance = importance.astype(np.float)
    contribution = Explanation["Contribution"].values

    n_features = len(feat_name)
    plt.figure(figsize=(30 if n_features >= 20 else 20, 8), dpi=200)

    plt.subplot(1, 2, 1)
    plt.stem(feat_name[:-1], importance, "-o")
    plt.xticks(rotation=-90)
    plt.title("Global Importance")

    plt.subplot(1, 2, 2)
    plt.stem(feat_name, contribution, "-o")
    plt.xticks(rotation=-90)
    plt.title("Contribution")

    plt.tight_layout()
    plt.savefig(output_path)
