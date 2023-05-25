"""
Attribution-NonCommercial-NoDerivatives 4.0 International

Copyright (c) 2023 Caglar Aytekin
"""
# Imports
from typing import Optional, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import BatchNormalization, Dense, Dropout, Input, Layer


class AddTauLayer(Layer):
    """Learns a tau vector directly, used only in the first layer."""

    def __init__(self, *args, **kwargs):
        super(AddTauLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.tau = self.add_weight(
            "tau", shape=input_shape[1:], initializer=tf.keras.initializers.random_uniform, trainable=True
        )

    def call(self, x):
        tau_added = x + self.tau
        tau = tau_added - x  # some trick to convert tau to usable format
        return tau


class EmbeddingBlock(Layer):
    def __init__(self, quantization_regions: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.bn = BatchNormalization(axis=-1)
        self.quantization_regions = quantization_regions

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, tau = inputs
        x = self.bn(x)
        x = tf.math.tanh(x + tau)  # Response to threshold
        # quantization_regions = -1 means no quantization
        if self.quantization_regions == -1:
            return x * tf.math.tanh(tau)  # return embedding
        # quantization_regions =  0 means quantization to -1,1
        # quantization_regions >  0 means quantization with quantization_regions+1 regions
        else:
            y = (x + 1) / 2  # bring to 0,1
            y = tf.math.round(y * self.quantization_regions) / self.quantization_regions  # quantize
            y = 2 * y - 1  # bring back to -1,1
            y = x + tf.stop_gradient(y - x)  # straight-through estimator
            return y * tf.math.tanh(self.tau)  # return embedding


def embedding_block(x, quantization_regions, tau, bn_no=0):
    # BN is directly applied on input
    x = BatchNormalization(axis=-1, name="batch_normalization_" + str(bn_no))(x)
    x = tf.math.tanh(x + tau)  # Response to threshold
    if quantization_regions == -1:
        return x * tf.math.tanh(tau)  # return embedding
    else:
        y = (x + 1) / 2  # bring to 0,1
        y = tf.math.round(y * quantization_regions) / quantization_regions  # quantize
        y = 2 * y - 1  # bring back to -1,1
        y = x + tf.stop_gradient(y - x)  # straight-through estimator
        return y * tf.math.tanh(tau)  # return embedding


# Defines LEURN model:
#     depth: depth of the model (except for first layer)
#     inp_dim: dimension of the input
#     class_no: 0 (regression), 1 (binary classification), >1 (multiclass classification)
#     quantization_regions: quantization type (see embedding block)
#     drop_rate: dropout rate


class LEURN(Model):
    def __init__(self, n_layers: int, n_classes: int, quantization_regions: int = 5, dropout_rate: float = 0.1):
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        pass


def def_model(n_layers, inp_dim, n_classes, quantization_regions, dropout_rate=0):
    # Finds new tau from previous embeddings by simple linear layer
    def tau_finder(embeddings, bn_no):
        embeddings = Dropout(dropout_rate)(embeddings)
        tau = Dense(
            inp_dim, kernel_initializer=tf.keras.initializers.GlorotNormal(), name="fully_connected_" + str(bn_no - 1)
        )(embeddings)
        return tau

    # Directly learns tau for first layer returns it with embedding
    def first_layer_embedding(x, quantization_regions):
        tau = AddTauLayer()(x)
        embedding = embedding_block(x, quantization_regions, tau)
        return tau, embedding

    alltau = []  # initialize a vector to contain all learned taus for later use

    Inp = Input(shape=(inp_dim,))
    tau, embeddings = first_layer_embedding(
        Inp, quantization_regions
    )  # Learn and return 1st layer's tau, return the first embedding
    alltau.append(-tau)  # Update alltau vector (note: we add tau to signal, so threshold is -tau)
    bn_no = 1
    for i in range(n_layers):
        tau = tau_finder(embeddings, bn_no)  # Find next tau from previous embeddings
        alltau.append(-tau)  # Update tau vector
        embedding_now = embedding_block(Inp, quantization_regions, tau, bn_no)  # Calculate embedding with new tau
        bn_no = bn_no + 1
        embeddings = tf.concat((embeddings, embedding_now), -1)  # Concatenate current embeddings to previous embeddings

    embeddings = Dropout(dropout_rate)(embeddings)
    if n_classes == 0:  # If class no is zero, no final activation (regression)
        output = Dense(
            1, kernel_initializer=tf.keras.initializers.GlorotNormal(), name="fully_connected_" + str(bn_no - 1)
        )(embeddings)
    elif n_classes == 1:  # If class no is one, activation is sigmoid (binary classification)
        output = Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            activation="sigmoid",
            name="fully_connected_" + str(bn_no - 1),
        )(embeddings)
    else:  # If class no larger than one, activation is softmax (multiclass classification)
        output = Dense(
            n_classes,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            activation="softmax",
            name="fully_connected_" + str(bn_no - 1),
        )(embeddings)

    model = Model(Inp, output)  # For training
    model_analyze = Model(Inp, (output, embeddings))  # For later analysis

    return model, model_analyze


# Prepare dataset pipe for training
def prepare_dataset(X_tr, Y_tr, X_val, Y_val, batch_no):
    tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
    tr_dataset = tr_ds.cache().shuffle(X_tr.shape[0]).batch(batch_no).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    val_dataset = val_ds.cache().batch(batch_no).prefetch(tf.data.AUTOTUNE)

    return tr_dataset, val_dataset


# Model training
# Takes model, training and validation sets, dataset name for saving , initial learning rate, number of epochs per learning rate and class number
def model_train(
    model,
    model_analyze,
    tr_dataset,
    val_dataset,
    dataset_name,
    init_lr,
    epoch_no,
    class_no,
    train_time=1,
    verbose=2,
    lr_reduction=10,
):
    if class_no == 0:  # Track minimum validation loss for regression problems
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model_" + dataset_name,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=True,
            save_best_only=True,
        )
    else:  # Track maximum accuracy for classification problems
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model_" + dataset_name,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=verbose,
            save_best_only=True,
        )

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=init_lr)

    if class_no == 0:
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt)
        model_analyze.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt)

    elif class_no == 1:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=["accuracy"])
        model_analyze.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=["accuracy"])
    else:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"])
        model_analyze.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"])

    # Train for set number of epochs per learning rate. Reduce LR, continue training
    lr = init_lr
    for tr in range(train_time):
        model.fit(
            tr_dataset,
            epochs=epoch_no,
            verbose=1,
            steps_per_epoch=len(tr_dataset),
            validation_data=val_dataset,
            callbacks=[model_checkpoint_callback],
        )
        model.optimizer.lr.assign(lr / lr_reduction)

    return model, model_analyze
