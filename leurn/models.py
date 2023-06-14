"""
Attribution-NonCommercial-NoDerivatives 4.0 International

Copyright (c) 2023 Caglar Aytekin
"""
import os
import warnings
from logging import warn
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Sequential
from keras.engine import data_adapter
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input, Layer


def _calculate_explainability(contrib, weight_now, bias_now, y_max, feat_names, feat_no):
    if y_max is None:
        y_max = 1  # for binary classification and multiclass classification

    #  DEPTH x QUANT x FEATURE
    contrib_bias_added = (tf.reduce_sum(contrib) + bias_now) * contrib / (tf.reduce_sum(contrib))
    Final_Contributions = tf.reduce_sum(contrib_bias_added, 0) * y_max
    Final_Feat_Name = feat_names

    # GLOBAL FEATURE IMPORTANCE (ROUGH)
    weight_now = tf.math.abs(np.reshape(weight_now, [-1, feat_no]))
    Global_Feat_Imp = tf.reduce_sum(weight_now, 0)
    Global_Feat_Imp = Global_Feat_Imp / tf.reduce_sum(Global_Feat_Imp)
    Final_Feat_Name = np.concatenate([Final_Feat_Name, np.array(["score"])])
    Global_Feat_Imp = np.concatenate([Global_Feat_Imp, np.array([np.NaN])])
    Final_Contributions = np.concatenate([Final_Contributions, np.array([np.sum(Final_Contributions)])])
    explanation = pd.DataFrame(
        {"Feature Name": Final_Feat_Name, "Global_Importance": Global_Feat_Imp, "Contribution": Final_Contributions}
    )
    return explanation


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
    """Embedding block for LEURN"""

    def __init__(self, quantization_regions: int = 5, name: Optional[str] = None):
        """
        Args:
            quantization_regions: Number of quantization regions
                quantization_regions = -1 means no quantization
                quantization_regions =  0 means quantization to -1,1
                quantization_regions >  0 means quantization with quantization_regions+1 regions

        """
        super().__init__(name=name)
        self.bn = BatchNormalization(axis=-1)
        self.quantization_regions = int(quantization_regions)
        self.input_dim = None

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        return super().build(input_shape)

    def get_output_shape_at(self, node_index):
        return (None, self.input_dim)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None) -> tf.Tensor:
        assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, "Input must be a tuple of (embeddings, tau)"
        x, tau = inputs

        x = self.bn(x, training=training)
        x = tf.math.tanh(x + tau)  # Response to threshold
        if self.quantization_regions == -1:
            return x * tf.math.tanh(tau)  # return embedding
        else:
            y = (x + 1) / 2  # bring to 0,1
            y = tf.math.round(y * self.quantization_regions) / self.quantization_regions  # quantize
            y = 2 * y - 1  # bring back to -1,1
            y = x + tf.stop_gradient(y - x)  # straight-through estimator
            return y * tf.math.tanh(tau)  # return embedding


class LEURN(Model):
    """LEURN model"""

    def __init__(
        self,
        input_dim: int = 10,
        n_classes: int = 1,
        n_layers: int = 10,
        quantization_regions: int = 5,
        dropout_rate: float = 0.1,
        dim_red_coeff: int = 0,
        name: Optional[str] = None,
    ):
        """
        Args:
            input_dim: dimension of the input
            n_classes: number of output classes, 0 for regression, 1 for binary classification,
                >1 for multiclass classification
            n_layers: number of hidden layers
            quantization_regions: quantization type (see embedding block)
            dropout_rate: dropout out rate
            dim_red_coeff: dimension reduction coefficient, if >0, then the number of neurons in tau_finder is `input_dim/dim_red_coeff`
        """
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.n_layers = int(n_layers)
        self.n_classes = int(n_classes)
        self.quantization_regions = int(quantization_regions)
        self.dropout_rate = float(dropout_rate)
        self.dim_red_coeff = int(dim_red_coeff)

        # ========  first embedding layer
        self.first_tau = AddTauLayer()
        self.first_embedding = EmbeddingBlock(self.quantization_regions, name="embedding_0")
        self.concat = Concatenate(axis=-1)

        # ======== middle layers
        for i in range(self.n_layers):
            if self.dim_red_coeff > 0:
                setattr(
                    self,
                    f"tau_finder{i + 1}",
                    Sequential(
                        [
                            Dropout(self.dropout_rate),
                            Dense(
                                input_dim // self.dim_red_coeff,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                name=f"dimension_reduction_{i+1}",
                            ),
                            Dense(
                                input_dim,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                name=f"fully_connected_{i+1}",
                            ),
                        ]
                    ),
                )
            else:
                setattr(
                    self,
                    f"tau_finder{i + 1}",
                    Sequential(
                        [
                            Dropout(self.dropout_rate),
                            Dense(
                                input_dim,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                name=f"fully_connected_{i+1}",
                            ),
                        ]
                    ),
                )

            setattr(self, f"embedding{i + 1}", EmbeddingBlock(self.quantization_regions, name=f"embedding_{i + 1}"))

        # ======== Output layer
        # If class no is zero, no final activation (regression)
        if self.n_classes == 0:  # If class no is zero, no final activation (regression)
            output = Dense(1, kernel_initializer=tf.keras.initializers.GlorotNormal(), name="fully_connected_output")
            self.task_type = "reg"
        elif self.n_classes == 1:  # If class no is one, activation is sigmoid (binary classification)
            output = Dense(
                1,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                activation="sigmoid",
                name="fully_connected_output",
            )
            self.task_type = "bin"
        elif self.n_classes > 1:  # If class no larger than one, activation is softmax (multiclass classification)
            output = Dense(
                int(self.n_classes),
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                activation="softmax",
                name="fully_connected_output",
            )
            self.task_type = "cls"
        else:
            raise ValueError(f"n_classes must be a positive integer, given {self.n_classes}")
        self.dropout = Dropout(self.dropout_rate)
        self.output_layer = output

        # ======== build the model
        self.build((None, self.input_dim))

    def train_step(self, data):
        """Just ignore the embeddings when training"""
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        """Just ignore the embeddings when scoring"""
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred, _ = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def functional(self) -> Model:
        """Convert the model to a functional model"""
        inputs = Input(shape=(self.input_dim,))
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs)

    def call(self, inputs, training=None):
        assert inputs.shape[-1] == self.input_dim, (
            f"Input dimension does not match, model created with input_dim={self.input_dim}, "
            f"given input has dimension {inputs.shape[-1]}"
        )

        alltau = []  # initialize a vector to contain all learned taus for later use

        # Directly learns tau for first layer returns it with embedding
        tau = self.first_tau(inputs)
        embeddings = self.first_embedding((inputs, tau), training=training)
        alltau.append(-tau)  # Update alltau vector (note: we add tau to signal, so threshold is -tau)

        for i in range(self.n_layers):
            # Finds new tau from previous embeddings by simple linear layer
            tau = getattr(self, f"tau_finder{i + 1}")(embeddings, training=training)
            alltau.append(-tau)

            # Calculate embedding with new tau
            embedding_now = getattr(self, f"embedding{i + 1}")((inputs, tau), training=training)
            embeddings = Concatenate(axis=-1)([embeddings, embedding_now])

        embeddings = self.dropout(embeddings, training=training)
        outputs = self.output_layer(embeddings)

        return outputs, embeddings

    def explain(
        self,
        test_sample: Union[tf.Tensor, tf.data.Dataset, np.ndarray, pd.DataFrame],
        feat_names: Optional[Sequence[str]] = None,
        y_max: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Finds Contributions of Each Rule in Each Layer in Input Subspace and Saves Them

        Returns:
            pd.DataFrame: Dataframe with contributions of each rule in each layer in input subspace,
                with columns ["Class", "Feature Name", "Global_Importance", "Contribution"]
        """
        if self.n_classes == 0 and y_max is None:
            raise ValueError("y_max must be provided for regression models")
        if feat_names is None:
            feat_names = [f"feat_{i}" for i in range(self.input_dim)]
        assert (
            len(feat_names) == self.input_dim
        ), f"feat_names must have length {self.input_dim}, given {len(feat_names)}"

        if isinstance(test_sample, pd.DataFrame):
            test_sample = test_sample.values
        elif isinstance(test_sample, tf.data.Dataset):
            pass
        if isinstance(test_sample, np.ndarray):
            test_sample = tf.convert_to_tensor(test_sample, dtype=tf.float32)
            if test_sample.shape[0] != 1:
                warnings.warn("Explain method only works for single sample, given multiple samples")
            if test_sample.ndim == 1:
                test_sample = tf.expand_dims(test_sample, 0)

        # Get output, taus, embeddings
        # embed_dim = input_dim * (n_layers + 1))
        _, embed = self(test_sample, training=False)  # [1, embed_dim]

        feat_no = len(feat_names)
        embed = np.swapaxes(embed, 1, 0)  # [embed_dim, 1]

        # Get the weight and bias of last layer
        output_layer = self.output_layer
        weight_now = output_layer.weights[0].numpy()  # [embed_dim, n_classes]
        bias_now = output_layer.weights[1].numpy()  # [n_classes]

        # Contributions to tau or final score (last layer) are calculated via weight*embedding
        n_classes = 1 if self.task_type in ("reg", "bin") else self.n_classes
        contrib = weight_now * embed  # [embed_dim, n_classes]
        contrib = np.reshape(contrib, [-1, feat_no, n_classes])

        # ======== regression or binary classification ========
        if n_classes == 1:
            explaination = _calculate_explainability(
                contrib=contrib[:, :, 0],
                weight_now=weight_now,
                bias_now=bias_now,
                y_max=y_max,
                feat_names=feat_names,
                feat_no=feat_no,
            )
            explaination["Class"] = 1
        # ======== multiclass classification ========
        else:
            # repeat for each class
            explaination = []
            for cls in range(n_classes):
                e = _calculate_explainability(
                    contrib=contrib[:, :, cls],
                    weight_now=weight_now[:, cls : cls + 1],
                    bias_now=bias_now[cls : cls + 1],
                    y_max=y_max,
                    feat_names=feat_names,
                    feat_no=feat_no,
                )
                e["Class"] = cls
                explaination.append(e)
            explaination = pd.concat(explaination)
        return explaination

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            dict(
                n_layers=self.n_layers,
                input_dim=self.input_dim,
                n_classes=self.n_classes,
                quantization_regions=self.quantization_regions,
                dropout_rate=self.dropout_rate,
            )
        )
        return cfg

    def visualize_model(self, expand_nested_ops: bool = False):
        """ Visualize the model using netron """
        try:
            import netron
        except ImportError:
            raise ImportError("Please install netron to visualize the model, `pip install netron`")
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            m = self.functional()

            if expand_nested_ops:
                try:
                    import tf2onnx
                except ImportError:
                    raise ImportError("Please install tf2onnx to expand nested ops, `pip install tf2onnx`")
                try:
                    import onnx
                except ImportError:
                    raise ImportError("Please install onnx to expand nested ops, `pip install onnx`")

                m = tf2onnx.convert.from_keras(m)[0]
                m = onnx.shape_inference.infer_shapes(m)
                onnx.save_model(m, os.path.join(tmpdir, "model.onnx"))
                netron.start(os.path.join(tmpdir, "model.onnx"))
            else:
                m.save(os.path.join(tmpdir, "model.h5"))
                netron.start(os.path.join(tmpdir, "model.h5"))

            input("Press Enter to exit...")
