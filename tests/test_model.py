from tensorflow.keras.models import save_model

from leurn import LEURN


def test_create_model():
    pass


# save keras model to h5
m = LEURN(dim_red_coeff=100, n_layers=10, input_dim=2075)
m.visualize_model(expand_nested_ops=True)

# from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.optimizers import Adam