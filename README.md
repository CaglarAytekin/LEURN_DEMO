# Code for LEURN: Learning Explainable Univariate Rules with Neural Networks

[Paper](https://arxiv.org/abs/2303.14937)

A demo is provided for training and making local explanations in Demo.py

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
(<https://creativecommons.org/licenses/by-nc-nd/4.0/>)

## Installation

### Install using pip

[Create personal access token on github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

```bash
pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/CaglarAytekin/LEURN_DEMO.git@v0.1
```

### Install from source

```bash
git clone git@github.com:CaglarAytekin/LEURN_DEMO.git
cd LEURN_DEMO
pip install .
```

## Running the demo

```bash
leurn-demo --help
# optional arguments:
#   -h, --help  show this help message and exit
#   -path PATH  output path
#   -bs BS      batch size
#   -lr LR      initial learning rate
#   -e E        number of epoch
#   -c C        number of training cycle, in each cycle, lr is reduced
#   -l L        depth of the network
#   -q Q        number of quantization regions
#   -d D        dropout rate
```

```bash
leurn-demo -path /tmp/housing -e 100 -c 2
```

Runing the tensorboard for monitoring the training process

```bash
tensorboard --logdir=/tmp/housing
```

Load the trained model and make local explanations

```python
from leurn import LEURN, plot_explaination
import json
model_config = json.load(open("/tmp/housing/model_config.json"))
model = LEURN(**model_config)
model.load_weights("/tmp/housing/best_model")
explain = model.explain(X_test, feat_names=X_names, y_max=y_max)
plot_explaination(explain, "/tmp/explain.png")
```

## Example

```python
from leurn import LEURN, load_data, plot_explaination, read_partition_process_data, train_model

data = load_data("housing")
X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean = read_partition_process_data(
    data, target_name="median_house_value", task_type="reg"
)
model: LEURN = train_model(X_train, y_train, X_val, y_val, task_type="reg", output_path="/tmp/housing", epoch_no=100)
explain = model.explain(X_test, feat_names=X_names, y_max=y_max)
plot_explaination(explain, "~/tmp/explain.png")
```

![Explaination](assets/explain.png)

## Contact

If you have any problem about our code, feel free to contact: Caglar Aytekin <cagosmail@gmail.com>
