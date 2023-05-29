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

### Install from source (Recommended)

> This method is recommended because the package is under development and there may be a lot of changes in the code.

```bash
git clone git@github.com:CaglarAytekin/LEURN_DEMO.git
cd LEURN_DEMO
# for normal installation
pip install .
# for development installation (changes in the code will be reflected immediately)
pip install -e .
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
import os

output_path = os.path.join(os.getcwd(), "leurn_example")
if not os.path.exists(output_path):
    print("Creating directory: {}".format(output_path))
    os.makedirs(output_path)

data = load_data("housing")
X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean, X_std = read_partition_process_data(
    data, target_name="median_house_value", task_type="reg"
)

model: LEURN = train_model(X_train, y_train, X_val, y_val, task_type="reg", output_path=output_path, epoch_no=1)
test_sample=X_test[0:1,:] # take one 1 sample per explanation
explain = model.explain(test_sample, feat_names=X_names, y_max=y_max)
plot_explaination(explain, os.path.join(output_path, "explain.png"))
```

![Explanation](assets/explain.png)

## Contact

If you have any problem about our code, feel free to contact: Caglar Aytekin <cagosmail@gmail.com>
