__version__ = "0.1"

__showed_license__ = False

LICENSE = """
# ==================================================================================== #
# Demo for LEURN: Learning Explainable Univariate Rules with Neural Networks           #
# [Paper](https://arxiv.org/abs/2303.14937)                                            #
# This work is licensed under a                                                        #
# Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  #
# (<https://creativecommons.org/licenses/by-nc-nd/4.0/>)                               #
# For a fast demo, please run: leurn-demo                                              #
# For more information, please contact: Caglar Aytekin <cagosmail@gmail.com>           #
# ==================================================================================== #
"""

if not __showed_license__:
    print(LICENSE)
    __showed_license__ = True


from leurn.data import load_data
from leurn.models import LEURN
from leurn.utils import plot_explaination, read_partition_process_data, train_model
