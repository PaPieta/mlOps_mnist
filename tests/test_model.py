import os

import pytest
import torch
from hydra import compose, initialize

from mlOps_mnist.models import model
from mlOps_mnist.models.predict_model import predict
from tests import _PATH_DATA


@pytest.mark.parametrize("test_data", [torch.randn(1, 1, 28, 28), torch.randn(20, 1, 28, 28)])
def test_model(test_data):
    """Test model output dimensions on random input."""

    model_hparams = {"l1_dim": 256, "l2_dim": 128, "l3_dim": 64, "dropout_rate": 0.2}

    net = model.MyNeuralNet(model_hparams, 784, 10)
    y = net(test_data)
    assert y.shape == torch.Size([test_data.shape[0], 10]), "Model output shape is incorrect"


# @pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pt"), reason="Data files not found")
# def test_predict_model():
#     """Test predict_model fuction."""

#     with initialize(version_base=None, config_path=f"../mlOps_mnist/config", job_name="test_training"):
#         config = compose(config_name="default_config.yaml")
#     pred = predict(config)
#     assert pred.shape == torch.Size([5000, 10]), "Model output shape is incorrect"


def test_error_on_wrong_shape():
    model_hparams = {"l1_dim": 256, "l2_dim": 128, "l3_dim": 64, "dropout_rate": 0.2}

    net = model.MyNeuralNet(model_hparams, 784, 10)
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        net(torch.randn(1, 2, 3))
