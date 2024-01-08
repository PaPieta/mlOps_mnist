import glob
import os

import pytest
from hydra import compose, initialize

from mlOps_mnist.models.train_model import train
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pt"), reason="Data files not found")
def test_train():
    with initialize(version_base=None, config_path=f"../mlOps_mnist/config", job_name="test_training"):
        config = compose(config_name="default_config.yaml")
    _, loss_list, model_savepath, fig_savepath = train(config)

    assert os.path.exists(f"{model_savepath}/model.pth"), "Model was not saved"
    assert os.path.exists(f"{fig_savepath}/loss.pdf"), "Loss plot was not saved"
    assert len(loss_list) > 0, "Loss list is empty"

    files = glob.glob(f"{model_savepath}/*")
    for f in files:
        os.remove(f)
    os.removedirs(f"{model_savepath}/")

    files = glob.glob(f"{fig_savepath}/*")
    for f in files:
        os.remove(f)
    os.removedirs(f"{fig_savepath}/")
