import os

import hydra
import pytest
import torch

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pt"), reason="Data files not found")
def test_data():
    train_set = torch.load(f"{_PATH_DATA}/processed/train_dataset.pt")
    test_set = torch.load(f"{_PATH_DATA}/processed/test_dataset.pt")

    assert len(train_set) == 45000, "Dataset did not have the correct number of samples"
    assert len(test_set) == 5000, "Dataset did not have the correct number of samples"
    assert train_set[:][0].shape == torch.Size([45000, 1, 28, 28]) and test_set[:][0].shape == torch.Size(
        [5000, 1, 28, 28]
    ), "The dimensions of samples are not correct"
    assert (torch.unique(train_set[:][1]) == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).sum() == torch.tensor(
        10
    ), "The dataset does not contain all the classes"
