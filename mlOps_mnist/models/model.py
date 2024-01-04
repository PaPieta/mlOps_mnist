import torch
import torch.nn.functional as F
from torch import nn


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, model_hparams: dict, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, model_hparams["l1_dim"])
        self.dr1 = nn.Dropout(p=model_hparams["dropout_rate"])
        self.fc2 = nn.Linear(model_hparams["l1_dim"], model_hparams["l2_dim"])
        self.dr2 = nn.Dropout(p=model_hparams["dropout_rate"])
        self.fc3 = nn.Linear(model_hparams["l2_dim"], model_hparams["l3_dim"])
        self.dr3 = nn.Dropout(p=model_hparams["dropout_rate"])
        self.fc4 = nn.Linear(model_hparams["l3_dim"], out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = x.view(x.shape[0], -1)

        x = self.dr1(F.relu(self.fc1(x)))
        x = self.dr2(F.relu(self.fc2(x)))
        x = self.dr3(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
