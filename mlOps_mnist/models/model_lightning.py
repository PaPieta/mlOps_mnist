import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim


class MyNeuralNet(LightningModule):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, model_hparams: dict, in_features: int, out_features: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, model_hparams["l1_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
            nn.Linear(model_hparams["l1_dim"], model_hparams["l2_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
            nn.Linear(model_hparams["l2_dim"], model_hparams["l3_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_hparams["l3_dim"], out_features),
            nn.LogSoftmax(dim=1),
        )

        self.criteriun = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = x.view(x.shape[0], -1)
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criteriun(output, target)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
