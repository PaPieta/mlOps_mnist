import datetime
import logging
import os

import hydra
import matplotlib.pyplot as plt

# from mlOps_mnist.models import model
import model_lightning
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

log = logging.getLogger(__name__)

log.info(f"Cuda available: {torch.cuda.is_available()}")


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train(config):
    """Train FNN on MNIST"""
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    model_hparams = config.model
    train_hparams = config.train
    torch.manual_seed(train_hparams["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_lightning.MyNeuralNet(model_hparams, train_hparams["x_dim"], train_hparams["class_num"]).to(device)
    train_set = torch.load(train_hparams["dataset_path"])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_hparams["batch_size"])

    early_stopping_callback = EarlyStopping(monitor="loss", patience=3, verbose=True, mode="min")

    trainer = Trainer(
        callbacks=[early_stopping_callback],
        max_epochs=train_hparams["n_epochs"],
        accelerator="gpu",
        logger=WandbLogger(project="fashion_mnist_mlops"),
        precision="16-mixed",
    )
    trainer.fit(net, train_dataloader)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_savepath = "models/" + timestamp
    os.makedirs(model_savepath)

    torch.save(net.state_dict(), f"{model_savepath}/model.pth")
    log.info(f"Model saved to {model_savepath}/model.pth")

    # fig = plt.figure()
    # plt.plot(loss_list)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title(f"Loss for model from {timestamp}")

    # fig_savepath = "reports/figures/" + timestamp
    # os.makedirs(fig_savepath)
    # plt.savefig(f"{fig_savepath}/loss.pdf")
    # log.info(f"Loss figure saved to {fig_savepath}/loss.pdf")

    # wandb.log({"Test loss plot": fig})


if __name__ == "__main__":
    train()
