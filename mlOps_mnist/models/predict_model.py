import click
import hydra

# import model
import torch

from mlOps_mnist.models import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base=None)
def predict(config):
    """Predict model on data."""
    print("Predicting model on data")
    model_hparams = config.model
    train_hparams = config.train

    net = model.MyNeuralNet(model_hparams, train_hparams["x_dim"], train_hparams["class_num"]).to(device)
    net.load_state_dict(torch.load(model_hparams["model_pred_path"], map_location=torch.device("cpu")))
    net.eval()
    data = torch.load(train_hparams["test_dataset_path"], map_location=torch.device("cpu"))

    pred = net(data[:][0].to(device)).cpu().detach()

    click.echo(torch.argmax(pred, dim=1))
    return pred


# def predict(
#     model: torch.nn.Module,
#     data: torch.tensor
# ) -> None:
#     """Run prediction for a given model and dataloader.

#     Args:
#         model: model to use for prediction
#         dataloader: dataloader with batches

#     Returns
#         Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

#     """

#     return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == """__main__""":
    predict()
