import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils import *
import seaborn as sns
import operator
from functools import reduce
from losses import Loss
from scheduler import ReduceLROnPlateau_perso

seed = 2102

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)

sns.set_theme()
sns.set_context("paper")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


def test_domain_and_border():
    """
    Test function for domain_and_border.

    Loads a phi tensor from file and computes domain and boundary tensors.
    Displays visualizations of the domain and boundaries.
    """
    phi = torch.tensor(np.load("../data/Phi.npy"), dtype=torch.float32)[None, 20, :, :]
    tmp_loss = Loss()
    domain, domain_1, domain_2 = tmp_loss.compute_boundaries(phi, 2)
    domain, domain_1, domain_2 = (
        domain.int().numpy()[0, :, :],
        domain_1.int().numpy()[0, :, :],
        domain_2.int().numpy()[0, :, :],
    )
    plt.figure(figsize=(6, 6))
    plt.imshow(domain + domain_1 + domain_2, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def test_data():
    """
    Test function for loading and visualizing data.

    Loads tensors F, Phi, G, and W from file and displays visualizations.
    """

    phi = torch.tensor(np.load("../data/Phi.npy"), dtype=torch.float32)[10]
    F = torch.tensor(np.load("../data/F.npy"), dtype=torch.float32)[10]
    W = torch.tensor(np.load("../data/W.npy"), dtype=torch.float32)[10]
    G = torch.tensor(np.load("../data/G.npy"), dtype=torch.float32)[10]

    domain, boundary = domain_and_border(phi.shape[0], phi)
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.grid(False)
    plt.subplot(2, 3, 2)
    plt.imshow(boundary, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.grid(False)
    plt.subplot(2, 3, 3)
    plt.imshow(F * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("F")
    plt.grid(False)
    plt.subplot(2, 3, 4)
    plt.imshow(G * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("G")
    plt.grid(False)
    plt.subplot(2, 3, 5)
    plt.imshow(W * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("W")
    plt.grid(False)
    plt.subplot(2, 3, 6)
    plt.imshow((W * phi + G) * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("U")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


class UnitGaussianNormalizer(object):
    """
    Class for normalizing and denormalizing tensors using unit Gaussian normalization.
    """

    def __init__(self, x, eps=0.00001):
        """
        Initializes the normalizer.

        Args:
            x (torch.Tensor): Input tensor to compute mean and std.
            eps (float): Small value to avoid division by zero.
        """
        super(UnitGaussianNormalizer, self).__init__()

        self.means = torch.mean(x, dim=(0, 2, 3))
        self.stds = torch.std(x, dim=(0, 2, 3))
        self.eps = eps

    def encode(self, x):
        """
        Normalizes the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        x = (x - self.means[None, :, None, None].to(x.device)) / (
            self.stds[None, :, None, None].to(x.device) + self.eps
        )
        return x

    def decode(self, x):
        """
        Denormalizes the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be denormalized.
            sample_idx (None): Placeholder for compatibility.

        Returns:
            torch.Tensor: Denormalized tensor.
        """

        stds = (self.stds + self.eps).to(x.device)  # n
        means = self.means.to(x.device)
        x = (x * stds[None, :, None, None]) + means[None, :, None, None]
        return x


class DataLoader:
    """
    DataLoader class for loading and preprocessing input data.
    """

    def __init__(self, small_data=False, dtype=torch.float32):
        """
        Initializes the DataLoader.

        Args:
            small_data (bool): Whether to use a smaller subset of the data.
            dtype (torch.dtype): Data type of the tensors.
        """

        F = torch.tensor(np.load(f"../data/F.npy"), dtype=dtype)
        nb_vert = F.shape[1]
        self.nb_vert = nb_vert
        Phi = torch.tensor(np.load(f"../data/Phi.npy"), dtype=dtype)
        G = torch.tensor(
            np.load(f"../data/G.npy"),
            dtype=dtype,
        )
        Y = torch.tensor(np.load(f"../data/W.npy"), dtype=dtype)[:, None, :, :]

        if small_data:
            data_size = 300
        else:
            data_size = F.shape[0]

        if small_data:
            F = F[:data_size]
            Phi = Phi[:data_size]
            G = G[:data_size]
            Y = Y[:data_size]

        nb_vert = F.shape[1]

        X = torch.stack([F, Phi, G], dim=1)
        self.input_shape = (None, X.shape[1], X.shape[2], X.shape[3])
        if small_data:
            nb_val = 100
        else:
            nb_val = 300  # data_size // 8
        nb_train = data_size - nb_val
        print("data_size,nb_val,nb_train:", data_size, nb_val, nb_train)
        print("data shape:", self.input_shape)

        def separe(A):
            return A[:nb_train], A[nb_train:]

        self.X_train, self.X_val = separe(X)
        self.Y_train, self.Y_val = separe(Y)

        self.x_normalizer = UnitGaussianNormalizer(self.X_train)
        self.X_train_normed = self.x_normalizer.encode(self.X_train)
        self.X_val_normed = self.x_normalizer.encode(self.X_val)

        self.y_normalizer = UnitGaussianNormalizer(self.Y_train)
        self.Y_train_normed = self.y_normalizer.encode(self.Y_train)
        self.Y_val_normed = self.y_normalizer.encode(self.Y_val)

        self.nb_vert = self.X_train.shape[1]
        self.nb_train, self.nb_val = nb_train, nb_val


def test_DataLoader():
    data = DataLoader()
    print("X_train", data.X_train.shape)
    print("X_val", data.X_val.shape)
    print("Y_train", data.Y_train.shape)
    print("Y_val", data.Y_val.shape)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Conv2d(self.n_channels, 32, 1)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = nn.Conv2d(32, self.n_classes, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


class Agent:
    """
    Agent: A class representing an agent for training and validating a FNO model.

    This agent is designed for solving partial differential equations (PDEs) using a Fourier Neural Operator
    in 2D. It includes functionalities for training the model, validating its performance, and saving the best model.

    Args:
        data (object): An instance of the data class containing training and validation data.
        level (int): loss level for training. Defaults to 2.
        relative (bool): If True, use relative error in loss calculation. Defaults to True.
        squared (bool): If True, use squared error in loss calculation. Defaults to False.
        initial_lr (float): Initial learning rate for the optimizer. Defaults to 5e-3.
        n_modes (int): Number of Fourier modes to consider. Defaults to 10.
        width (int): Width of the network layers. Defaults to 20.
        batch_size (int): Batch size for training. Defaults to 64.
        l2_lambda (float): L2 regularization strength. Defaults to 1e-3.
        pad_prop (float): Proportion of padding to be applied to the input. Defaults to 0.0.
        pad_mode (str): Padding mode for convolutional layers. Defaults to "one_side_reflect".
        nb_layers (int): Number of layers in the network. Defaults to 4.
        activation (str): Activation function to be used. Supported options: 'relu', 'tanh', 'elu', 'gelu'.
            Defaults to 'gelu'.

    Attributes:
        data (object): An instance of the data class containing training and validation data.
        level (int): PDE level for training.
        relative (bool): If True, use relative error in loss calculation.
        squared (bool): If True, use squared error in loss calculation.
        initial_lr (float): Initial learning rate for the optimizer.
        n_modes (int): Number of Fourier modes to consider.
        width (int): Width of the network layers.
        batch_size (int): Batch size for training.
        l2_lambda (float): L2 regularization strength.
        pad_prop (float): Proportion of padding to be applied to the input.
        pad_mode (str): Padding mode for convolutional layers.
        nb_layers (int): Number of layers in the network.
        activation (str): Activation function used in the network.
        X_train (torch.Tensor): Training input data.
        X_train_normed (torch.Tensor): Normalized training input data.
        Y_train (torch.Tensor): Training output data.
        X_val (torch.Tensor): Validation input data.
        X_val_normed (torch.Tensor): Normalized validation input data.
        Y_val (torch.Tensor): Validation output data.
        nb_train_data (int): Number of training data points.
        nb_val_data (int): Number of validation data points.
        model (FNO2d): Neural network model for solving PDEs.
        optimizer (torch.optim.Adam): Optimizer for training the model.
        scheduler (ReduceLROnPlateau_perso): Learning rate scheduler.
        loss_function (Loss): Loss function for training.
        nb_batch (int): Number of batches in one training epoch.
        test_batch_size (int): Batch size for evaluating losses on a subset of data during training.
        losses (list): List to store training losses during training.
        losses_dict (list): List to store dictionary-based losses during training.
        losses_array (list): List to store array-based losses during training.
        nb_train_epochs (int): Number of training epochs completed.
        nweights (int): Number of learnable parameters in the model.
    """

    def __init__(
        self,
        data,
        level=2,
        relative=True,
        squared=False,
        initial_lr=5e-3,
        n_modes=10,
        width=20,
        batch_size=64,
        l2_lambda=1e-3,
        pad_prop=0.0,
        pad_mode="one_side_reflect",
        nb_layers=4,
        activation="gelu",
    ):
        self.data = data
        self.level = level
        self.relative = relative
        self.squared = squared
        self.initial_lr = initial_lr
        self.n_modes = n_modes
        self.width = width
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.pad_prop = pad_prop
        self.pad_mode = pad_mode
        self.nb_layers = nb_layers
        self.activation = activation
        self.X_train = self.data.X_train
        self.X_train_normed = self.data.X_train_normed
        self.Y_train = self.data.Y_train
        self.X_val = self.data.X_val
        self.X_val_normed = self.data.X_val_normed
        self.Y_val = self.data.Y_val

        nb_data_train = self.X_train.shape[0]
        in_channels = self.X_train.shape[1]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not (self.device == self.Y_train.device):
            self.X_train = self.data.X_train.to(self.device)
            self.X_train_normed = self.data.X_train_normed.to(self.device)
            self.Y_train = self.data.Y_train.to(self.device)
            self.X_val = self.data.X_val.to(self.device)
            self.X_val_normed = self.data.X_val_normed.to(self.device)
            self.Y_val = self.data.Y_val.to(self.device)

        self.nb_train_data = self.X_train.shape[0]
        self.nb_val_data = self.X_val.shape[0]

        self.model = UNet().to(self.device)
        #     in_channels,
        #     modes=self.n_modes,
        #     width=self.width,
        #     pad_prop=self.pad_prop,
        #     pad_mode=self.pad_mode,
        #     nb_layers=self.nb_layers,
        #     activation=self.activation,
        # ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_lr)
        self.scheduler = ReduceLROnPlateau_perso(
            self.optimizer,
            patience=20,
            factor=0.7,
            cooldown=5,
            min_lr=1e-5,
        )
        self.loss_function = Loss()
        self.nb_batch = nb_data_train // self.batch_size
        self.test_batch_size = 300
        if self.test_batch_size > self.X_val.shape[0]:
            self.test_batch_size = self.X_val.shape[0]
        self.losses = []
        self.losses_dict = []
        self.losses_array = []
        self.nb_train_epochs = 0
        self.nweights = 0
        for name, weights in self.model.named_parameters():
            if "bias" not in name:
                self.nweights = self.nweights + weights.numel()

    def train_one_epoch(self):
        """
        Train the model for one epoch using the training data.

        Updates the model parameters based on the training data and computes the training loss.
        """
        self.nb_train_epochs += 1
        X_train, Y_train = self.X_train, self.Y_train
        X_train_normed = self.X_train_normed
        rand_i = torch.randperm(X_train.shape[0])

        X = X_train_normed[rand_i]
        Y = Y_train[rand_i]
        X_denormed = X_train[rand_i]
        self.model.train()
        loss_i = 0.0
        for i in range(self.nb_batch):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x, y_true = X[sli], Y[sli]
            self.optimizer.zero_grad()
            y_pred_normed = self.model(x)
            y_pred = self.data.y_normalizer.decode(y_pred_normed)
            loss = self.loss_function(
                X_denormed[sli],
                y_pred,
                y_true,
                mode="train",
                level=self.level,
                relative=self.relative,
                squared=self.squared,
            )
            L2_term = torch.tensor(0.0, requires_grad=False)
            for name, weights in self.model.named_parameters():
                if "bias" not in name:
                    weights_sum_sq = torch.sum((weights * weights.conj()).real)
                    L2_term = L2_term + weights_sum_sq
            L2_term = L2_term / 2
            loss = loss + L2_term * self.l2_lambda
            loss.backward()
            self.optimizer.step()
            loss_i += loss / self.batch_size
        self.losses.append(loss / self.nb_batch)

    def validate(self):
        """
        Validate the model on the validation data.

        Returns:
            float: Mean validation loss on a part of the validation dataset.
        """
        indices = torch.randperm(self.X_val.shape[0])[: self.test_batch_size]
        x, y_true = self.X_val_normed[indices], self.Y_val[indices]
        self.model.eval()
        with torch.no_grad():
            y_pred_normed = self.model(x)
            y_pred = self.data.y_normalizer.decode(y_pred_normed)
            loss_v, loss_0_v, loss_1_v, loss_2_v = self.loss_function(
                self.X_val[indices],
                y_pred,
                y_true,
                mode="val",
                level=self.level,
                relative=self.relative,
                squared=self.squared,
            )
        return loss_v / self.test_batch_size

    def eval_losses(self):
        """
        Evaluate losses on a subset of training and validation data.

        Returns:
            float: Mean validation losses per dataset.
        """
        self.model.eval()
        with torch.no_grad():
            # check a part of the training sample
            indices = torch.randperm(self.X_train.shape[0])[: self.test_batch_size]
            x, y_true = self.X_train_normed[indices], self.Y_train[indices]
            y_pred_normed = self.model(x)
            y_pred = self.data.y_normalizer.decode(y_pred_normed)
            loss_T, loss_0_T, loss_1_T, loss_2_T = self.loss_function(
                self.X_train[indices],
                y_pred,
                y_true,
                mode="val",
                level=self.level,
                relative=True,
                squared=False,
            )

            # check a part of the validation sample
            indices = torch.randperm(self.X_val.shape[0])[: self.test_batch_size]
            x, y_true = self.X_val_normed[indices], self.Y_val[indices]
            y_pred_normed = self.model(x)
            y_pred = self.data.y_normalizer.decode(y_pred_normed)
            loss_v, loss_0_v, loss_1_v, loss_2_v = self.loss_function(
                self.X_val[indices],
                y_pred,
                y_true,
                mode="val",
                level=self.level,
                relative=True,
                squared=False,
            )
        self.losses_dict.append(
            {
                "epoch": self.nb_train_epochs,
                "loss_T": f"{(loss_T.item() / self.test_batch_size):.2e}",
                "loss_0_T": f"{(loss_0_T.item() / self.test_batch_size):.2e}",
                "loss_1_T": f"{(loss_1_T.item() / self.test_batch_size):.2e}",
                "loss_2_T": f"{(loss_2_T.item() / self.test_batch_size):.2e}",
                "loss_V": f"{(loss_v.item() / self.test_batch_size):.2e}",
                "loss_0_V": f"{(loss_0_v.item() / self.test_batch_size):.2e}",
                "loss_1_V": f"{(loss_1_v.item() / self.test_batch_size):.2e}",
                "loss_2_V": f"{(loss_2_v.item() / self.test_batch_size):.2e}",
            }
        )
        self.losses_array.append(
            [
                self.nb_train_epochs,
                loss_T.item() / self.test_batch_size,
                loss_0_T.item() / self.test_batch_size,
                loss_1_T.item() / self.test_batch_size,
                loss_2_T.item() / self.test_batch_size,
                loss_v.item() / self.test_batch_size,
                loss_0_v.item() / self.test_batch_size,
                loss_1_v.item() / self.test_batch_size,
                loss_2_v.item() / self.test_batch_size,
            ]
        )
        return loss_v.item() / self.test_batch_size

    def train(self, nb_epochs=2000, models_repo="./models", save_models=True):
        """
        Train the model for multiple epochs and save the best model.

        Args:
            nb_epochs (int): Number of training epochs. Defaults to 2000.
            models_repo (str): Directory to store saved models. Defaults to "./models".
            save_models (bool): If True, save models at specified intervals. Defaults to True.
        """

        best_mean_loss_validation = 1e8
        if not os.path.exists(f"{models_repo}/best_model"):
            os.makedirs(f"./{models_repo}/best_model")
        for epoch in range(1, nb_epochs + 1):
            self.train_one_epoch()
            validation_loss = self.validate()
            self.scheduler.step(validation_loss)
            if self.scheduler.lr_has_changed:
                if self.scheduler.patience <= 60:
                    self.scheduler.patience = int(self.scheduler.patience * 1.5)
                print(
                    f"{self.scheduler.patience = } {self.scheduler._last_lr[-1] = :.3e}"
                )
            if epoch % 1 == 0:
                mean_loss_validation = self.eval_losses()
                if mean_loss_validation <= best_mean_loss_validation:
                    best_mean_loss_validation = mean_loss_validation

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        f"{models_repo}/best_model.pkl",
                    )

                    file = open(
                        f"{models_repo}/best_model/best_epoch.txt",
                        "w",
                    )
                    file.write(
                        f"(level, relative, squared, initial_lr, n_modes, width, batch_size, l2_lambda, pad_prop, pad_mode) = "
                        + f"{(self.level, self.relative, self.squared, self.initial_lr, self.n_modes, self.width, self.batch_size, self.l2_lambda, self.pad_prop, self.pad_mode)} \n"
                    )
                    file.write(f"best_epoch = {epoch} \n")
                    file.write(f"loss = {best_mean_loss_validation} \n")
                    file.write(f"loss = {best_mean_loss_validation:.2e} \n")
                    file.write(f"{self.losses_dict[-1]} \n")
                    file.close()
                print(f"{self.losses_dict[-1]}")

            if epoch % 10 == 0:
                if save_models:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        f"./{models_repo}/model_{epoch}.pkl",
                    )

            np.save(f"{models_repo}/losses_array.npy", np.array(self.losses_array))

    def plot_losses(self, models_repo="./models", save=True):
        losses_array = np.load(f"{models_repo}/losses_array.npy")

        epochs = losses_array[:, 0]
        list_loss_train = losses_array[:, 1]
        list_loss_0_train = losses_array[:, 2]
        list_loss_1_train = losses_array[:, 3]
        list_loss_2_train = losses_array[:, 4]
        list_loss_val = losses_array[:, 5]
        list_loss_0_val = losses_array[:, 6]
        list_loss_1_val = losses_array[:, 7]
        list_loss_2_val = losses_array[:, 8]

        file = open(f"{models_repo}/best_model/best_epoch.txt")
        for y in file.read().split(" "):
            if y.isdigit():
                best_epoch = int(y)
        best_epoch_index = list(epochs).index(best_epoch)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        indices_to_plot = list(range(0, len(epochs), 5))
        ax0.plot(
            epochs[indices_to_plot],
            list_loss_0_train[indices_to_plot],
            color="b",
            label=r"$\mathcal{L}_0(train)$",
        )
        ax0.plot(
            epochs[indices_to_plot],
            list_loss_1_train[indices_to_plot],
            color="purple",
            label=r"$\mathcal{L}_1(train)$",
        )
        ax0.plot(
            epochs[indices_to_plot],
            list_loss_2_train[indices_to_plot],
            "r",
            label=r"$\mathcal{L}_2(train)$",
        )
        ax0.plot(
            epochs[indices_to_plot],
            list_loss_train[indices_to_plot],
            linestyle=(0, (5, 8)),
            color="k",
            alpha=0.6,
            label=r"$\mathcal{L}(train)$",
        )

        ax0.plot(
            best_epoch,
            list_loss_0_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax0.plot(
            best_epoch,
            list_loss_1_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax0.plot(
            best_epoch,
            list_loss_2_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax0.plot(
            best_epoch,
            list_loss_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )

        ax0.legend(
            ncol=2,
            fontsize=16,
        )

        ax0.set_xlabel("Epochs", fontsize=16)
        ax0.set_yscale("log")
        ax0.set_title(
            r"Evolution of $\mathcal{L}$ and $\mathcal{L}_i$ on the training set",
            fontsize=16,
        )

        ax1.plot(
            epochs[indices_to_plot],
            list_loss_0_val[indices_to_plot],
            color="b",
            label=r"$\mathcal{L}_0(val)$",
        )
        ax1.plot(
            epochs[indices_to_plot],
            list_loss_1_val[indices_to_plot],
            color="purple",
            label=r"$\mathcal{L}_1(val)$",
        )
        ax1.plot(
            epochs[indices_to_plot],
            list_loss_2_val[indices_to_plot],
            color="r",
            label=r"$\mathcal{L}_2(val)$",
        )
        ax1.plot(
            epochs[indices_to_plot],
            list_loss_val[indices_to_plot],
            linestyle=(0, (5, 8)),
            color="k",
            alpha=0.8,
            label=r"$\mathcal{L}(val)$",
        )

        ax1.plot(
            best_epoch,
            list_loss_0_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax1.plot(
            best_epoch,
            list_loss_1_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax1.plot(
            best_epoch,
            list_loss_2_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax1.plot(
            best_epoch,
            list_loss_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )

        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_title(
            r"Evolution of $\mathcal{L}$ and $\mathcal{L}_i$ on the validation set",
            fontsize=16,
        )
        ax1.legend(
            ncol=2,
            fontsize=16,
        )

        plt.tight_layout()

        if not (os.path.exists("./images")) and save:
            os.makedirs("./images")
        if save:
            # plt.savefig(f"./images/losses.png")
            plt.savefig(f"./{models_repo}/losses.png")
        plt.show()


if __name__ == "__main__":
    test_domain_and_border()
    test_data()
    data = DataLoader(False)
    # {'level': 2, 'relative': True, 'squared': False, 'initial_lr': 0.001, 'n_modes': 25, 'width': 16, 'batch_size': 32, 'l2_lambda': 1e-05, 'pad_prop': 0.05, 'pad_mode': 'constant', 'nb_layers': 4, 'activation': 'gelu', 'essaie': 1}

    training_agent = Agent(
        data,
        level=0,
        relative=True,
        squared=False,
        initial_lr=1e-3,
        n_modes=10,
        width=20,
        batch_size=32,
        pad_prop=0.05,
        pad_mode="reflect",
        l2_lambda=1e-3,
    )
    print(
        f"(level, relative, squared, initial_lr, n_modes, width, batch_size, l2_lambda, pad_prop, pad_mode) = "
        + f"{(training_agent.level, training_agent.relative, training_agent.squared, training_agent.initial_lr, training_agent.n_modes, training_agent.width, training_agent.batch_size, training_agent.l2_lambda, training_agent.pad_prop, training_agent.pad_mode)} \n"
    )
    nb_epochs = 2000
    start_training = time.time()
    training_agent.train(nb_epochs, models_repo="./models_unet")
    end_training = time.time()
    time_training = end_training - start_training
    print(
        f"Total time to train the operator : {time_training:.3f}s. Average time : {time_training/nb_epochs:.3f}s."
    )
    training_agent.plot_losses(models_repo="./models_unet")  # models_repo=repos[i])
