import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
import torch.nn as nn
import time
from utils import *
import seaborn as sns
import operator
from functools import reduce
from losses import Loss
from scheduler import LR_Scheduler

seed = 221024

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)

sns.set_theme("paper", rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


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
        self.means = torch.nanmean(x, dim=(0, 2, 3))
        self.stds = torch.tensor(
            np.nanstd(x.cpu().detach().numpy(), axis=(0, 2, 3)), dtype=torch.float32
        )
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

        F = torch.tensor(np.load(f"../../data/F.npy"), dtype=dtype)
        nb_vert = F.shape[1]
        self.nb_vert = nb_vert
        Phi = torch.tensor(np.load(f"../../data/Phi.npy"), dtype=dtype)
        G = torch.tensor(
            np.load(f"../../data/G.npy"),
            dtype=dtype,
        )
        Y = torch.tensor(np.load(f"../../data/U_std_fem.npy"), dtype=dtype)[
            :, None, :, :
        ]

        if small_data:
            data_size = 300
        else:
            data_size = F.shape[0]

        if small_data:
            F = F[:data_size]
            Phi = Phi[:data_size]
            G = G[:data_size]
            Y = Y[:data_size]

        X = torch.stack([F, Phi, G], dim=1)
        self.input_shape = (None, X.shape[1], X.shape[2], X.shape[3])
        if small_data:
            nb_val = 100
        else:
            nb_val = 300  # data_size // 5
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

        # We don't want to take the zeros outside Omega_h into account for the normalization of the output
        self.Y_train_nan = torch.zeros_like(self.Y_train)
        loss = Loss()
        domain_tmp = self.X_train[:, 1, :, :] <= 3e-16
        neighborhood = loss.neighborhood_6(domain_tmp)
        domain = ((neighborhood.int() + domain_tmp.int()) != 0).cpu().detach().numpy()
        domains_tmp = domain.flatten()
        domains_nan = domain.copy().flatten().astype(float)
        domains_nan[np.where(domains_tmp == False)] = np.nan
        domains_nan = np.reshape(domains_nan, domain.shape)
        domains_nan = torch.tensor(domains_nan[:, None, :, :], dtype=dtype)

        Y_train_nan = domains_nan * self.Y_train
        self.y_normalizer = UnitGaussianNormalizer(Y_train_nan)
        Y_train_nan = None
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


class SpectralConv2d(nn.Module):
    """
    SpectralConv2d: Complex-valued 2D convolutional layer using spectral convolution.

    This layer applies a 2D convolution in the frequency domain using complex-valued weights.
    The weights are parameterized by complex-valued tensors and are used to perform spectral
    convolution via element-wise multiplication in the Fourier space.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int): Number of Fourier modes used for the convolution.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int): Number of Fourier modes used for the convolution.
        scale (float): Scaling factor for the weights initialization.
        weights1 (nn.Parameter): Learnable complex-valued weights for the first convolution.
        weights2 (nn.Parameter): Learnable complex-valued weights for the second convolution.

    Methods:
        compl_mul2d(input, weights):
            Perform complex multiplication of 2D tensors.

        forward(x):
            Forward pass of the spectral convolution layer.

    """

    def __init__(self, in_channels, out_channels, modes):
        """
        Initializes the SpectralConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of Fourier modes used for the convolution.
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels,
                self.out_channels,
                self.modes,
                self.modes,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels,
                self.out_channels,
                self.modes,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(self, input, weights):
        """
        Perform complex multiplication of 2D tensors.

        Args:
            input (torch.Tensor): Input tensor with shape (batch, in_channel, x, y).
            weights (nn.Parameter): Complex-valued weights with shape
                                   (in_channel, out_channel, x, y).

        Returns:
            torch.Tensor: Result of complex multiplication with shape
                          (batch, out_channel, x, y).
        """
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward pass of the spectral convolution layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channel, x, y).

        Returns:
            torch.Tensor: Output tensor after spectral convolution with shape
                          (batch, out_channel, x, y).
        """

        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        factor1 = self.compl_mul2d(
            x_ft[:, :, : self.modes, : self.modes],
            self.weights1,
        )
        factor2 = self.compl_mul2d(
            x_ft[:, :, -self.modes :, : self.modes],
            self.weights2,
        )
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes, : self.modes] = factor1
        out_ft[:, :, -self.modes :, : self.modes] = factor2

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2dLayer(nn.Module):
    def __init__(self, conv, bias):
        super(FNO2dLayer, self).__init__()

        self.conv = conv
        self.bias = bias

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bias(x)
        return x1 + x2


class FNO2d(nn.Module):
    """
    FNO2d: Neural network for solving partial differential equations (PDEs) using Fourier Neural Operators in 2D.

    This network consists of multiple layers, each comprising a spectral convolution (SpectralConv2d)
    followed by a bias term (nn.Conv2d), with activation functions applied in between.
    The architecture is designed for solving PDEs on 2D domains.

    Args:
        in_channels (int): Number of input channels.
        modes (int): Number of Fourier modes to consider.
        width (int): Width of the network layers.
        pad_prop (float): Proportion of padding to be applied to the input.
        nb_layers (int): Number of layers in the network.
        pad_mode (str): Padding mode for convolutional layers.
        activation (str): Activation function to be used. Supported options: 'relu', 'tanh', 'elu', 'gelu'.
    """

    def __init__(
        self,
        in_channels,
        modes,
        width,
        pad_prop=0.05,
        nb_layers=4,
        pad_mode="reflect",
        activation="gelu",
    ):
        super(FNO2d, self).__init__()

        self.modes = modes
        self.width = width
        self.pad_prop = pad_prop
        self.pad_mode = pad_mode
        assert (
            self.pad_mode == "reflect"
            or self.pad_mode == "constant"
            or self.pad_mode == "replicate"
            or self.pad_mode == "one_side_reflect"
        )
        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "tanh":
            self.activation = torch.nn.functional.tanh
        elif activation == "elu":
            self.activation = torch.nn.functional.elu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise Exception(f"activation function:{activation} not allowed")
        self.in_channels = in_channels
        self.fc0 = nn.Conv2d(
            in_channels + 2, self.width, 1
        )  # input channel is 5: (f(x,y), phi(x,y), g(x,y), x, y)
        self.nb_layers = nb_layers

        self.layers = nn.ModuleList()
        for i in range(self.nb_layers):
            self.layers.append(
                FNO2dLayer(
                    SpectralConv2d(self.width, self.width, self.modes),
                    nn.Conv2d(self.width, self.width, 1),
                )
            )

        self.fc1 = nn.Conv2d(self.width, 128, 1)
        self.fc2 = nn.Conv2d(128, 1, 1)

    def get_grid(self, shape, device):
        """
        Generate a grid of coordinates based on the input shape.

        Args:
            shape (torch.Size): Shape of the input tensor.
            device (torch.device): Device on which the grid tensor should be created.

        Returns:
            torch.Tensor: Grid tensor with coordinates.
        """
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def forward(self, x):
        """
        Forward pass of the FNO2d network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        self.padding = int(x.shape[2] * self.pad_prop)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        if self.padding != 0 and self.pad_mode != "one_side_reflect":
            x = torch.nn.functional.pad(
                x,
                [self.padding, self.padding, self.padding, self.padding],
                mode=self.pad_mode,
            )
        elif self.padding != 0 and self.pad_mode == "one_side_reflect":
            x = torch.nn.functional.pad(
                x,
                [0, self.padding, 0, self.padding],
                mode="reflect",
            )
        for i in range(self.nb_layers):
            x = self.layers[i](x)
            if i < self.nb_layers - 1:
                x = self.activation(x)
        if self.padding != 0 and self.pad_mode != "one_side_reflect":
            x = x[..., self.padding : -self.padding, self.padding : -self.padding]
        elif self.padding != 0 and self.pad_mode == "one_side_reflect":
            x = x[..., : -self.padding, : -self.padding]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


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
        scheduler (LR_Scheduler): Learning rate scheduler.
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
        level=1,
        relative=True,
        squared=False,
        initial_lr=5e-3,
        n_modes=10,
        width=20,
        batch_size=32,
        l2_lambda=1e-3,
        pad_prop=0.05,
        pad_mode="reflect",
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

        self.model = FNO2d(
            in_channels,
            modes=self.n_modes,
            width=self.width,
            pad_prop=self.pad_prop,
            pad_mode=self.pad_mode,
            nb_layers=self.nb_layers,
            activation=self.activation,
        ).to(self.device)
        print(f"{count_params(self.model)=}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_lr)
        self.scheduler = LR_Scheduler(
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
        return loss_v

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
                relative=self.relative,
                squared=False,
                plot=self.nb_train_epochs % 500 == 0,
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
                relative=self.relative,
                squared=False,
                plot=self.nb_train_epochs % 500 == 0,
            )
        if self.level == 2:
            self.losses_dict = [
                {
                    "epoch": self.nb_train_epochs,
                    "loss_T": f"{(loss_T.item() ):.2e}",
                    "loss_0_T": f"{(loss_0_T.item() ):.2e}",
                    "loss_1_T": f"{(loss_1_T.item() ):.2e}",
                    "loss_2_T": f"{(loss_2_T.item() ):.2e}",
                    "loss_V": f"{(loss_v.item() ):.2e}",
                    "loss_0_V": f"{(loss_0_v.item() ):.2e}",
                    "loss_1_V": f"{(loss_1_v.item() ):.2e}",
                    "loss_2_V": f"{(loss_2_v.item() ):.2e}",
                }
            ]
        elif self.level == 1 or self.level == 0.5:
            self.losses_dict = [
                {
                    "epoch": self.nb_train_epochs,
                    "loss_T": f"{(loss_T.item() ):.2e}",
                    "rel-L2_T": f"{(loss_0_T.item() ):.2e}",
                    "rel-h1_T": f"{(loss_1_T.item() ):.2e}",
                    "loss_V": f"{(loss_v.item() ):.2e}",
                    "rel-L2_V": f"{(loss_0_v.item() ):.2e}",
                    "rel-h1_V": f"{(loss_1_v.item() ):.2e}",
                }
            ]
        else:
            self.losses_dict = [
                {
                    "epoch": self.nb_train_epochs,
                    "loss_T": f"{(loss_T.item() ):.2e}",
                    "loss_0_T": f"{(loss_0_T.item() ):.2e}",
                    "loss_V": f"{(loss_v.item() ):.2e}",
                    "loss_0_V": f"{(loss_0_v.item() ):.2e}",
                }
            ]

        self.losses_array.append(
            [
                self.nb_train_epochs,
                loss_T.item(),
                loss_0_T.item(),
                loss_1_T.item(),
                loss_2_T.item(),
                loss_v.item(),
                loss_0_v.item(),
                loss_1_v.item(),
                loss_2_v.item(),
            ]
        )
        return loss_v.item()

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
            np.save(f"{models_repo}/losses_array.npy", np.array(self.losses_array))

    def plot_losses(self, models_repo="./models", save=True):
        losses_array = np.load(f"{models_repo}/losses_array.npy")

        epochs = losses_array[:, 0]
        list_loss_train = losses_array[:, 1]
        list_loss_0_train = losses_array[:, 2]
        list_loss_val = losses_array[:, 5]
        list_loss_0_val = losses_array[:, 6]

        file = open(f"{models_repo}/best_model/best_epoch.txt")
        for y in file.read().split(" "):
            if y.isdigit():
                best_epoch = int(y)
        best_epoch_index = list(epochs).index(best_epoch)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        indices_to_plot = list(range(0, len(epochs), 5))

        ax0.plot(
            epochs[indices_to_plot],
            list_loss_train[indices_to_plot],
            label=r"$\mathcal{L}(train)$",
        )

        ax0.plot(
            best_epoch,
            list_loss_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )
        ax0.plot(
            epochs[indices_to_plot],
            list_loss_val[indices_to_plot],
            label=r"$\mathcal{L}(validation)$",
        )

        ax0.plot(
            best_epoch,
            list_loss_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )

        ax0.legend(ncol=1, fontsize=16, loc="upper right")
        ax0.set_xlabel("Epochs", fontsize=16)
        ax0.set_yscale("log")
        ax0.set_title(
            r"Evolution of $\mathcal{L}$ on the training" + "\nand validation sets",
            fontsize=16,
        )

        ax1.plot(
            epochs[indices_to_plot],
            list_loss_0_train[indices_to_plot],
            label=r"$\mathcal{L}_0(train)$",
        )
        ax1.plot(
            best_epoch,
            list_loss_0_train[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )

        ax1.plot(
            epochs[indices_to_plot],
            list_loss_0_val[indices_to_plot],
            label=r"$\mathcal{L}_0(validation)$",
        )

        ax1.plot(
            best_epoch,
            list_loss_0_val[best_epoch_index],
            marker="x",
            color="k",
            markersize=10,
        )

        ax1.set_yscale("log")
        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_title(
            r"Evolution of the relative $L^2$ error"
            + "\non the training and validation sets",
            fontsize=16,
        )
        ax1.legend(ncol=1, fontsize=16, loc="upper right")

        plt.tight_layout()
        if save:
            plt.savefig(f"./{models_repo}/losses.pdf")
        plt.show()


if __name__ == "__main__":
    test_DataLoader()
