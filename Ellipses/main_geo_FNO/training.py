import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities import *
from Adam import Adam
import torch
import numpy as np
import torch.nn as nn
import time
import seaborn as sns
import os
from scheduler import LR_Scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


torch.backends.cudnn.deterministic = True
set_seed(0)
set_seed(0)
sns.set_theme()
sns.set_context("paper")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


class DataLoader:
    def __init__(self, small_data=False, dtype=torch.float32):
        PATH_Sigma = "./data/U_geo_fno.npy"
        PATH_XY = "./data/XY_geo_fno.npy"

        input_s = np.load(PATH_Sigma)[:, :, None]
        input_xy = np.load(PATH_XY)

        input_f = np.load("./data/F.npy")[:, :, None]
        input_g = np.load("./data/G.npy")[:, :, None]

        input_s = torch.tensor(input_s, dtype=torch.float)
        input_xy = torch.tensor(input_xy, dtype=torch.float)
        input_f = torch.tensor(input_f, dtype=torch.float)
        input_g = torch.tensor(input_g, dtype=torch.float)

        print(f"{input_s.shape=}  {input_xy.shape=} {input_f.shape=} {input_g.shape=}")

        input_xy = torch.cat([input_xy, input_f, input_g], dim=-1)
        print(f"{input_xy.shape=}")

        ntrain = 1500
        ntest = 300

        def separe(A):
            return A[:ntrain], A[-ntest:]

        self.X_train, self.X_val = separe(input_xy)
        self.Y_train, self.Y_val = separe(input_s)

        print(
            f"{self.X_train.shape=}  {self.X_val.shape=} {self.Y_train.shape=} {self.Y_val.shape=}"
        )
        self.nb_vert = self.X_train.shape[1]
        self.nb_train, self.nb_val = ntrain, ntest


def test_DataLoader():
    data = DataLoader()
    print("X_train", data.X_train.shape)
    print("X_val", data.X_val.shape)
    print("Y_train", data.Y_train.shape)
    print("Y_val", data.Y_val.shape)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2
        factor1 = self.compl_mul2d(
            u_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        factor2 = self.compl_mul2d(
            u_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        if x_out == None:
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                s1,
                s2 // 2 + 1,
                dtype=torch.cfloat,
                device=u.device,
            )
            out_ft[:, :, : self.modes1, : self.modes2] = factor1
            out_ft[:, :, -self.modes1 :, : self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-(self.modes1), end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        K1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K = K1 + K2
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-(self.modes1), end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        K1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K = K1 + K2
        basis = torch.exp(1j * 2 * np.pi * K).to(device)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class FNO2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        in_channels,
        out_channels,
        is_mesh=True,
        s1=64,
        s2=64,
    ):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, s1, s2
        )
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, s1, s2
        )
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        if self.is_mesh and x_in == None:
            x_in = u[:, :, :2]
        if self.is_mesh and x_out == None:
            x_out = u[:, :, :2]
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(
            0, 3, 1, 2
        )

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()
        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3 * self.width, 4 * self.width)
        self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 2)
        self.center = torch.tensor([0.5, 0.5], device="cuda").reshape(1, 1, 2)

        self.B = np.pi * torch.pow(
            2, torch.arange(0, self.width // 4, dtype=torch.float, device="cuda")
        ).reshape(1, 1, 1, self.width // 4)

    def forward(self, x, code=None):
        angle = torch.atan2(
            x[:, :, 1] - self.center[:, :, 1], x[:, :, 0] - self.center[:, :, 0]
        )
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:, :, 0], x[:, :, 1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        x_cos = torch.cos(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b, n, 3 * self.width)

        if code != None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1, xd.shape[1], 1)
            xd = torch.cat([cd, xd], dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = F.gelu(xd)
        xd = self.fc2(xd)
        xd = F.gelu(xd)
        xd = self.fc3(xd)
        return x + x * xd


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


class Agent:
    def __init__(self, data):
        self.data = data
        self.initial_lr = 0.005
        self.initial_lr_iphi = 0.001
        self.n_modes = 10
        self.width = 20
        self.batch_size = 32

        self.X_train = self.data.X_train
        self.Y_train = self.data.Y_train
        self.X_val = self.data.X_val
        self.Y_val = self.data.Y_val

        nb_data_train = self.X_train.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not (self.device == self.Y_train.device):
            self.X_train = self.data.X_train.to(self.device)
            self.Y_train = self.data.Y_train.to(self.device)
            self.X_val = self.data.X_val.to(self.device)
            self.Y_val = self.data.Y_val.to(self.device)

        self.nb_train_data = self.X_train.shape[0]
        self.nb_val_data = self.X_val.shape[0]

        self.model = FNO2d(
            self.n_modes,
            self.n_modes,
            self.width,
            in_channels=4,
            out_channels=1,
        ).to(self.device)

        self.optimizer_fno = Adam(
            self.model.parameters(), lr=self.initial_lr, weight_decay=5e-3
        )
        # self.scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer_fno, T_max=200
        # )
        self.scheduler_fno = LR_Scheduler(
            self.optimizer_fno,
            patience=20,
            factor=0.7,
            cooldown=5,
            min_lr=1e-5,
        )
        self.model_iphi = IPHI().to(self.device)
        self.optimizer_iphi = Adam(
            self.model_iphi.parameters(), lr=self.initial_lr_iphi, weight_decay=1e-3
        )

        self.scheduler_iphi = LR_Scheduler(
            self.optimizer_iphi,
            patience=20,
            factor=0.7,
            cooldown=5,
            min_lr=1e-5,
        )

        print(f"{count_params(self.model)=}, {count_params(self.model_iphi)=}")

        self.loss_function = LpLoss(size_average=False)
        self.nb_batch = nb_data_train // self.batch_size
        self.test_batch_size = self.X_val.shape[0]
        self.losses = []
        self.losses_dict = []
        self.losses_array = []
        self.nb_train_epochs = 0
        self.nweights = 0

    def train_one_epoch(self):
        self.nb_train_epochs += 1
        X_train, Y_train = self.X_train, self.Y_train
        rand_i = torch.randperm(X_train.shape[0])

        X = X_train[rand_i]
        Y = Y_train[rand_i]
        self.model.train()
        self.model_iphi.train()
        loss_i = 0.0
        for i in range(self.nb_batch):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x, y_true = X[sli], Y[sli]
            self.optimizer_fno.zero_grad()
            self.optimizer_iphi.zero_grad()
            y_pred = self.model(x, iphi=self.model_iphi)
            loss = self.loss_function(
                y_pred.view(self.batch_size, -1), y_true.view(self.batch_size, -1)
            )
            loss.backward()
            self.optimizer_fno.step()
            self.optimizer_iphi.step()
            loss_i += loss.item() / self.batch_size
        self.losses.append(loss_i / self.nb_batch)

    def validate(self):
        # x, y_true = self.X_val, self.Y_val
        # self.model.eval()
        # with torch.no_grad():
        #     y_pred = self.model(x, iphi=self.model_iphi)
        #     loss_v = self.loss_function(
        #         y_pred.view(self.test_batch_size, -1),
        #         y_true.view(self.test_batch_size, -1),
        #     )
        # return loss_v / self.test_batch_size
        loss_i = 0.0
        X, Y = self.X_val, self.Y_val
        nb_val_data = X.shape[0]
        test_batch_size = 100
        nb_test_batches = 3
        self.model.eval()
        self.model_iphi.eval()
        for i in range(nb_test_batches):
            with torch.no_grad():
                sli = slice(i * test_batch_size, (i + 1) * test_batch_size)
                x, y_true = X[sli], Y[sli]
                y_pred = self.model(x, iphi=self.model_iphi)
                loss = self.loss_function(
                    y_pred.view(test_batch_size, -1), y_true.view(test_batch_size, -1)
                )
                loss_i += loss.item() / test_batch_size
        return loss_i / nb_test_batches

    def eval_losses(self):
        self.model.eval()
        with torch.no_grad():
            # training part
            test_batch_size = 100
            nb_test_batches = 3
            self.model.eval()
            self.model_iphi.eval()
            indices = torch.randperm(self.X_train.shape[0])[: self.test_batch_size]
            X, Y = self.X_train[indices], self.Y_train[indices]
            loss_T = torch.tensor(0.0).to(X.device)
            for i in range(nb_test_batches):
                sli = slice(i * test_batch_size, (i + 1) * test_batch_size)
                x, y_true = X[sli], Y[sli]
                y_pred = self.model(x, iphi=self.model_iphi)
                loss_T_i = self.loss_function(
                    y_pred.view(test_batch_size, -1),
                    y_true.view(test_batch_size, -1),
                )
                loss_T += loss_T_i
            loss_0_T, loss_1_T, loss_2_T = (
                loss_T,
                torch.tensor(torch.nan),
                torch.tensor(torch.nan),
            )

            # validation part
            X, Y = self.X_val, self.Y_val
            loss_V = torch.tensor(0.0).to(X.device)
            for i in range(nb_test_batches):
                sli = slice(i * test_batch_size, (i + 1) * test_batch_size)
                x, y_true = X[sli], Y[sli]
                y_pred = self.model(x, iphi=self.model_iphi)
                loss_V_i = self.loss_function(
                    y_pred.view(test_batch_size, -1),
                    y_true.view(test_batch_size, -1),
                )
                loss_V += loss_V_i
            loss_0_V, loss_1_V, loss_2_V = (
                loss_V,
                torch.tensor(torch.nan),
                torch.tensor(torch.nan),
            )
        self.losses_dict.append(
            {
                "epoch": self.nb_train_epochs,
                "loss_T": f"{(loss_T.item() / self.test_batch_size):.2f}",
                "loss_0_T": f"{(loss_0_T.item() / self.test_batch_size):.2f}",
                "loss_1_T": f"{(loss_1_T.item() / self.test_batch_size):.2f}",
                "loss_2_T": f"{(loss_2_T.item() / self.test_batch_size):.2f}",
                "loss_V": f"{(loss_V.item() / self.test_batch_size):.2f}",
                "loss_0_V": f"{(loss_0_V.item() / self.test_batch_size):.2f}",
                "loss_1_V": f"{(loss_1_V.item() / self.test_batch_size):.2f}",
                "loss_2_V": f"{(loss_2_V.item() / self.test_batch_size):.2f}",
            }
        )
        self.losses_array.append(
            [
                self.nb_train_epochs,
                loss_T.item() / self.test_batch_size,
                loss_0_T.item() / self.test_batch_size,
                loss_1_T.item() / self.test_batch_size,
                loss_2_T.item() / self.test_batch_size,
                loss_V.item() / self.test_batch_size,
                loss_0_V.item() / self.test_batch_size,
                loss_1_V.item() / self.test_batch_size,
                loss_2_V.item() / self.test_batch_size,
            ]
        )
        if self.nb_train_epochs % 50 == 0 and self.nb_train_epochs > 1:
            x, y_true = (
                self.X_val.detach().cpu().numpy()[-1],
                self.Y_val.squeeze().detach().cpu().numpy()[-1],
            )
            pred = y_pred[-1].squeeze().detach().cpu().numpy()
            lims = dict(cmap="RdBu_r", vmin=y_true.min(), vmax=y_true.max())

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1)
            plt.scatter(x[:, 0], x[:, 1], c=y_true, edgecolor="w", **lims)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)
            plt.title("Truth")
            plt.subplot(2, 3, 2)
            plt.scatter(x[:, 0], x[:, 1], c=pred, edgecolor="w", **lims)
            plt.title("Pred")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)
            plt.subplot(2, 3, 3)
            plt.scatter(x[:, 0], x[:, 1], c=y_true - pred, edgecolor="w", cmap="RdBu_r")
            plt.title("Diff")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)

            x, y_true = (
                self.X_val.detach().cpu().numpy()[-5],
                self.Y_val.squeeze().detach().cpu().numpy()[-5],
            )
            pred = y_pred[-5].squeeze().detach().cpu().numpy()
            lims = dict(cmap="RdBu_r", vmin=y_true.min(), vmax=y_true.max())

            plt.subplot(2, 3, 4)
            plt.scatter(x[:, 0], x[:, 1], c=y_true, edgecolor="w", **lims)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)
            plt.title("Truth")
            plt.subplot(2, 3, 5)
            plt.scatter(x[:, 0], x[:, 1], c=pred, edgecolor="w", **lims)
            plt.title("Pred")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)
            plt.subplot(2, 3, 6)
            plt.scatter(x[:, 0], x[:, 1], c=y_true - pred, edgecolor="w", cmap="RdBu_r")
            plt.title("Diff")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.colorbar(shrink=0.6)
            if not os.path.exists("./outputs"):
                os.makedirs("./outputs")
            plt.savefig(f"./outputs/output_{self.nb_train_epochs}.png")
            if self.nb_train_epochs == 2000 or self.nb_train_epochs == 50:
                plt.show()
        return loss_V.item() / self.test_batch_size

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
            start = time.time()
            self.train_one_epoch()
            validation_loss = self.validate()
            end = time.time()
            self.scheduler_fno.step(validation_loss)
            if self.scheduler_fno.lr_has_changed:
                if self.scheduler_fno.patience <= 60:
                    self.scheduler_fno.patience = int(self.scheduler_fno.patience * 1.5)
                print(
                    f"{self.scheduler_fno.patience = } {self.scheduler_fno._last_lr[-1] = :.3e}"
                )
            self.scheduler_iphi.step(validation_loss)
            if self.scheduler_iphi.lr_has_changed:
                if self.scheduler_iphi.patience <= 60:
                    self.scheduler_iphi.patience = int(
                        self.scheduler_iphi.patience * 1.5
                    )
                print(
                    f"{self.scheduler_iphi.patience = } {self.scheduler_iphi._last_lr[-1] = :.3e}"
                )

            if epoch % 1 == 0:
                mean_loss_validation = self.eval_losses()
                if mean_loss_validation <= best_mean_loss_validation:
                    best_mean_loss_validation = mean_loss_validation

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_fno_state_dict": self.optimizer_fno.state_dict(),
                            "model_iphi_state_dict": self.model_iphi.state_dict(),
                            "optimizer_iphi_state_dict": self.optimizer_iphi.state_dict(),
                        },
                        f"{models_repo}/best_model.pkl",
                    )

                    file = open(
                        f"{models_repo}/best_model/best_epoch.txt",
                        "w",
                    )

                    file.write(f"best_epoch = {epoch} \n")
                    file.write(f"loss = {best_mean_loss_validation} \n")
                    file.write(f"loss = {best_mean_loss_validation:.2e} \n")
                    file.write(f"{self.losses_dict[-1]} \n")
                    file.close()
                print(f"{self.losses_dict[-1]}")
            print(f"Time: {end-start:.3f}s")
            if epoch % 10 == 0:
                if save_models:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_fno_state_dict": self.optimizer_fno.state_dict(),
                            "model_iphi_state_dict": self.model_iphi.state_dict(),
                            "optimizer_iphi_state_dict": self.optimizer_iphi.state_dict(),
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

        # if not (os.path.exists("./images")) and save:
        #     os.makedirs("./images")
        if save:
            # plt.savefig(f"./images/losses.png")
            plt.savefig(f"./{models_repo}/losses.png")
        plt.show()


if __name__ == "__main__":
    data = DataLoader(False)

    training_agent = Agent(data)
    # print(
    #     f"(level, relative, squared, initial_lr, n_modes, width, batch_size, l2_lambda, pad_prop, pad_mode) = "
    #     + f"{(training_agent.level, training_agent.relative, training_agent.squared, training_agent.initial_lr, training_agent.n_modes, training_agent.width, training_agent.batch_size, training_agent.l2_lambda, training_agent.pad_prop, training_agent.pad_mode)} \n"
    # )
    nb_epochs = 2000
    start_training = time.time()
    training_agent.train(nb_epochs, models_repo="./models")
    end_training = time.time()
    time_training = end_training - start_training
    print(
        f"Total time to train the operator : {time_training:.3f}s. Average time : {time_training/nb_epochs:.3f}s."
    )
    training_agent.plot_losses(models_repo="./models")  # models_repo=repos[i])
