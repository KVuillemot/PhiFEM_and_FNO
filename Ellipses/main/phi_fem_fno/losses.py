import torch
import numpy as np
from utils import *
from prepare_data import *


class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def compute_L2_norm_squared(self, U, domain):
        nb_vertices = torch.sum(domain, (1, 2), False)
        norm = (1.0 / nb_vertices) * torch.sum(U**2 * domain, (1, 2), False)
        return norm

    def compute_L1_norm(self, U, domain):
        nb_vertices = torch.sum(domain, (1, 2), False)
        norm = (1.0 / nb_vertices) * torch.sum(U * domain, (1, 2), False)
        return norm

    def compute_Linf_norm(self, U, domain):
        norm = torch.amax(torch.abs(U * domain), (1, 2))
        return norm

    def border_torch(self, domains):
        """
        Determine the pixels on the boundary of multiple domains using PyTorch.

        Parameters:
            domains (tensor): Tensor defining the domains (0 outside, 1 inside).

        Returns:
            tensor: Tensor with 1 at boundary pixels, 0 otherwise.
        """
        pad_domains = torch.nn.functional.pad(
            domains, pad=(1, 1, 1, 1), mode="constant", value=0
        )

        diff = (
            (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, :-2, 1:-1])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 2:, 1:-1])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 1:-1, :-2])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 1:-1, 2:])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, :-2, :-2])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, :-2, 2:])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 2:, :-2])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 2:, 2:])
        )

        res = diff.int()
        domain_tmp = torch.where(domains == 0, torch.tensor(10), torch.tensor(1))
        res -= domain_tmp
        return (res == 0).int()

    def neighborhood_6(self, domains):
        """
        Determine the pixels on the boundary of multiple domains using PyTorch.
        Using the 6-th neighborhood
        Parameters:
            domains (tensor): Tensor defining the domains (0 outside, 1 inside).

        Returns:
            tensor: Tensor with 1 at boundary pixels, 0 otherwise.
        """
        pad_domains = torch.nn.functional.pad(
            domains, pad=(1, 1, 1, 1), mode="constant", value=0
        )

        diff = (
            (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, :-2, 1:-1])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 2:, 1:-1])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 1:-1, :-2])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 1:-1, 2:])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, :-2, :-2])
            | (pad_domains[:, 1:-1, 1:-1] != pad_domains[:, 2:, 2:])
        )

        res = diff.int()
        domain_tmp = torch.where(domains == 1, torch.tensor(10), torch.tensor(1))
        res -= domain_tmp
        return (res == 0).int()

    def plot(self, U_true, U_pred, domain):
        x = np.linspace(0, 1, domain.shape[-2])
        y = np.linspace(0, 1, domain.shape[-1])
        plt.figure(figsize=(12, 9))
        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            x,
            y,
            (U_true * domain).cpu().detach().numpy()[0, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{true}$")
        plt.gca().set_aspect("equal")
        plt.colorbar(shrink=0.6)
        plt.subplot(1, 3, 2)
        plt.pcolormesh(
            x,
            y,
            (U_pred * domain).cpu().detach().numpy()[0, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{pred}$")
        plt.colorbar(shrink=0.6)
        plt.gca().set_aspect("equal")
        plt.subplot(1, 3, 3)
        plt.pcolormesh(
            x,
            y,
            torch.absolute((U_pred - U_true) * domain)
            .cpu()
            .detach()
            .numpy()[0, :, :]
            .T,
            cmap="viridis",
        )
        plt.title(r"$|U_{true} - U_{pred}|$")
        plt.gca().set_aspect("equal")
        plt.colorbar(shrink=0.6)
        plt.show()

    def compute_boundaries(self, phi, level):
        domain_tmp = (phi <= 3e-16).to(phi.device)
        neighborhood = self.neighborhood_6(domain_tmp).to(phi.device)
        domain = ((neighborhood.int() + domain_tmp.int()) != 0).to(phi.device)
        if level == 0:
            return domain
        else:
            boundary = self.border_torch(domain).to(phi.device)
            domain_1 = ((domain.int() + boundary.int()) == 1).to(phi.device)
            domain_1 = ((domain_1.int() == 1) & (domain.int() == 1)).to(phi.device)
            if level == 1:
                return domain, domain_1
            else:
                boundary_1 = self.border_torch(domain_1).to(phi.device)
                domain_2 = ((domain_1.int() + boundary_1.int()) == 1).to(phi.device)
                domain_2 = ((domain_2.int() == 1) & (domain_1.int() == 1)).to(
                    phi.device
                )
                return domain, domain_1, domain_2

    def compute_loss(
        self, X, y_pred, y_true, mode, level, relative, squared, plot=False
    ):
        assert mode == "train" or mode == "val" or mode == "eval_performances"
        assert level == 0 or level == 1 or level == 2
        assert relative == True or relative == False
        assert squared == True or squared == False

        magnitude = magnitude_0 = magnitude_1 = magnitude_2 = 1.0

        if level == 0:
            Y_true, Y_pred = y_true, y_pred
            Phi, G = X[:, 1, :, :], X[:, 2, :, :]

            U_true = Y_true[:, 0, :, :] * Phi + G
            U_pred = Y_pred[:, 0, :, :] * Phi + G
            domain = self.compute_boundaries(Phi, level)
            error_0 = self.compute_L2_norm_squared((U_pred - U_true), domain)

            if mode == "train":
                loss = torch.mean(error_0)
                return loss

            elif mode == "val":
                domain = self.compute_boundaries(Phi, 0)
                domains_tmp = domain.cpu().detach().numpy().flatten()
                domains_nan = (
                    domain.cpu().detach().numpy().copy().flatten().astype(float)
                )
                domains_nan[np.where(domains_tmp == False)] = np.nan
                domains_nan = np.reshape(domains_nan, domain.shape)
                domains_nan = torch.tensor(domains_nan).to(Phi.device)
                if plot:
                    self.plot(U_true, U_pred, domains_nan)

                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                loss_0 = torch.mean(torch.sqrt(error_0 / magnitude_0))
                loss = torch.mean((error_0))

                return loss, loss_0, torch.tensor(torch.nan), torch.tensor(torch.nan)
            else:
                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                loss_0 = torch.sqrt(error_0 / magnitude_0)
                loss = error_0
                return loss, loss_0, torch.tensor(torch.nan), torch.tensor(torch.nan)
        elif level == 1:
            derivator_level1 = Derivator_fd(
                axes=[1, 2], interval_lengths=[1, 1], derivative_symbols=[(0,), (1,)]
            )

            Y_true, Y_pred = y_true, y_pred
            Phi, G = X[:, 1, :, :], X[:, 2, :, :]

            U_true = Y_true[:, 0, :, :] * Phi + G
            U_pred = Y_pred[:, 0, :, :] * Phi + G
            domain, domain_1 = self.compute_boundaries(Phi, level)
            (U_true_x, U_true_y) = derivator_level1(U_true)
            (U_pred_x, U_pred_y) = derivator_level1(U_pred)

            error_0 = self.compute_L2_norm_squared((U_pred - U_true), domain)
            error_1 = self.compute_L2_norm_squared(
                (U_pred_x - U_true_x), domain_1
            ) + self.compute_L2_norm_squared((U_pred_y - U_true_y), domain_1)

            if mode == "train":
                loss = torch.mean(error_0 + error_1)
                return loss

            elif mode == "val":
                domains_tmp = domain.cpu().detach().numpy().flatten()
                domains_nan = (
                    domain.cpu().detach().numpy().copy().flatten().astype(float)
                )
                domains_nan[np.where(domains_tmp == False)] = np.nan
                domains_nan = np.reshape(domains_nan, domain.shape)
                domains_nan = torch.tensor(domains_nan).to(Phi.device)
                if plot:
                    self.plot(U_true, U_pred, domains_nan)

                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                magnitude_1 = self.compute_L2_norm_squared(
                    (U_true_x), domain_1
                ) + self.compute_L2_norm_squared((U_true_y), domain_1)
                loss_0 = torch.mean(torch.sqrt(error_0 / magnitude_0))
                loss_1 = torch.mean(torch.sqrt(error_1 / magnitude_1))
                loss = torch.mean(error_0 + error_1)
                return loss, loss_0, loss_1, torch.tensor(torch.nan)
            else:
                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                magnitude_1 = self.compute_L2_norm_squared(
                    (U_true_x), domain_1
                ) + self.compute_L2_norm_squared((U_true_y), domain_1)
                loss_0 = torch.sqrt(error_0 / magnitude_0)
                loss_1 = torch.sqrt(error_1 / magnitude_1)
                loss = error_0 + error_1

                return loss, loss_0, loss_1, torch.tensor(torch.nan)

        elif level == 2:
            derivator_level2 = Derivator_fd(
                axes=[1, 2],
                interval_lengths=[1, 1],
                derivative_symbols=[(0,), (1,), (0, 0), (1, 1), (0, 1)],
            )  # {"x":(0,),"y":(1,),"xx":(0,0),"yy":(1,1),"xy":(0,1)}

            Y_true, Y_pred = y_true, y_pred
            Phi, G = X[:, 1, :, :], X[:, 2, :, :]

            U_true = Y_true[:, 0, :, :] * Phi + G
            U_pred = Y_pred[:, 0, :, :] * Phi + G
            domain, domain_1, domain_2 = self.compute_boundaries(Phi, level=level)
            (U_true_x, U_true_y, U_true_xx, U_true_yy, U_true_xy) = derivator_level2(
                U_true
            )
            (U_pred_x, U_pred_y, U_pred_xx, U_pred_yy, U_pred_xy) = derivator_level2(
                U_pred
            )

            error_0 = self.compute_L2_norm_squared((U_pred - U_true), domain)
            error_1 = self.compute_L2_norm_squared(
                (U_pred_x - U_true_x), domain_1
            ) + self.compute_L2_norm_squared((U_pred_y - U_true_y), domain_1)
            error_2 = (
                self.compute_L2_norm_squared((U_pred_xx - U_true_xx), domain_2)
                + self.compute_L2_norm_squared((U_pred_xy - U_true_xy), domain_2)
                + self.compute_L2_norm_squared((U_pred_yy - U_true_yy), domain_2)
            )

            if mode == "train":
                loss = torch.mean(
                    error_0 / magnitude_0
                    + error_1 / magnitude_1
                    + error_2 / magnitude_2
                )
                return loss

            elif mode == "val":
                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                magnitude_1 = self.compute_L2_norm_squared(
                    (U_true_x), domain_1
                ) + self.compute_L2_norm_squared((U_true_y), domain_1)
                magnitude_2 = (
                    self.compute_L2_norm_squared((U_true_xx), domain_2)
                    + self.compute_L2_norm_squared((U_true_xy), domain_2)
                    + self.compute_L2_norm_squared((U_true_yy), domain_2)
                )
                loss_0 = torch.mean(torch.sqrt(error_0 / magnitude_0))
                loss_1 = torch.mean(torch.sqrt(error_1 / magnitude_1))
                loss_2 = torch.mean(torch.sqrt(error_2 / magnitude_2))
                loss = torch.mean(
                    error_0 / magnitude_0
                    + error_1 / magnitude_1
                    + error_2 / magnitude_2
                )

                return loss, loss_0, loss_1, loss_2
            else:
                magnitude_0 = self.compute_L2_norm_squared((U_true), domain)
                magnitude_1 = self.compute_L2_norm_squared(
                    (U_true_x), domain_1
                ) + self.compute_L2_norm_squared((U_true_y), domain_1)
                magnitude_2 = (
                    self.compute_L2_norm_squared((U_true_xx), domain_2)
                    + self.compute_L2_norm_squared((U_true_xy), domain_2)
                    + self.compute_L2_norm_squared((U_true_yy), domain_2)
                )
                loss_0 = torch.sqrt(error_0 / magnitude_0)
                loss_1 = torch.sqrt(error_1 / magnitude_1)
                loss_2 = torch.sqrt(error_2 / magnitude_2)
                loss = (
                    (error_0 / magnitude_0)
                    + (error_1 / magnitude_1)
                    + (error_2 / magnitude_2)
                )

                return loss, loss_0, loss_1, loss_2

    def __call__(
        self, X, y_pred, y_true, mode, level=2, relative=True, squared=True, plot=False
    ):
        return self.compute_loss(
            X, y_pred, y_true, mode, level, relative, squared, plot
        )
