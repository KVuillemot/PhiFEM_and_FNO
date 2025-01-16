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
        norm = torch.amax(torch.abs(U * domain[:, None, :, :]), (1, 2, 3))
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

    def plot(self, U_true, U_pred, domain):
        x = np.linspace(0, 1, domain.shape[-2])
        y = np.linspace(0, 1, domain.shape[-1])
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 3, 1)
        plt.pcolormesh(
            x,
            y,
            (U_true * domain[:, None, :, :]).cpu().detach().numpy()[0, 0, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{true,x}$")
        plt.gca().set_aspect("equal")
        plt.colorbar(shrink=0.6)
        plt.subplot(2, 3, 2)
        plt.pcolormesh(
            x,
            y,
            (U_pred * domain[:, None, :, :]).cpu().detach().numpy()[0, 0, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{pred,x}$")
        plt.colorbar(shrink=0.6)
        plt.gca().set_aspect("equal")
        plt.subplot(2, 3, 3)
        plt.pcolormesh(
            x,
            y,
            torch.absolute((U_pred - U_true) * domain[:, None, :, :])
            .cpu()
            .detach()
            .numpy()[0, 0, :, :]
            .T,
            cmap="viridis",
        )
        plt.title(r"$|U_{true,x} - U_{pred,x}|$")
        plt.gca().set_aspect("equal")
        plt.colorbar(shrink=0.6)
        plt.subplot(2, 3, 4)
        plt.pcolormesh(
            x,
            y,
            (U_true * domain[:, None, :, :]).cpu().detach().numpy()[0, 1, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{true,y}$")
        plt.colorbar(shrink=0.6)
        plt.gca().set_aspect("equal")
        plt.subplot(2, 3, 5)
        plt.pcolormesh(
            x,
            y,
            (U_pred * domain[:, None, :, :]).cpu().detach().numpy()[0, 1, :, :].T,
            cmap="viridis",
        )
        plt.title(r"$U_{pred,y}$")
        plt.colorbar(shrink=0.6)
        plt.gca().set_aspect("equal")
        plt.subplot(2, 3, 6)
        plt.pcolormesh(
            x,
            y,
            (torch.absolute((U_pred - U_true) * domain[:, None, :, :]))
            .cpu()
            .detach()
            .numpy()[0, 1, :, :]
            .T,
            cmap="viridis",
        )
        plt.title(r"$|U_{true,y} - U_{pred,y}|$")
        plt.colorbar(shrink=0.6)
        plt.gca().set_aspect("equal")
        plt.show()

    def compute_loss(
        self, X, y_pred, y_true, mode, level, relative, squared, plot=False
    ):
        assert mode == "train" or mode == "val" or mode == "eval_performances"
        assert level == 0 or level == 1 or level == 2 or level == 0.5
        assert relative == True or relative == False
        assert squared == True or squared == False

        if mode == "train":

            Y_true, Y_pred = y_true, y_pred
            U_true = Y_true
            U_pred = Y_pred
            Phi = X[:, 0, :, :]

            domain, domain_1 = self.compute_boundaries(Phi, 1)
            derivator_level1 = Derivator_fd(
                axes=[1, 2], interval_lengths=[1, 1], derivative_symbols=[(0,), (1,)]
            )
            (U_x_true_x, U_x_true_y) = derivator_level1(U_true[:, 0, :, :])
            (U_x_pred_x, U_x_pred_y) = derivator_level1(U_pred[:, 0, :, :])
            (U_y_true_x, U_y_true_y) = derivator_level1(U_true[:, 1, :, :])
            (U_y_pred_x, U_y_pred_y) = derivator_level1(U_pred[:, 1, :, :])

            U_true_x = torch.stack([U_x_true_x, U_y_true_x], dim=1)
            U_true_y = torch.stack([U_x_true_y, U_y_true_y], dim=1)
            U_pred_x = torch.stack([U_x_pred_x, U_y_pred_x], dim=1)
            U_pred_y = torch.stack([U_x_pred_y, U_y_pred_y], dim=1)

            error_1_x_x = self.compute_L2_norm_squared(
                (U_pred_x[:, 0, :, :] - U_true_x[:, 0, :, :]), domain_1
            )
            error_1_x_y = self.compute_L2_norm_squared(
                (U_pred_x[:, 1, :, :] - U_true_x[:, 1, :, :]), domain_1
            )
            error_1_y_x = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 0, :, :], domain_1
            )
            error_1_y_y = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 1, :, :], domain_1
            )
            loss = torch.mean(error_1_x_x + error_1_x_y + error_1_y_x + error_1_y_y)

            return loss

        elif mode == "val":
            Y_true, Y_pred = y_true, y_pred
            Phi = X[:, 0, :, :]
            U_true = Y_true

            domain, domain_1 = self.compute_boundaries(Phi, 1)
            domains_tmp = domain.cpu().detach().numpy()  # .flatten()
            domains_nan = (
                domain.cpu().detach().numpy().copy().astype(float)
            )  # .flatten().astype(float)
            domains_nan[np.where(domains_tmp == False)] = np.nan
            # domains_nan = np.reshape(domains_nan, domain.shape)
            domains_nan = torch.tensor(domains_nan).to(Phi.device)

            avg_low_displacement = torch.nanmean(
                (Y_pred * domains_nan[:, None, :, :])[:, :, :, 0], dim=2
            )[:, :, None, None]

            U_pred = Y_pred - avg_low_displacement
            if plot:
                self.plot(U_true, U_pred, domains_nan)

            derivator_level1 = Derivator_fd(
                axes=[1, 2], interval_lengths=[1, 1], derivative_symbols=[(0,), (1,)]
            )
            (U_x_true_x, U_x_true_y) = derivator_level1(U_true[:, 0, :, :])
            (U_x_pred_x, U_x_pred_y) = derivator_level1(U_pred[:, 0, :, :])
            (U_y_true_x, U_y_true_y) = derivator_level1(U_true[:, 1, :, :])
            (U_y_pred_x, U_y_pred_y) = derivator_level1(U_pred[:, 1, :, :])

            U_true_x = torch.stack([U_x_true_x, U_y_true_x], dim=1)
            U_true_y = torch.stack([U_x_true_y, U_y_true_y], dim=1)
            U_pred_x = torch.stack([U_x_pred_x, U_y_pred_x], dim=1)
            U_pred_y = torch.stack([U_x_pred_y, U_y_pred_y], dim=1)

            error_0_x = self.compute_L2_norm_squared(
                (U_pred - U_true)[:, 0, :, :], domain
            )
            error_0_y = self.compute_L2_norm_squared(
                (U_pred - U_true)[:, 1, :, :], domain
            )
            error_1_x_x = self.compute_L2_norm_squared(
                (U_pred_x[:, 0, :, :] - U_true_x[:, 0, :, :]), domain_1
            )
            error_1_x_y = self.compute_L2_norm_squared(
                (U_pred_x[:, 1, :, :] - U_true_x[:, 1, :, :]), domain_1
            )
            error_1_y_x = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 0, :, :], domain_1
            )
            error_1_y_y = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 1, :, :], domain_1
            )
            magnitude_0_x = self.compute_L2_norm_squared((U_true)[:, 0, :, :], domain)
            magnitude_0_y = self.compute_L2_norm_squared((U_true)[:, 1, :, :], domain)

            loss_0 = torch.mean(
                torch.sqrt(((error_0_x + error_0_y) / (magnitude_0_x + magnitude_0_y)))
            )

            magnitude_inf = self.compute_Linf_norm((U_true)[:, :, :, :], domain)
            loss_1 = torch.mean(torch.sqrt(((error_0_x + error_0_y))) / magnitude_inf)
            loss = torch.mean(error_1_x_x + error_1_x_y + error_1_y_x + error_1_y_y)

            return loss, loss_0, loss_1, torch.tensor(torch.nan)

        else:
            Y_true, Y_pred = y_true, y_pred
            Phi = X[:, 0, :, :]
            U_true = Y_true
            U_pred = Y_pred
            # avg_low_displacement = torch.mean(U_pred[:, :, 0, :], dim=2)[
            #     :, :, None, None
            # ]
            # U_pred = Y_pred - avg_low_displacement

            domain, domain_1 = self.compute_boundaries(Phi, 1)
            if plot:
                self.plot(U_true, U_pred, domain)

            derivator_level1 = Derivator_fd(
                axes=[1, 2], interval_lengths=[1, 1], derivative_symbols=[(0,), (1,)]
            )
            (U_x_true_x, U_x_true_y) = derivator_level1(U_true[:, 0, :, :])
            (U_x_pred_x, U_x_pred_y) = derivator_level1(U_pred[:, 0, :, :])
            (U_y_true_x, U_y_true_y) = derivator_level1(U_true[:, 1, :, :])
            (U_y_pred_x, U_y_pred_y) = derivator_level1(U_pred[:, 1, :, :])

            U_true_x = torch.stack([U_x_true_x, U_y_true_x], dim=1)
            U_true_y = torch.stack([U_x_true_y, U_y_true_y], dim=1)
            U_pred_x = torch.stack([U_x_pred_x, U_y_pred_x], dim=1)
            U_pred_y = torch.stack([U_x_pred_y, U_y_pred_y], dim=1)

            error_0_x = self.compute_L2_norm_squared(
                (U_pred - U_true)[:, 0, :, :], domain
            )
            error_0_y = self.compute_L2_norm_squared(
                (U_pred - U_true)[:, 1, :, :], domain
            )
            error_1_x_x = self.compute_L2_norm_squared(
                (U_pred_x[:, 0, :, :] - U_true_x[:, 0, :, :]), domain_1
            )
            error_1_x_y = self.compute_L2_norm_squared(
                (U_pred_x[:, 1, :, :] - U_true_x[:, 1, :, :]), domain_1
            )
            error_1_y_x = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 0, :, :], domain_1
            )
            error_1_y_y = self.compute_L2_norm_squared(
                (U_pred_y - U_true_y)[:, 1, :, :], domain_1
            )
            magnitude_0_x = self.compute_L2_norm_squared((U_true)[:, 0, :, :], domain)
            magnitude_0_y = self.compute_L2_norm_squared((U_true)[:, 1, :, :], domain)
            magnitude_1_x_x = self.compute_L2_norm_squared(
                (U_true_x[:, 0, :, :]), domain_1
            )
            magnitude_1_x_y = self.compute_L2_norm_squared(
                (U_true_x[:, 1, :, :]), domain_1
            )
            magnitude_1_y_x = self.compute_L2_norm_squared(
                (U_true_y)[:, 0, :, :], domain_1
            )
            magnitude_1_y_y = self.compute_L2_norm_squared(
                (U_true_y)[:, 1, :, :], domain_1
            )
            loss_0 = torch.sqrt(
                ((error_0_x + error_0_y) / (magnitude_0_x + magnitude_0_y))
            )
            loss_1 = torch.sqrt(
                (error_1_x_x + error_1_x_y + error_1_y_x + error_1_y_y)
                / (
                    magnitude_1_x_x
                    + magnitude_1_x_y
                    + magnitude_1_y_x
                    + magnitude_1_y_y
                )
            )
            loss = (
                error_0_x
                + error_0_y
                + error_1_x_x
                + error_1_x_y
                + error_1_y_x
                + error_1_y_y
            )

            return loss, loss_0, loss_1, torch.tensor(torch.nan)

    def __call__(
        self, X, y_pred, y_true, mode, level=1, relative=True, squared=True, plot=False
    ):
        return self.compute_loss(
            X, y_pred, y_true, mode, level, relative, squared, plot
        )
