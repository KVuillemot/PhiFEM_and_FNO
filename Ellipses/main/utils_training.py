import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# import dolfin as df
import time
from utils import *
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)

pre_laplacian = Derivator_fd(
    [(0, 0), (1, 1)], interval_lenghts=[1, 1], axes=[1, 2]
)


def laplacian_fn(U):
    """
    Compute the Laplacian of a given tensor U.

    Parameters:
        U (tf.Tensor): Input tensor of shape (batch_size, nx, ny).

    Returns:
        tf.Tensor: The Laplacian of U, with the same shape as U.
    """
    Uxx, Uyy = pre_laplacian(U)
    return Uxx + Uyy


def test_domain_and_border():
    """
    Test the domain and boundary creation function.

    Display two plots: the domain and the boundary.
    """
    phi = tf.constant(np.load("../data/Phi.npy"), dtype=tf.float32)[20]
    domain, boundary = domain_and_border(np.shape(phi)[0], phi)
    plt.figure(figsize=(6, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(domain, cmap=my_cmap, origin="lower")
    plt.subplot(1, 2, 2)
    plt.imshow(boundary, cmap=my_cmap, origin="lower")
    plt.tight_layout()
    plt.show()


def test_data():
    """
    Test the data creation function.

    Display six plots: the domain, the boundary, F, G, W, and U.
    """
    phi = tf.constant(np.load("../data/Phi.npy"), dtype=tf.float32)[10]
    F = tf.constant(np.load("../data/F.npy"), dtype=tf.float32)[10]
    W = tf.constant(np.load("../data/W.npy"), dtype=tf.float32)[10]
    G = tf.constant(np.load("../data/G.npy"), dtype=tf.float32)[10]

    domain, boundary = domain_and_border(np.shape(phi)[0], phi)
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(domain, cmap=my_cmap, origin="lower")
    plt.subplot(2, 3, 2)
    plt.imshow(boundary, cmap=my_cmap, origin="lower")
    plt.subplot(2, 3, 3)
    plt.imshow(F * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("F")
    plt.subplot(2, 3, 4)
    plt.imshow(G * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("G")
    plt.subplot(2, 3, 5)
    plt.imshow(W * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("W")
    plt.subplot(2, 3, 6)
    plt.imshow((W * phi + G) * domain, cmap=my_cmap, origin="lower")
    plt.colorbar(shrink=0.6)
    plt.title("U")
    plt.tight_layout()
    plt.show()


class DataLoader:
    """
    DataLoader class for loading and processing data.
    """

    def __init__(self, small_data=False, dtype=tf.float32):
        """
        Initialize the DataLoader object.

        Parameters:
            small_data (bool, optional): Whether to use a smaller dataset. Defaults to False.
            dtype (tf.DType, optional): Data type to use for loading data. Defaults to tf.float32.
        """
        F = tf.constant(np.load(f"../data/F.npy"), dtype=dtype)
        self.max_norm_F = tf.constant(
            np.max(np.sqrt(np.mean(F**2, axis=(1, 2)))), dtype=dtype
        )
        print("max_norm_F :", self.max_norm_F.numpy())

        F = tf.constant(
            np.load(f"../data/F.npy") / self.max_norm_F,
            dtype=dtype,
        )

        Phi = tf.constant(np.load(f"../data/Phi.npy"), dtype=dtype)
        G = tf.constant(
            np.load(f"../data/G.npy"),
            dtype=dtype,
        )

        Y = tf.constant(np.load(f"../data/W.npy"), dtype=dtype)[:, :, :, None]

        if small_data:
            data_size = 100
        else:
            data_size = F.shape[0]

        if small_data:
            F = F[:data_size]
            Phi = Phi[:data_size]
            G = G[:data_size]
            Y = Y[:data_size]

        nb_vert = F.shape[1]
        domain, boundary = create_domain_and_boundary(nb_vert, Phi)
        domain, boundary = tf.constant(domain, dtype=dtype), tf.constant(
            boundary, dtype=dtype
        )

        X = tf.stack([F, Phi, G, domain], axis=3)
        self.input_shape = (None, X.shape[1], X.shape[2], X.shape[3])

        nb_val = data_size // 8
        nb_train = data_size - nb_val
        print("data_size,nb_val,nb_train:", data_size, nb_val, nb_train)
        print("data shape:", self.input_shape)

        def separe(A):
            return A[:nb_train], A[nb_train:]

        self.X_train, self.X_val = separe(X)
        self.Y_train, self.Y_val = separe(Y)
        self.nb_vert = self.X_train.shape[1]

        residues_interior = self.compute_residues(self.X_val, self.Y_val)
        print("on val data: residues_interior = ", residues_interior.numpy())
        self.residues_interior_val = residues_interior

    def compute_residues(self, X, Y):
        F = X[:, :, :, 0] * self.max_norm_F
        Phi = X[:, :, :, 1]
        G = X[:, :, :, 2]
        W = Y[:, :, :, 0]
        domain = X[:, :, :, -1]
        laplacian = laplacian_fn(Phi * W + G)
        error = tf.reduce_mean(
            ((laplacian + F) ** 2 * domain),
            axis=[1, 2],
        )
        magnitude = tf.reduce_mean(((F) ** 2 * domain), axis=[1, 2])
        residues_interior = tf.sqrt(tf.reduce_mean(error / magnitude))

        return residues_interior


def test_DataLoader():
    """
    Test the DataLoader class.
    """
    data = DataLoader()
    print("X_train", data.X_train.shape)
    print("X_val", data.X_val.shape)
    print("Y_train", data.Y_train.shape)
    print("Y_val", data.Y_val.shape)


def reflexive_padding_2d(W: tf.Tensor, pad_1: int, pad_2: int):
    """
    Apply reflexive padding to a 2D tensor.

    Parameters:
        W (tf.Tensor): Input tensor of shape (batch_size, nx, ny, channels).
        pad_1 (int): Padding size along the first dimension.
        pad_2 (int): Padding size along the second dimension.

    Returns:
        tf.Tensor: The padded tensor.
    """
    return tf.pad(
        W, [[0, 0], [pad_1, pad_1], [pad_2, pad_2], [0, 0]], mode="REFLECT"
    )


class SpectralConv2d(tf.keras.layers.Layer):
    """
    SpectralConv2d layer for a FNO.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialize the SpectralConv2d layer.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes to multiply along the first dimension.
            modes2 (int): Number of Fourier modes to multiply along the second dimension.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N_/2) + 1
        self.modes2 = modes2

        self.weights1 = self.get_complex_weights_2d(in_channels, out_channels)
        self.weights2 = self.get_complex_weights_2d(in_channels, out_channels)

    def get_complex_weights_2d(self, in_channels, out_channels):
        scale = 1 / (in_channels * out_channels)
        real = tf.random.uniform(
            [in_channels, out_channels, self.modes1, self.modes2]
        )
        img = tf.random.uniform(
            [in_channels, out_channels, self.modes1, self.modes2]
        )
        return tf.Variable(tf.complex(real, img) * scale)

    def channel_mix(self, inputs, weights):
        # (batch, in_channel, nx,ny ), (in_channel, out_channel, nx) -> (batch, out_channel, ny)
        return tf.einsum("bixy,ioxy->boxy", inputs, weights)

    def call(self, A):
        A = tf.transpose(A, [0, 3, 1, 2])

        # Compute Fourier coeffcients
        A_ft = tf.signal.rfft2d(A)  # shape=(...,A.shape[-2],A.shape[-1]//2+1)

        # Multiply relevant Fourier modes
        out_ft_corner_SO = self.channel_mix(
            A_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft_corner_NO = self.channel_mix(
            A_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        out_ft_corner_SO = tf.pad(
            out_ft_corner_SO,
            [
                [0, 0],
                [0, 0],
                [0, A.shape[2] - self.modes1],
                [0, A_ft.shape[3] - self.modes2],
            ],
        )
        out_ft_corner_NO = tf.pad(
            out_ft_corner_NO,
            [
                [0, 0],
                [0, 0],
                [A.shape[2] - self.modes1, 0],
                [0, A_ft.shape[3] - self.modes2],
            ],
        )
        out_ft = out_ft_corner_SO + out_ft_corner_NO
        # Return to physical space
        A = tf.signal.irfft2d(out_ft, fft_length=[A.shape[2], A.shape[3]])

        # channel axis goes back at the end
        A = tf.transpose(A, [0, 2, 3, 1])

        return A


class FNO2d(tf.keras.Model):
    """Implementation of the FNO model."""

    def __init__(self, modes: int, width: int, pad_prop=0.05):
        """
        Initialize the FNO2d model.

        Parameters:
            modes (int): Number of Fourier modes to use in the spectral convolution.
            width (int): Width of the FNO2d model.
            pad_prop (float, optional): Proportion of padding to apply. Defaults to 0.05.
        """
        super().__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        self.pad_prop = pad_prop

        print(
            f"FNO2d model created with hyperparameters : modes:{modes}, width:{width}, pad_prop:{self.pad_prop} "
        )

        self.fc0 = tf.keras.layers.Dense(
            self.width
        )  # input channel is dim_a+1: (a(A), A)

        nb_layer = 4
        self.convs = [
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(nb_layer)
        ]
        self.addis = [
            tf.keras.layers.Conv2D(self.width, 1, padding="SAME")
            for _ in range(nb_layer)
        ]

        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, A):
        pad_1 = int(A.shape[1] * self.pad_prop)
        pad_2 = int(A.shape[2] * self.pad_prop)

        A = reflexive_padding_2d(A, pad_1, pad_2)

        A = self.fc0(A)

        for i, (layer, addi) in enumerate(zip(self.convs, self.addis)):
            x1 = layer.call(A)
            x2 = addi(A)
            A = x1 + x2
            if i != len(self.convs) - 1:
                A = tf.nn.gelu(A)

        A = A[:, pad_1:-pad_1, pad_2:-pad_2, :]
        A = self.fc1(A)
        A = tf.nn.gelu(A)
        A = self.fc2(A)

        return A


def test_spectral_conv_2D():
    """
    Test the SpectralConv2d layer.
    """
    batch_size = 1
    in_channels = 2
    out_channels = 3
    modes1 = 5
    modes2 = 6

    nx = 10 * modes1 + 1  # must be greater than modes
    ny = 5 * modes2

    spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)

    X = tf.ones([batch_size, nx, ny, in_channels])
    Y = spectral_conv.call(X)

    assert Y.shape == (
        batch_size,
        nx,
        ny,
        out_channels,
    ), f"Y.shape:{Y.shape} must be equal to (batch_size,nx,out_channels):{(batch_size,nx,ny, out_channels)} "

    print("X.shape", X.shape)
    print("Y.shape", Y.shape)


def test_FNO2d():
    """
    Test the FNO2d model.
    """
    batch_size = 7
    in_channels = 1
    width_model = 17
    modes = 5
    nx = 10 * modes
    ny = 5 * modes
    X = tf.ones([batch_size, nx, ny, in_channels])
    print("X.shape", X.shape)

    model = FNO2d(modes, width_model)
    Y = model.call(X)

    ti0 = time.time()
    for _ in range(10):
        Y = model.call(X)

    duration = time.time() - ti0
    print("duration:", duration)

    assert Y.shape == (
        batch_size,
        nx,
        ny,
        1,
    ), f"Y.shape:{Y.shape} must be equal to (batch_size,nx,1):{(batch_size,nx,ny, 1)} "

    print("Y.shape", Y.shape)


class Agent:
    """
    Agent class for training and evaluating the FNO2d model.
    """

    def __init__(self, data: DataLoader, small_model=False):
        """
        Initialize the Agent object.

        Parameters:
            data (DataLoader): DataLoader object for loading and processing data.
            small_model (bool, optional): Whether to use a smaller model. Defaults to False.
        """
        self.nb_vert = data.nb_vert
        self.data = data

        width = 20
        # for local tests
        if small_model:
            width = 10

        self.model = FNO2d(20, width, 0.05)
        self.learning_rate = 1e-3
        self.optimizer_misfit = tf.keras.optimizers.Adam(self.learning_rate)
        x = tf.linspace(0.0, 1.0, self.nb_vert)
        xx, yy = tf.meshgrid(x, x)
        self.x = tf.Variable(tf.reshape(xx, [-1]))
        self.y = tf.Variable(tf.reshape(yy, [-1]))

        self.best_possible_residues_interior = self.data.residues_interior_val

        self.memo_misfit_0_train = []
        self.memo_misfit_0_val = []
        self.memo_misfit_1_train = []
        self.memo_misfit_1_val = []
        self.memo_misfit_2_train = []
        self.memo_misfit_2_val = []

        self.memo_residues_interior_val = []

        self.memo_val_steps = []
        self.memo_optimize_misfit = []

        self.derivator_level1 = Derivator_fd(
            axes=[1, 2],
            interval_lenghts=[1, 1],
            derivative_symbols=[(0,), (1,)],
        )
        self.derivator_level2 = Derivator_fd(
            axes=[1, 2],
            interval_lenghts=[1, 1],
            derivative_symbols=[(0,), (1,), (0, 0), (1, 1), (0, 1)],
        )  # {"x":(0,),"y":(1,),"xx":(0,0),"yy":(1,1),"xy":(0,1)}

    def H_loss(self, Y_true, Y_pred, Phi, level, domain, G):
        nb_vert = np.shape(domain)[-1]
        domain_prop = tf.reduce_sum(domain, axis=[1, 2]) / nb_vert**2

        loss_0 = tf.reduce_mean(
            tf.reduce_mean(
                (
                    (
                        (Y_true * Phi[:, :, :, None] + G[:, :, :, None])
                        - (Y_pred * Phi[:, :, :, None] + G[:, :, :, None])
                    )
                    * domain[:, :, :, None]
                )
                ** 2,
                axis=[1, 2, 3],
            )
            / domain_prop[:]
        )

        if level == 0:
            return loss_0, np.nan, np.nan

        elif level == 1:
            Y_true_x, Y_true_y = self.derivator_level1(
                Y_true * Phi[:, :, :, None] + G[:, :, :, None]
            )
            Y_pred_x, Y_pred_y = self.derivator_level1(
                Y_pred * Phi[:, :, :, None] + G[:, :, :, None]
            )

            loss_1 = (
                tf.reduce_mean(
                    tf.reduce_mean(
                        ((Y_true_x - Y_pred_x) * domain[:, :, :, None]) ** 2,
                        axis=[1, 2, 3],
                    )
                    / domain_prop[:]
                )
            ) + (
                tf.reduce_mean(
                    tf.reduce_mean(
                        ((Y_true_y - Y_pred_y) * domain[:, :, :, None]) ** 2,
                        axis=[1, 2, 3],
                    )
                    / domain_prop[:]
                )
            )

            return loss_0, loss_1, np.nan

        else:
            (
                Y_true_x,
                Y_true_y,
                Y_true_xx,
                Y_true_yy,
                Y_true_xy,
            ) = self.derivator_level2(
                Y_true * Phi[:, :, :, None] + G[:, :, :, None]
            )
            (
                Y_pred_x,
                Y_pred_y,
                Y_pred_xx,
                Y_pred_yy,
                Y_pred_xy,
            ) = self.derivator_level2(
                Y_pred * Phi[:, :, :, None] + G[:, :, :, None]
            )

            loss_1 = (
                tf.reduce_mean(
                    tf.reduce_mean(
                        ((Y_true_x - Y_pred_x) * domain[:, :, :, None]) ** 2,
                        axis=[1, 2, 3],
                    )
                    / domain_prop[:]
                )
            ) + (
                tf.reduce_mean(
                    tf.reduce_mean(
                        ((Y_true_y - Y_pred_y) * domain[:, :, :, None]) ** 2,
                        axis=[1, 2, 3],
                    )
                    / domain_prop[:]
                )
            )

            loss_2 = (
                (
                    tf.reduce_mean(
                        tf.reduce_mean(
                            ((Y_true_xx - Y_pred_xx) * domain[:, :, :, None])
                            ** 2,
                            axis=[1, 2, 3],
                        )
                        / domain_prop[:]
                    )
                )
                + (
                    tf.reduce_mean(
                        tf.reduce_mean(
                            ((Y_true_xy - Y_pred_xy) * domain[:, :, :, None])
                            ** 2,
                            axis=[1, 2, 3],
                        )
                        / domain_prop[:]
                    )
                )
                + (
                    tf.reduce_mean(
                        tf.reduce_mean(
                            ((Y_true_yy - Y_pred_yy) * domain[:, :, :, None])
                            ** 2,
                            axis=[1, 2, 3],
                        )
                        / domain_prop[:]
                    )
                )
            )

            return loss_0, loss_1, loss_2

    @tf.function
    def validate(self):
        print("validate method tracing")
        Y_pred = self.model.call(self.data.X_val)
        misfit_0, misfit_1, misfit_2 = self.H_loss(
            self.data.Y_val,
            Y_pred,
            self.data.X_val[:, :, :, 1],
            level=2,
            domain=self.data.X_val[:, :, :, -1],
            G=self.data.X_val[:, :, :, 2],
        )
        residues_interior = self.data.compute_residues(self.data.X_val, Y_pred)
        return (
            misfit_0,
            misfit_1,
            misfit_2,
            residues_interior,
        )

    @tf.function
    def train_one_epoch_misfit_acc(self, X, Y, batch_size, nb_batch, level):
        """
        X = [F, Phi, G, domain]
        """
        print(f"tracing train_one_epoch_misfit_acc, level={level}")

        total_misfit_0 = 0
        total_misfit_1 = 0
        total_misfit_2 = 0

        for i in range(nb_batch):
            sli = slice(i * batch_size, (i + 1) * batch_size)
            X_, Y_ = X[sli], Y[sli]

            with tf.GradientTape() as tape:
                y_pred = self.model.call(X_)
                misfit_0, misfit_1, misfit_2 = self.H_loss(
                    Y_,
                    y_pred,
                    X_[:, :, :, 1],
                    level,
                    X_[:, :, :, -1],
                    X_[:, :, :, 2],
                )
                if level == 0:
                    misfit = misfit_0
                elif level == 1:
                    misfit = misfit_0 + misfit_1
                elif level == 2:
                    misfit = misfit_0 + misfit_1 + misfit_2
                else:
                    raise ValueError("Invalid level")

                tv = self.model.trainable_variables
                grad = tape.gradient(misfit, tv)
                self.optimizer_misfit.apply_gradients(zip(grad, tv))

            total_misfit_0 += misfit_0
            total_misfit_1 += misfit_1
            total_misfit_2 += misfit_2

        return (
            total_misfit_0 / nb_batch,
            total_misfit_1 / nb_batch,
            total_misfit_2 / nb_batch,
        )

    def scheduler(self, epoch, lr):
        if (epoch + 1) % 200 == 0.0:
            return lr / 2.0
        else:
            return lr

    def train(self, misfit_level, epochs):
        try:
            for i in range(epochs):
                self.learning_rate = self.scheduler(i, self.learning_rate)
                self.optimizer_misfit.lr.assign(self.learning_rate)
                self.memo_optimize_misfit.append(i)
                self.train_one_epoch(misfit_level, i)

                if i > 0 and (i + 1) % 10 == 0:
                    (
                        misfit_0,
                        misfit_1,
                        misfit_2,
                        residues_interior,
                    ) = self.validate()
                    self.memo_misfit_0_val.append(misfit_0)
                    self.memo_misfit_1_val.append(misfit_1)
                    self.memo_misfit_2_val.append(misfit_2)

                    self.memo_residues_interior_val.append(residues_interior)
                    print(
                        f"VAL:step:{i}, misfit_0:{misfit_0:.2e},misfit_1:{misfit_1:.2e},misfit_2:{misfit_2:.2e},residues_interior:{residues_interior:.2e}"
                    )
                    if (
                        not (os.path.exists(f"./models/best_model/"))
                        or (i + 1) == 10
                    ):
                        self.model.save_weights(
                            f"./models/best_model/model_weights"
                        )
                        file = open(f"models/best_model/best_epoch.txt", "w")
                        file.write(f"best_epoch = {i+1} \n")
                        (
                            best_misfits_0,
                            best_misfits_1,
                            best_misfits_2,
                            best_residues,
                        ) = (
                            self.memo_misfit_0_val[-1],
                            self.memo_misfit_1_val[-1],
                            self.memo_misfit_2_val[-1],
                            self.memo_residues_interior_val[-1],
                        )
                        if misfit_level == 0:
                            best_loss = best_misfits_0
                        elif misfit_level == 1:
                            best_loss = best_misfits_0 + best_misfits_1
                        else:
                            best_loss = (
                                best_misfits_0
                                + best_misfits_1
                                + best_misfits_2
                            )
                    else:
                        if misfit_level == 0:
                            loss = self.memo_misfit_0_val[-1]
                        elif misfit_level == 1:
                            loss = (
                                self.memo_misfit_0_val[-1]
                                + self.memo_misfit_1_val[-1]
                            )
                        else:
                            loss = (
                                self.memo_misfit_0_val[-1]
                                + self.memo_misfit_1_val[-1]
                                + self.memo_misfit_2_val[-1]
                            )
                        if loss < best_loss:
                            best_loss = loss
                            file = open(
                                f"models/best_model/best_epoch.txt", "w"
                            )
                            file.write(f"best_epoch = {i+1} \n")

                    self.memo_val_steps.append(i)
                    self.model.save_weights(
                        f"./models/model_{i+1}/model_weights"
                    )
                if (i + 1) % 50 == 0 and i > 0:
                    if not (os.path.exists(f"./models")):
                        os.makedirs(f"./models/")
                    if not (os.path.exists(f"./models/plots")):
                        os.makedirs(f"./models/plots/")

                    self.show_curves(save=True, i=i + 1, level=misfit_level)

        except KeyboardInterrupt:
            pass

        self.show_curves(level=misfit_level)

    def train_one_epoch(self, misfit_level, i):
        ti0 = time.time()
        batch_size = 64
        nb_data = len(self.data.X_train)
        nb_batch = nb_data // batch_size
        if nb_batch == 0:
            batch_size = nb_data
            nb_batch = 1
        if nb_batch <= 2:
            print(
                f"Warning, nb_data={nb_data} is small compared to batchsize={batch_size}, the number of batch per epoch is {nb_batch} "
            )

        rand_i = tf.random.shuffle(range(len(self.data.X_train)))
        X = tf.gather(self.data.X_train, rand_i)
        Y = tf.gather(self.data.Y_train, rand_i)

        misfit_0, misfit_1, misfit_2 = (
            np.nan,
            np.nan,
            np.nan,
        )

        misfit_0, misfit_1, misfit_2 = self.train_one_epoch_misfit_acc(
            X, Y, batch_size, nb_batch, misfit_level
        )

        self.memo_misfit_0_train.append(misfit_0)
        self.memo_misfit_1_train.append(misfit_1)
        self.memo_misfit_2_train.append(misfit_2)
        print(
            f"step:{i},misfit_0:{misfit_0:.2e},misfit_1:{misfit_1:.2e},misfit_2:{misfit_2:.2e},duration:{time.time()-ti0}"
        )

    def show_curves(self, save=False, i=0, level=2):
        plt.figure(figsize=(8, 6))
        plt.plot(self.memo_misfit_0_train, "k", label="misfit_0")
        plt.plot(self.memo_val_steps, self.memo_misfit_0_val, "k.")

        if level >= 1:
            plt.plot(self.memo_misfit_1_train, "b", label="misfit_1")
            plt.plot(self.memo_val_steps, self.memo_misfit_1_val, "b.")

        if level >= 2:
            plt.plot(self.memo_misfit_2_train, "c", label="misfit_2")
            plt.plot(self.memo_val_steps, self.memo_misfit_2_val, "c.")

        plt.title("Misfits")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"./models/plots/misfits_{i}.png")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(
            self.memo_val_steps,
            self.memo_residues_interior_val,
            "k.",
            label="residues",
        )
        plt.title(
            f"$\phi$-FEM residues : {self.best_possible_residues_interior:.1e}"
        )
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"./models/plots/residues_{i}.png")
        plt.show()

    def show_results(self):
        nb = 5
        X_val, Y_val = self.data.X_val[:nb], self.data.Y_val[:nb]
        Y_pred = self.model.call(X_val)
        F = X_val[:, :, :, 0] * self.data.max_norm_F
        U_fe = Y_val[:, :, :, 0] * X_val[:, :, :, 1] + X_val[:, :, :, 2]
        U_pi = Y_pred[:, :, :, 0] * X_val[:, :, :, 1] + X_val[:, :, :, 2]

        plot_laplacian(F, U_fe, U_pi, X_val[:, :, :, -1])
        plot_results(U_fe, U_pi, X_val[:, :, :, -1])


def plot_results(U_fe, U_pi, domain):
    nb = U_fe.shape[0]

    fig, axs = plt.subplots(nb, 3, figsize=(6, 2 * nb))

    U_fe_d = U_fe * domain
    U_pi_d = U_pi * domain
    delta = U_fe_d - U_pi_d

    vmin_pi = tf.reduce_min(U_pi_d).numpy()
    vmax_pi = tf.reduce_max(U_pi_d).numpy()
    print(
        f"for U_pi[:nb] with nb={nb}, on disk, min,max values are",
        vmin_pi,
        vmax_pi,
    )

    vmin_fe = tf.reduce_min(U_fe_d).numpy()
    vmax_fe = tf.reduce_max(U_fe_d).numpy()
    print(
        f"for U_fe[:nb] with nb={nb}, on disk, min,max values are",
        vmin_fe,
        vmax_fe,
    )

    vmin = np.min([vmin_fe, vmin_pi])
    vmax = np.max([vmax_fe, vmax_pi])

    im = None
    for i in range(nb):
        for j in [0, 1, 2]:
            axs[i, j].axis("off")
        im = axs[i, 0].imshow(
            U_fe_d[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )
        axs[i, 1].imshow(
            U_pi_d[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )
        axs[i, 2].imshow(
            delta[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )

    axs[0, 0].set_title("U_fe")
    axs[0, 1].set_title("U_pi")
    axs[0, 2].set_title("U_fe - U_pi")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def plot_laplacian(F, U_fe, U_pi, domain):
    if U_pi is None:
        U_pi = tf.zeros_like(U_fe)

    pre_laplacian = Derivator_fd(
        [(0, 0), (1, 1)], interval_lenghts=[1, 1], axes=[1, 2]
    )

    def laplacian(U):
        Uxx, Uyy = pre_laplacian(U)
        return Uxx + Uyy

    ti0 = time.time()
    delta_U_fe = laplacian(U_fe)
    delta_U_pi = laplacian(U_pi)

    print("duration:", time.time() - ti0)

    _deltaU_fe_d = -delta_U_fe * domain
    _deltaU_pi_d = -delta_U_pi * domain
    F_d = F * domain

    vmin_fe = tf.reduce_min(_deltaU_fe_d).numpy()
    vmax_fe = tf.reduce_max(_deltaU_fe_d).numpy()

    nb = F.shape[0]
    print(
        f"for - div grad U_fe[:nb] with nb={nb}, on disk, min,max values are",
        vmin_fe,
        vmax_fe,
    )

    vmin_pi = tf.reduce_min(_deltaU_pi_d).numpy()
    vmax_pi = tf.reduce_max(_deltaU_pi_d).numpy()
    print(
        f"for - div grad U_pi[:nb] with nb={nb}, on disk, min,max values are",
        vmin_pi,
        vmax_pi,
    )

    vmin_f = tf.reduce_min(F_d).numpy()
    vmax_f = tf.reduce_max(F_d).numpy()
    print(
        f"for F[:nb] with nb={nb}, on disk, min,max values are", vmin_f, vmax_f
    )

    vmin = np.min([vmin_f, vmin_fe, vmin_pi])
    vmax = np.max([vmax_f, vmax_fe, vmax_pi])

    fig, axs = plt.subplots(nb, 3, figsize=(6, 2 * nb))
    im = None
    for i in range(nb):
        for j in [0, 1, 2]:
            axs[i, j].axis("off")
        axs[i, 0].imshow(
            F_d[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )
        im = axs[i, 1].imshow(
            _deltaU_fe_d[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )
        axs[i, 2].imshow(
            _deltaU_pi_d[i], vmin=vmin, vmax=vmax, cmap=my_cmap, origin="lower"
        )

    axs[0, 0].set_title("F")
    axs[0, 1].set_title("-delta (U_fe)")
    axs[0, 2].set_title("-delta (U_pi)")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
